import logging
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from damage_mapping.models.utils import calc_batch_metrics, calc_epoch_metrics, move_to_device, save_checkpoint


class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        exp_dir: str | Path,
        ckpt_dir: str | Path,
        device: str | torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        encoder: nn.Module,
        change_fusion: nn.Module,
        decoder: nn.Module,
        criterion: nn.Module,
        optimizer,
        logger: logging.Logger | None = None,
        use_wandb: bool = False,
        curriculum_manager=None,
    ) -> None:
        self.cfg = cfg
        self.exp_dir = Path(exp_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.device = torch.device(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.encoder = encoder
        self.change_fusion = change_fusion
        self.decoder = decoder
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger or logging.getLogger(__name__)
        self.use_wandb = use_wandb
        self.curriculum_manager = curriculum_manager
        self.writer = SummaryWriter(log_dir=str(self.exp_dir))

        self.model_cfg   = cfg.model
        self.encoder_cfg = cfg.encoder
        self.train_cfg   = cfg.train_loader
        self.val_cfg     = cfg.validation_loader
        self.trainer_cfg = cfg.trainer

        self.n_epochs = int(getattr(self.trainer_cfg, "n_epochs", self.model_cfg.num_epochs))

        # Stage tracking: "flood" during Stage 1, "conflict" in Stage 2 / no-curriculum
        self.current_stage: str = "flood" if curriculum_manager is not None else "conflict"

        # Independent best-checkpoint tracking per stage.
        # Stage 1 ("flood") and Stage 2 ("conflict") losses are on different distributions and must NOT be compared against each other.
        self._stage_best: dict[str, dict] = {
            "flood":    {"val_loss": float("inf"), "metrics": None, "epoch": None},
            "conflict": {"val_loss": float("inf"), "metrics": None, "epoch": None},
        }

        # _last_val_metrics holds the most recent epoch's metrics dict so
        # _save_best_checkpoint can write it into _stage_best without re-computing.
        self._last_val_metrics: dict[str, float] | None = None
        encoder_name = str(getattr(self.encoder_cfg, "name", "Terramind")).strip().lower()
        self.encoder_mode = self.encoder.train if (encoder_name == "unet" or bool(getattr(self.encoder_cfg, "finetune", False))) else self.encoder.eval

        if self.use_wandb:
            import wandb
            self.wandb = wandb


    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self) -> float:
        if self.curriculum_manager is not None:
            cfg_n_epochs = int(getattr(self.trainer_cfg, "n_epochs", self.model_cfg.num_epochs))
            self.n_epochs = self.curriculum_manager.total_epochs
            if cfg_n_epochs != self.n_epochs:
                self.logger.warning(
                    "trainer.n_epochs=%d is overridden by curriculum total "
                    "(flood_epochs=%d + conflict_epochs=%d = %d). "
                    "Edit curriculum.flood_epochs / conflict_epochs to control total training length.",
                    cfg_n_epochs,
                    self.curriculum_manager.flood_epochs,
                    self.curriculum_manager.conflict_epochs,
                    self.n_epochs,
                )

        self.logger.info("Trainer started")
        self.logger.info("Output directory: %s", self.exp_dir)
        self.logger.info("Device: %s", self.device)
        self.logger.info("Model config: %s", OmegaConf.to_container(self.model_cfg, resolve=True))
        self.logger.info("Train loader config (conflict): %s", OmegaConf.to_container(self.train_cfg, resolve=True))
        self.logger.info("Validation loader config (conflict): %s", OmegaConf.to_container(self.val_cfg, resolve=True))
        self.logger.info("Criterion config: %s", OmegaConf.to_container(self.cfg.criterion, resolve=True))
        self.logger.info("Train patches: %d", len(self.train_loader.dataset))
        self.logger.info("Validation patches: %d", len(self.val_loader.dataset))
        self.logger.info("Train batches: %d", len(self.train_loader))
        self.logger.info("Validation batches: %d", len(self.val_loader))
        self.logger.info("Starting training for %d epoch(s)", self.n_epochs)

        if self.curriculum_manager is not None:
            cm = self.curriculum_manager
            self.logger.info(
                "Curriculum: flood stage %d ep (lr=%.2e) → conflict stage %d ep (lr=%.2e)",
                cm.flood_epochs, self.optimizer.param_groups[0]["lr"],
                cm.conflict_epochs, cm.stage2_lr,
            )
            # Log both dataset sizes upfront so the user sees total scope at a glance
            self.logger.info(
                "Flood   train=%d patches | val=%s patches",
                len(cm.flood_loader.dataset),
                len(cm.flood_val_loader.dataset) if cm.flood_val_loader is not None else "N/A (using conflict val)",
            )
            self.logger.info(
                "Conflict train=%d patches | val=%s patches",
                len(cm.conflict_loader.dataset),
                len(cm.conflict_val_loader.dataset) if cm.conflict_val_loader is not None else "N/A",
            )

        try:
            for epoch in range(self.n_epochs):
                if self.curriculum_manager is not None:
                    self._apply_curriculum_update(epoch)
                train_loss = self._train_one_epoch(epoch)
                val_loss, val_metrics = self.validate()
                self._last_val_metrics = val_metrics
                self._log_epoch(epoch, train_loss, val_loss, val_metrics)
                self._save_best_checkpoint(epoch, val_loss)
                self._write_tensorboard(epoch, train_loss, val_loss, val_metrics)
                self._write_wandb(epoch, train_loss, val_loss, val_metrics)
            self.logger.info("Training completed successfully")
        except Exception:
            self.logger.exception("Trainer failed")
            raise
        finally:
            self.writer.close()
            self.logger.info("Closed TensorBoard writer")

        # Return best conflict-stage IoU (the stage used by the Evaluator).
        # Fall back to flood if conflict never improved (e.g. conflict_epochs=0).
        final_metrics = (self._stage_best["conflict"]["metrics"] or self._stage_best["flood"]["metrics"])
        if final_metrics is None:
            raise RuntimeError("Trainer completed without recording best validation metrics.")
        return float(final_metrics["IoU"])


    def _apply_curriculum_update(self, epoch: int) -> None:
        """Swap train/val loaders and adjust LR at the stage boundary."""
        epoch_1 = epoch + 1  # CurriculumDataManager uses 1-indexed epochs

        self.current_stage = self.curriculum_manager.get_stage(epoch_1)
        self.train_loader  = self.curriculum_manager.get_loader(epoch_1)

        new_val = self.curriculum_manager.get_val_loader(epoch_1)
        if new_val is not None:
            self.val_loader = new_val

        # Warn early (epoch 1 only) if flood validation is falling back to conflict set
        if epoch_1 == 1 and self.curriculum_manager.flood_val_loader is None:
            self.logger.warning(
                "flood_validation_loader not configured: Stage 1 validation metrics "
                "will be computed on the conflict validation set. "
                "Add flood_validation_loader to your config for stage-matched validation."
            )

        if self.curriculum_manager.should_switch(epoch_1):
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.curriculum_manager.stage2_lr

            # Criterion class weights were computed once from the flood loader at
            # startup. If conflict and flood label distributions differ significantly,
            # Stage 2 loss weighting will be suboptimal. Set criterion.apply_weight_loss:
            # false in the config, or recompute the criterion externally, to avoid this.
            if getattr(getattr(self.cfg, "criterion", None), "apply_weight_loss", False):
                self.logger.warning(
                    "Curriculum stage switch: criterion class weights were computed from "
                    "the flood dataset and will remain in effect for Stage 2 (conflict). "
                    "If class distributions differ between stages, consider setting "
                    "criterion.apply_weight_loss: false or recomputing weights manually."
                )

            flood_best = self._stage_best["flood"]
            self.logger.info(
                "Curriculum: flood → conflict at epoch %d | lr → %.2e | "
                "flood best: epoch=%s val_loss=%.4f IoU=%.4f",
                epoch_1,
                self.curriculum_manager.stage2_lr,
                flood_best["epoch"],
                flood_best["val_loss"],
                (flood_best["metrics"] or {}).get("IoU", float("nan")),
            )
            if self.curriculum_manager.conflict_val_loader is not None:
                self.logger.info(
                    "Val loader swapped to conflict validation set (%d patches)",
                    len(self.val_loader.dataset),
                )
            else:
                self.logger.info(
                    "Val loader unchanged (no conflict_val_loader configured; "
                    "continuing with current val set, %d patches)",
                    len(self.val_loader.dataset),
                )


    def validate(self) -> tuple[float, dict[str, float]]:
        self.encoder.eval()
        self.change_fusion.eval()
        self.decoder.eval()

        running_val_loss = 0.0
        true_positive = false_positive = false_negative = true_negative = 0.0

        with torch.no_grad():
            for inputs, target in self.val_loader:
                inputs = move_to_device(inputs, self.device)
                target = target.to(self.device)

                logits = self._forward(inputs)
                batch_loss = self.criterion(logits, target)
                batch_size = next(iter(inputs["before"].values())).size(0)
                running_val_loss += batch_loss.item() * batch_size

                batch_metrics = calc_batch_metrics(
                    logits,
                    target,
                    ignore_index=self.model_cfg.ignore_index,
                    positive_class=self.model_cfg.positive_class,
                    negative_class=self.model_cfg.negative_class,
                )
                true_positive  += batch_metrics[0]
                false_positive += batch_metrics[1]
                false_negative += batch_metrics[2]
                true_negative  += batch_metrics[3]

        n_val = len(self.val_loader.dataset)
        if n_val == 0:
            raise RuntimeError(
                f"[{self.current_stage}] Validation dataset is empty. "
                "Check that the val directory contains images and that patch_size/stride "
                "produce at least one patch per image."
            )
        val_loss = running_val_loss / n_val
        metrics  = calc_epoch_metrics(true_positive, false_positive, false_negative, true_negative)
        return val_loss, metrics


    def _train_one_epoch(self, epoch: int) -> float:
        self.encoder_mode()
        self.change_fusion.train()
        self.decoder.train()
        running_train_loss = 0.0
        num_batches = len(self.train_loader)
        stage_tag = f"[{self.current_stage.upper()}] " if self.curriculum_manager is not None else ""

        for batch_idx, (inputs, target) in enumerate(self.train_loader, start=1):
            inputs = move_to_device(inputs, self.device)
            target = target.to(self.device)

            logits = self._forward(inputs)
            train_loss = self.criterion(logits, target)
            batch_size = next(iter(inputs["before"].values())).size(0)
            running_train_loss += train_loss.item() * batch_size

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            if batch_idx % getattr(self.trainer_cfg, "log_interval", 1) == 0:
                self.logger.info(
                    "%sEpoch %d/%d | batch %d/%d | train_loss=%.4f",
                    stage_tag, epoch + 1, self.n_epochs,
                    batch_idx, num_batches, train_loss.item(),
                )

            if self.use_wandb:
                global_step = epoch * num_batches + batch_idx
                stage = self.current_stage
                self.wandb.log({
                    "train/global_step":            global_step,
                    f"train/{stage}/batch_loss":    train_loss.item(),
                    "train/learning_rate":          self.optimizer.param_groups[0]["lr"],
                    "train/epoch":                  epoch + 1,
                    "train/stage":                  stage,
                })

        n_train = len(self.train_loader.dataset)
        if n_train == 0:
            raise RuntimeError(
                f"[{self.current_stage}] Training dataset is empty. "
                "Check that the train directory contains images and that patch_size/stride "
                "produce at least one patch per image."
            )
        return running_train_loss / n_train


    def _forward(self, inputs: dict) -> torch.Tensor:
        z_before = self.encoder(inputs["before"])
        z_after  = self.encoder(inputs["after"])
        fused_features = self.change_fusion(z_before, z_after)
        return self.decoder(fused_features)


    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float, metrics: dict[str, float]) -> None:
        stage_tag = f"[{self.current_stage.upper()}] " if self.curriculum_manager is not None else ""
        self.logger.info(
            "%sEpoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
            "IoU=%.4f | Acc=%.4f | Prec=%.4f | Recall=%.4f | F1=%.4f",
            stage_tag, epoch + 1, self.n_epochs,
            train_loss, val_loss,
            metrics["IoU"], metrics["Accuracy"],
            metrics["Precision"], metrics["Recall"], metrics["F1"],
        )


    def _save_best_checkpoint(self, epoch: int, val_loss: float) -> None:
        stage     = self.current_stage
        stage_rec = self._stage_best[stage]

        if val_loss >= stage_rec["val_loss"]:
            return

        stage_rec["val_loss"] = val_loss
        stage_rec["epoch"]    = epoch
        stage_rec["metrics"]  = self._last_val_metrics

        # Distinct filename prefix per stage keeps Stage-1 and Stage-2
        # checkpoints from deleting each other.
        ckpt_prefix = "best_flood" if stage == "flood" else "best"
        save_checkpoint(
            self.encoder, self.change_fusion, self.decoder, self.optimizer,
            epoch, val_loss, self.cfg,
            save_dir=str(self.ckpt_dir),
            prefix=ckpt_prefix,
        )
        self.logger.info(
            "[%s] New best checkpoint at epoch %d | val_loss=%.4f | IoU=%.4f",
            stage.upper(), epoch + 1, val_loss,
            (self._last_val_metrics or {}).get("IoU", float("nan")),
        )


    def _write_tensorboard(self, epoch: int, train_loss: float,
                           val_loss: float, metrics: dict[str, float]) -> None:
        # When curriculum is active use stage-prefixed tags so flood and
        # conflict curves appear as separate series in TensorBoard.
        # Without curriculum the original flat tag names are preserved.
        if self.curriculum_manager is not None:
            s = self.current_stage
            tag = lambda base: f"{s}/{base}"
        else:
            tag = lambda base: base

        self.writer.add_scalar(tag("Loss/train"),       train_loss,           epoch)
        self.writer.add_scalar(tag("Loss/validation"),  val_loss,             epoch)
        self.writer.add_scalar(tag("Metrics/IoU"),      metrics["IoU"],       epoch)
        self.writer.add_scalar(tag("Metrics/Accuracy"), metrics["Accuracy"],  epoch)
        self.writer.add_scalar(tag("Metrics/Precision"),metrics["Precision"], epoch)
        self.writer.add_scalar(tag("Metrics/Recall"),   metrics["Recall"],    epoch)
        self.writer.add_scalar(tag("Metrics/F1"),       metrics["F1"],        epoch)


    def _write_wandb(self, epoch: int, train_loss: float,
                     val_loss: float, metrics: dict[str, float]) -> None:
        if not self.use_wandb:
            return

        stage = self.current_stage
        # With curriculum: keys become e.g. "val/flood/loss", "best/conflict/IoU".
        # Without curriculum: s="" so keys stay "val/loss", "best/IoU" (backward compat).
        s = f"{stage}/" if self.curriculum_manager is not None else ""

        payload = {
            "val/epoch":               epoch + 1,
            f"train/{s}epoch_loss":    train_loss,
            f"val/{s}loss":            val_loss,
            f"val/{s}IoU":             metrics["IoU"],
            f"val/{s}Accuracy":        metrics["Accuracy"],
            f"val/{s}Precision":       metrics["Precision"],
            f"val/{s}Recall":          metrics["Recall"],
            f"val/{s}F1":              metrics["F1"],
            f"best/{s}val_loss":       self._stage_best[stage]["val_loss"],
        }

        stage_metrics = self._stage_best[stage]["metrics"]
        if stage_metrics is not None:
            payload[f"best/{s}IoU"]       = stage_metrics["IoU"]
            payload[f"best/{s}Accuracy"]  = stage_metrics["Accuracy"]
            payload[f"best/{s}Precision"] = stage_metrics["Precision"]
            payload[f"best/{s}Recall"]    = stage_metrics["Recall"]
            payload[f"best/{s}F1"]        = stage_metrics["F1"]

        self.wandb.log(payload)
