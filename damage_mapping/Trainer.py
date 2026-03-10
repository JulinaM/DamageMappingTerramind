import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from damage_mapping.datasets.DataLoader import Train_Val_Loader
from damage_mapping.models.Decoder_UNet2D import UNet2D
from damage_mapping.models.Encoder_TerraMind import TerraMindEncoder
from damage_mapping.models.utils import (
    calc_batch_metrics,
    calc_epoch_metrics,
    move_to_device,
    save_checkpoint,
    weights,
)


class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        exp_dir: str | Path,
        ckpt_dir: str | Path,
        device: str | torch.device,
        logger: logging.Logger | None = None,
        use_wandb: bool = False,
    ) -> None:
        self.cfg = cfg
        self.exp_dir = Path(exp_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.device = torch.device(device)
        self.logger = logger or logging.getLogger(__name__)
        self.use_wandb = use_wandb
        self.writer = SummaryWriter(log_dir=str(self.exp_dir))

        self.model_cfg = cfg.model
        self.train_cfg = cfg.train_loader
        self.val_cfg = cfg.validation_loader
        self.trainer_cfg = cfg.trainer

        self.n_epochs = int(getattr(self.trainer_cfg, "n_epochs", self.model_cfg.num_epochs))
        self.best_val_loss = float("inf")

        self.train_loader, self.train_dataset = self._build_train_loader()
        self.val_loader, self.val_dataset = self._build_validation_loader()
        self.criterion = self._build_criterion()
        self.encoder, self.decoder, self.optimizer, self.encoder_mode = self._build_training_components()

    def train(self) -> None:
        self.logger.info("Trainer started")
        self.logger.info("Output directory: %s", self.exp_dir)
        self.logger.info("Device: %s", self.device)
        self.logger.info("Model config: %s", OmegaConf.to_container(self.model_cfg, resolve=True))
        self.logger.info("Train loader config: %s", OmegaConf.to_container(self.train_cfg, resolve=True))
        self.logger.info("Validation loader config: %s", OmegaConf.to_container(self.val_cfg, resolve=True))
        self.logger.info("Starting training for %d epoch(s)", self.n_epochs)

        try:
            for epoch in range(self.n_epochs):
                train_loss = self._train_one_epoch(epoch)
                val_loss, val_metrics = self.validate()
                self._log_epoch(epoch, train_loss, val_loss, val_metrics)
                self._save_best_checkpoint(epoch, val_loss)
                self._write_tensorboard(epoch, train_loss, val_loss, val_metrics)
            self.logger.info("Training completed successfully")
        except Exception:
            self.logger.exception("Trainer failed")
            raise
        finally:
            self.writer.close()
            self.logger.info("Closed TensorBoard writer")

    def validate(self) -> tuple[float, dict[str, float]]:
        self.encoder.eval()
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
                true_positive += batch_metrics[0]
                false_positive += batch_metrics[1]
                false_negative += batch_metrics[2]
                true_negative += batch_metrics[3]

        val_loss = running_val_loss / len(self.val_dataset)
        metrics = calc_epoch_metrics(
            true_positive,
            false_positive,
            false_negative,
            true_negative,
        )
        return val_loss, metrics

    def _build_train_loader(self) -> tuple[DataLoader, Train_Val_Loader]:
        train_modalities = {
            name: (paths.before, paths.after) for name, paths in self.train_cfg.modalities.items()
        }
        dataset = Train_Val_Loader(
            modalities=train_modalities,
            label_dir=self.train_cfg.label_dir,
            split="train",
            num_augmentations=self.train_cfg.num_augmentations,
            patch_size=self.train_cfg.patch_size,
            stride=self.train_cfg.stride,
            preload=self.train_cfg.preload,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=self.train_cfg.shuffle,
            num_workers=getattr(self.train_cfg, "num_workers", 0),
        )
        self.logger.info("Train patches: %d", len(dataset))
        return loader, dataset

    def _build_validation_loader(self) -> tuple[DataLoader, Train_Val_Loader]:
        val_modalities = {
            name: (paths.before, paths.after) for name, paths in self.val_cfg.modalities.items()
        }
        dataset = Train_Val_Loader(
            modalities=val_modalities,
            label_dir=self.val_cfg.label_dir,
            split="validation",
            patch_size=self.val_cfg.patch_size,
            stride=self.val_cfg.stride,
            preload=self.val_cfg.preload,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.val_cfg.batch_size,
            shuffle=self.val_cfg.shuffle,
            num_workers=getattr(self.val_cfg, "num_workers", 0),
        )
        self.logger.info("Validation patches: %d", len(dataset))
        return loader, dataset

    def _build_criterion(self) -> nn.Module:
        if not self.model_cfg.apply_weight_loss:
            return nn.CrossEntropyLoss(ignore_index=self.model_cfg.ignore_index)

        inverse, pixels = weights(
            self.train_loader,
            num_classes=self.model_cfg.num_classes,
            ignore_index=self.model_cfg.ignore_index,
            device=self.device,
        )
        if self.model_cfg.weight_type == "pixels":
            return nn.CrossEntropyLoss(weight=pixels, ignore_index=self.model_cfg.ignore_index)
        if self.model_cfg.weight_type == "inverse":
            return nn.CrossEntropyLoss(weight=inverse, ignore_index=self.model_cfg.ignore_index)
        raise ValueError(f"Invalid weight_type: {self.model_cfg.weight_type}")

    def _build_training_components(
        self,
    ) -> tuple[TerraMindEncoder, UNet2D, optim.Optimizer, callable]:
        encoder = TerraMindEncoder(
            version=self.model_cfg.TM_version,
            pretrained=self.model_cfg.pretrained,
            modalities=list(self.model_cfg.modalities),
        )
        decoder = UNet2D(num_classes=self.model_cfg.num_classes)
        encoder.to(self.device)
        decoder.to(self.device)

        if self.model_cfg.TM_finetune:
            optimizer = optim.Adam(
                list(encoder.parameters()) + list(decoder.parameters()),
                lr=self.model_cfg.learning_rate,
            )
            encoder_mode = encoder.train
            self.logger.info("Fine-tuning encoder + decoder")
        else:
            optimizer = optim.Adam(decoder.parameters(), lr=self.model_cfg.learning_rate)
            encoder_mode = encoder.eval
            self.logger.info("Training decoder only (encoder frozen)")

        return encoder, decoder, optimizer, encoder_mode

    def _train_one_epoch(self, epoch: int) -> float:
        self.encoder_mode()
        self.decoder.train()
        running_train_loss = 0.0

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
                    "Epoch %d/%d | batch %d/%d | train_loss=%.4f",
                    epoch + 1,
                    self.n_epochs,
                    batch_idx,
                    len(self.train_loader),
                    train_loss.item(),
                )

        return running_train_loss / len(self.train_dataset)

    def _forward(self, inputs: dict) -> torch.Tensor:
        z_before = self.encoder(inputs["before"])
        z_after = self.encoder(inputs["after"])
        z_differenced = [after - before for before, after in zip(z_before, z_after)]
        return self.decoder(z_differenced)

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float, metrics: dict[str, float]) -> None:
        self.logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | IoU=%.4f | Acc=%.4f | Prec=%.4f | Recall=%.4f | F1=%.4f",
            epoch + 1,
            self.n_epochs,
            train_loss,
            val_loss,
            metrics["IoU"],
            metrics["Accuracy"],
            metrics["Precision"],
            metrics["Recall"],
            metrics["F1"],
        )

    def _save_best_checkpoint(self, epoch: int, val_loss: float) -> None:
        if val_loss >= self.best_val_loss:
            return

        self.best_val_loss = val_loss
        save_checkpoint(
            self.encoder,
            self.decoder,
            self.optimizer,
            epoch,
            val_loss,
            self.cfg,
            save_dir=str(self.ckpt_dir),
        )
        self.logger.info("Saved new best checkpoint at epoch %d with val_loss=%.4f", epoch + 1, val_loss)

    def _write_tensorboard(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: dict[str, float],
    ) -> None:
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/validation", val_loss, epoch)
        self.writer.add_scalar("Metrics/IoU", metrics["IoU"], epoch)
        self.writer.add_scalar("Metrics/Accuracy", metrics["Accuracy"], epoch)
        self.writer.add_scalar("Metrics/Precision", metrics["Precision"], epoch)
        self.writer.add_scalar("Metrics/Recall", metrics["Recall"], epoch)
        self.writer.add_scalar("Metrics/F1", metrics["F1"], epoch)
