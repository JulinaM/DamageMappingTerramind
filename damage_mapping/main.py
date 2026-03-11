import os
import pathlib
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from damage_mapping.Evaluator import Evaluator
from damage_mapping.Trainer import Trainer
from damage_mapping.datasets.DataLoader import TestLoader, Train_Val_Loader
from damage_mapping.logger import init_logger
from damage_mapping.models.Decoder_UNet2D import UNet2D
from damage_mapping.models.Encoder_TerraMind import TerraMindEncoder
from damage_mapping.models.utils import set_seeds, weights

REPO_DIR = pathlib.Path(__file__).parent.parent
WORK_DIR = REPO_DIR / "data/input"
EXPERIMENT_DIR = REPO_DIR / "data/experiments"
CONFIG_DIR = REPO_DIR / "configs/"


def build_loader(loader_cfg: DictConfig, split: str) -> DataLoader:
    modalities = {name: (paths.before, paths.after) for name, paths in loader_cfg.modalities.items()}
    dataset = Train_Val_Loader(
        modalities=modalities,
        label_dir=loader_cfg.label_dir,
        split=split,
        num_augmentations=getattr(loader_cfg, "num_augmentations", 0),
        patch_size=loader_cfg.patch_size,
        stride=loader_cfg.stride,
        preload=loader_cfg.preload,
    )
    return DataLoader(
        dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=loader_cfg.shuffle,
        num_workers=getattr(loader_cfg, "num_workers", 0),
    )


def build_holdout_loader(cfg: DictConfig) -> DataLoader | None:
    holdout_cfg = cfg.get("holdout_loader")
    if holdout_cfg is None:
        return None

    modalities = {name: (paths.before, paths.after) for name, paths in holdout_cfg.modalities.items()}
    dataset = TestLoader(
        modalities=modalities,
        label_dir=holdout_cfg.label_dir,
        patch_size=holdout_cfg.patch_size,
        stride=holdout_cfg.stride,
    )
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=getattr(holdout_cfg, "num_workers", 0),
    )


def build_model_components(cfg: DictConfig, device: torch.device, train_loader: DataLoader):
    if cfg.model.apply_weight_loss:
        inverse, pixels = weights(
            train_loader,
            num_classes=cfg.model.num_classes,
            ignore_index=cfg.model.ignore_index,
            device=device,
        )
        if cfg.model.weight_type == "pixels":
            criterion = nn.CrossEntropyLoss(weight=pixels, ignore_index=cfg.model.ignore_index)
        elif cfg.model.weight_type == "inverse":
            criterion = nn.CrossEntropyLoss(weight=inverse, ignore_index=cfg.model.ignore_index)
        else:
            raise ValueError(f"Invalid weight_type: {cfg.model.weight_type}")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=cfg.model.ignore_index)

    encoder = TerraMindEncoder(
        version=cfg.model.TM_version,
        pretrained=cfg.model.pretrained,
        modalities=list(cfg.model.modalities),
    )
    decoder = UNet2D(num_classes=cfg.model.num_classes)
    encoder.to(device)
    decoder.to(device)

    if cfg.model.TM_finetune:
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.model.learning_rate)
    else:
        optimizer = optim.Adam(decoder.parameters(), lr=cfg.model.learning_rate)

    return encoder, decoder, criterion, optimizer


@hydra.main(version_base="1.2", config_path=str(CONFIG_DIR), config_name="terramind")
def main(cfg: DictConfig):
    set_seeds(cfg.model.seed)

    distributed = bool(getattr(cfg, "distributed", False))

    if distributed:
        torch.distributed.init_process_group(backend="nccl")
    else:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    hc = HydraConfig.get()
    # exp_dir = Path(os.path.join(hc["sweep"]["dir"], hc["sweep"]["subdir"])) if "sweep" in hc else Path(hc.runtime.output_dir)
    exp_dir = Path(hc.runtime.output_dir)
    exp_name = exp_dir.name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger_path = exp_dir / "out.log"

    if cfg.use_wandb and rank == 0:
        import wandb
        wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project="damage_mapping_agri",
            name=exp_name,
            dir=str(exp_dir),
            config=wandb_cfg,
        )

    logger = init_logger(logger_path, rank=rank)
    logger.info("Experiment name: %s", exp_name)
    logger.info("Device name: %s", device)
    logger.info("The experiment is stored in %s", exp_dir)

    train_loader = build_loader(cfg.train_loader, "train")
    val_loader = build_loader(cfg.validation_loader, "validation")
    holdout_loader = build_holdout_loader(cfg)
    encoder, decoder, criterion, optimizer = build_model_components(cfg, device, train_loader)
    trainer = Trainer(
        cfg=cfg,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        encoder=encoder,
        decoder=decoder,
        criterion=criterion,
        optimizer=optimizer,
        logger=logger,
        use_wandb=cfg.use_wandb,
    )
    evaluator = Evaluator(
        cfg=cfg,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        device=device,
        dataloader=holdout_loader,
        logger=logger,
    )

    trainer.train()
    evaluator.evaluate()

    if distributed:
        torch.distributed.destroy_process_group()

    if cfg.use_wandb and rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
