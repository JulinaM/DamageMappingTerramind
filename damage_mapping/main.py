import os
import pathlib
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from damage_mapping.Evaluator import Evaluator
from damage_mapping.Trainer import Trainer
from damage_mapping.logger import init_logger
from damage_mapping.models.utils import set_seeds

REPO_DIR = pathlib.Path(__file__).parent.parent
WORK_DIR = REPO_DIR / "data/input"
EXPERIMENT_DIR = REPO_DIR / "data/experiments"
CONFIG_DIR = REPO_DIR / "configs/"


@hydra.main(version_base="1.2", config_path=str(CONFIG_DIR), config_name="terramind")
def main(cfg: DictConfig):
    set_seeds(cfg.model.seed)
    job_type = str(getattr(cfg, "job_type", "train_eval")).strip().lower()

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
    logger.info("Job type: %s", job_type)
    logger.info("Device name: %s", device)
    logger.info("The experiment is stored in %s", exp_dir)

    trainer = Trainer(
        cfg=cfg,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        device=device,
        logger=logger,
        use_wandb=cfg.use_wandb,
    )
    evaluator = Evaluator(
        cfg=cfg,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        device=device,
        logger=logger,
    )

    if job_type == "train":
        trainer.train()
    elif job_type == "eval":
        evaluator.evaluate()
    elif job_type == "train_eval":
        trainer.train()
        evaluator.evaluate()
    else:
        raise ValueError(f"Unsupported job_type: {job_type}. Expected one of: train, eval, train_eval")

    if distributed:
        torch.distributed.destroy_process_group()

    if cfg.use_wandb and rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
