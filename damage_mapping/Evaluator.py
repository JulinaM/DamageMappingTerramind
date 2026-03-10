import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio as rio
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from damage_mapping.datasets.DataLoader import TestLoader
from damage_mapping.models.Decoder_UNet2D import UNet2D
from damage_mapping.models.Encoder_TerraMind import TerraMindEncoder
from damage_mapping.models.utils import calc_test_metrics, move_to_device, tensor_to_color_image


COLOR_TABLE = {
    0: (0, 0, 0, 255),
    1: (0, 255, 0, 255),
    2: (255, 0, 0, 255),
    3: (255, 255, 0, 255),
}


class Evaluator:
    def __init__(
        self,
        cfg: DictConfig,
        exp_dir: str | Path,
        ckpt_dir: str | Path,
        device: str | torch.device,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = cfg
        self.exp_dir = Path(exp_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.device = torch.device(device)
        self.logger = logger or logging.getLogger(__name__)
        self.writer = SummaryWriter(log_dir=str(self.exp_dir))
        self.holdout_cfg = cfg.get("holdout_loader")

    def is_configured(self) -> bool:
        return self.holdout_cfg is not None

    def evaluate(self, checkpoint_path: str | Path | None = None) -> dict[int, dict[str, float]] | None:
        if not self.is_configured():
            self.logger.info("Holdout evaluator skipped: no holdout_loader section found in config.")
            self.writer.close()
            return None

        dataloader = self._build_dataloader()
        checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else self._find_best_checkpoint()
        encoder, decoder = self._load_models(checkpoint_path)

        self.logger.info("Evaluator started")
        self.logger.info("Holdout patches: %d", len(dataloader.dataset))
        self.logger.info("Using checkpoint: %s", checkpoint_path)

        tile_reconstruction, padding, metas = self._collect_patch_outputs(dataloader, encoder, decoder)
        image_tiles_true, image_tiles_pred = self._reconstruct_tiles(
            tile_reconstruction,
            patch_size=self.holdout_cfg.patch_size,
        )
        self._remove_padding(image_tiles_true, padding)
        self._remove_padding(image_tiles_pred, padding)
        self._mask_background_predictions(image_tiles_pred, image_tiles_true)

        geotiff_dir = self._save_geotiffs(image_tiles_pred, metas)
        metrics = self._save_metrics_and_visualizations(image_tiles_pred, image_tiles_true)
        self.writer.close()

        self.logger.info("Saved holdout GeoTIFFs to %s", geotiff_dir)
        self.logger.info("Holdout evaluation completed")
        return metrics

    def _build_dataloader(self) -> DataLoader:
        holdout_modalities = {
            name: (paths.before, paths.after) for name, paths in self.holdout_cfg.modalities.items()
        }
        dataset = TestLoader(
            modalities=holdout_modalities,
            label_dir=self.holdout_cfg.label_dir,
            patch_size=self.holdout_cfg.patch_size,
            stride=self.holdout_cfg.stride,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=getattr(self.holdout_cfg, "num_workers", 0),
        )

    def _find_best_checkpoint(self) -> Path:
        matches = sorted(self.ckpt_dir.glob("best_model_*.pt"))
        if not matches:
            raise FileNotFoundError(f"No best checkpoint found in {self.ckpt_dir}")
        return matches[-1]

    def _load_models(self, checkpoint_path: Path) -> tuple[TerraMindEncoder, UNet2D]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        encoder = TerraMindEncoder(
            version=self.cfg.model.TM_version,
            pretrained=self.cfg.model.pretrained,
            modalities=list(self.cfg.model.modalities),
        )
        decoder = UNet2D(num_classes=self.cfg.model.num_classes)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        encoder.to(self.device).eval()
        decoder.to(self.device).eval()
        return encoder, decoder

    def _collect_patch_outputs(
        self,
        dataloader: DataLoader,
        encoder: TerraMindEncoder,
        decoder: UNet2D,
    ) -> tuple[dict[int, list], dict[int, tuple[int, int, int, int]], dict[int, dict]]:
        padding = {}
        metas = {}
        tile_reconstruction = defaultdict(list)

        with torch.no_grad():
            for inputs, target, (idx, coord_y, coord_x), pad, meta in dataloader:
                inputs = move_to_device(inputs, self.device)
                z_before = encoder(inputs["before"])
                z_after = encoder(inputs["after"])
                z_differenced = [after - before for before, after in zip(z_before, z_after)]
                logits = decoder(z_differenced)
                prediction = torch.argmax(logits, dim=1).cpu()

                idx = self._to_int(idx)
                tile_reconstruction[idx].append(
                    (prediction, target.cpu(), self._to_int(coord_y), self._to_int(coord_x))
                )
                if idx not in padding:
                    padding[idx] = tuple(self._to_int(value) for value in pad)
                if idx not in metas:
                    metas[idx] = meta

        if not tile_reconstruction:
            raise RuntimeError("No holdout patches were generated. Check holdout_loader paths.")
        return tile_reconstruction, padding, metas

    def _reconstruct_tiles(
        self,
        tile_reconstruction: dict[int, list],
        patch_size: int,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        image_tiles_true = {}
        image_tiles_pred = {}

        for idx, patches in tile_reconstruction.items():
            max_coord_x = max(coord_x for _, _, _, coord_x in patches)
            max_coord_y = max(coord_y for _, _, coord_y, _ in patches)
            height = max_coord_y + patch_size
            width = max_coord_x + patch_size
            image_tiles_true[idx] = torch.zeros((height, width), dtype=torch.float32)
            image_tiles_pred[idx] = torch.zeros((height, width), dtype=torch.float32)

        for idx, patches in tile_reconstruction.items():
            for prediction, target, coord_y, coord_x in patches:
                y_slice = slice(coord_y, coord_y + patch_size)
                x_slice = slice(coord_x, coord_x + patch_size)
                image_tiles_true[idx][y_slice, x_slice] = target.squeeze()
                image_tiles_pred[idx][y_slice, x_slice] = prediction.squeeze()

        return image_tiles_true, image_tiles_pred

    def _remove_padding(self, image_tiles: dict[int, torch.Tensor], padding: dict[int, tuple[int, int, int, int]]) -> None:
        for idx, image in image_tiles.items():
            height, width = image.shape[-2], image.shape[-1]
            pad_left, pad_right, pad_top, pad_bottom = padding[idx]
            image_tiles[idx] = image[pad_top:height - pad_bottom, pad_left:width - pad_right]

    def _mask_background_predictions(
        self,
        image_tiles_pred: dict[int, torch.Tensor],
        image_tiles_true: dict[int, torch.Tensor],
    ) -> None:
        for idx, prediction in image_tiles_pred.items():
            truth = image_tiles_true[idx]
            image_tiles_pred[idx] = torch.where(truth == 0, torch.zeros_like(prediction), prediction)

    def _save_geotiffs(self, image_tiles_pred: dict[int, torch.Tensor], metas: dict[int, dict]) -> Path:
        geotiff_dir = self.exp_dir / "geotiffs"
        geotiff_dir.mkdir(parents=True, exist_ok=True)

        for idx, prediction in image_tiles_pred.items():
            meta_out = metas[idx].copy()
            meta_out.pop("photometric", None)
            meta_out.update(
                {
                    "driver": "GTiff",
                    "height": prediction.shape[0],
                    "width": prediction.shape[1],
                    "count": 1,
                    "dtype": "uint8",
                    "nodata": 0,
                    "photometric": "palette",
                }
            )
            output_path = geotiff_dir / f"predicted_map_{idx}_colored.tif"
            with rio.open(output_path, "w", **meta_out) as dst:
                dst.write(prediction.numpy().astype(np.uint8), 1)
                dst.write_colormap(1, COLOR_TABLE)

        return geotiff_dir

    def _save_metrics_and_visualizations(
        self,
        image_tiles_pred: dict[int, torch.Tensor],
        image_tiles_true: dict[int, torch.Tensor],
    ) -> dict[int, dict[str, float]]:
        metrics = calc_test_metrics(
            image_tiles_pred,
            image_tiles_true,
            ignore_index=self.cfg.model.ignore_index,
            positive_class=self.cfg.model.positive_class,
            negative_class=self.cfg.model.negative_class,
        )
        metrics_path = self.exp_dir / "metrics.txt"
        with metrics_path.open("w") as handle:
            for idx, image_metrics in metrics.items():
                handle.write(f"\nImage {idx} metrics:\n")
                for key, value in image_metrics.items():
                    handle.write(f"  {key}: {value:.4f}\n")
                    self.writer.add_scalar(f"holdout/{key}", value, global_step=idx)

        for idx in list(image_tiles_pred.keys())[:3]:
            pred_rgb = tensor_to_color_image(image_tiles_pred[idx], num_classes=self.cfg.model.num_classes)
            true_rgb = tensor_to_color_image(image_tiles_true[idx], num_classes=self.cfg.model.num_classes)
            self.writer.add_image(f"holdout/comparison_{idx}", torch.cat((true_rgb, pred_rgb), dim=2))

        self.logger.info("Saved holdout metrics to %s", metrics_path)
        return metrics

    @staticmethod
    def _to_int(value) -> int:
        if torch.is_tensor(value):
            return int(value.item())
        return int(value)
