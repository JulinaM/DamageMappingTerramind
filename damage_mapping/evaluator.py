import os
from collections import defaultdict

import hydra
import numpy as np
import rasterio as rio
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.Encoder_TerraMind import TerraMindEncoder
from models.Decoder_UNet2D import UNet2D
from models.utils import move_to_device, calc_test_metrics, tensor_to_color_image
from datasets.DataLoader import TestLoader
from logger import init_logger


CONFIG_DIR = "/users/PGS0218/julina/projects/geography/damage_mapping_terramind/V2/configs/old_configs/test"
EVAL_MODALITIES = ["S2L2A"]
COLOR_TABLE = {
    0: (0, 0, 0, 255),      # background
    1: (0, 255, 0, 255),    # healthy
    2: (255, 0, 0, 255),    # damaged
}


def _to_int(value):
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _get_output_dir() -> str:
    hc = HydraConfig.get()
    if "sweep" in hc:
        return os.path.join(hc["sweep"]["dir"], hc["sweep"]["subdir"])
    return hc["runtime"]["output_dir"]


def _build_test_loader(cfg: DictConfig) -> DataLoader:
    missing = [name for name in EVAL_MODALITIES if name not in cfg.paths.modalities]
    if missing:
        raise ValueError(f"Missing configured modality paths for: {missing}")

    test_modalities = {
        name: (cfg.paths.modalities[name].before, cfg.paths.modalities[name].after)
        for name in EVAL_MODALITIES
    }
    test_data = TestLoader(
        modalities=test_modalities,
        label_dir=cfg.paths.label_dir,
        patch_size=cfg.model.patch_size,
        stride=cfg.model.stride,
    )
    return DataLoader(test_data, batch_size=None, num_workers=cfg.model.num_workers)


def _load_models(cfg: DictConfig, device: str):
    checkpoint = torch.load(cfg.paths.trained_model, map_location=device)

    encoder = TerraMindEncoder(
        version="terramind_v1_base",
        pretrained=True,
        modalities=EVAL_MODALITIES,
    )
    decoder = UNet2D(num_classes=4)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder.to(device).eval()
    decoder.to(device).eval()
    return encoder, decoder


def _collect_patch_outputs(test_dataloader, encoder, decoder, device):
    padding = {}
    metas = {}
    tile_reconstruction = defaultdict(list)

    with torch.no_grad():
        for x, y, (i, coord_y, coord_x), (pad_left, pad_right, pad_top, pad_bottom), meta in test_dataloader:
            x = move_to_device(x, device)
            z_before, z_after = encoder(x["before"]), encoder(x["after"])
            z_differenced = [after - before for before, after in zip(z_before, z_after)]
            logits = decoder(z_differenced)
            y_hat = torch.argmax(logits, dim=1).cpu()
            y_true = y.cpu()

            idx = _to_int(i)
            tile_reconstruction[idx].append((y_hat, y_true, _to_int(coord_y), _to_int(coord_x)))

            if idx not in padding:
                padding[idx] = tuple(_to_int(p) for p in (pad_left, pad_right, pad_top, pad_bottom))
            if idx not in metas:
                metas[idx] = meta

    return tile_reconstruction, padding, metas


def _reconstruct_tiles(tile_reconstruction, patch_size):
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
        for y_hat, y_true, coord_y, coord_x in patches:
            y_slice = slice(coord_y, coord_y + patch_size)
            x_slice = slice(coord_x, coord_x + patch_size)
            image_tiles_true[idx][y_slice, x_slice] = y_true.squeeze()
            image_tiles_pred[idx][y_slice, x_slice] = y_hat.squeeze()

    return image_tiles_true, image_tiles_pred


def _remove_padding(image_tiles, padding):
    for idx, img in image_tiles.items():
        height, width = img.shape[-2], img.shape[-1]
        pad_left, pad_right, pad_top, pad_bottom = padding[idx]
        image_tiles[idx] = img[pad_top:height - pad_bottom, pad_left:width - pad_right]


def _mask_background_predictions(image_tiles_pred, image_tiles_true):
    for idx, pred in image_tiles_pred.items():
        real = image_tiles_true[idx]
        image_tiles_pred[idx] = torch.where(real == 0, torch.zeros_like(pred), pred)


def _save_geotiffs(output_dir, image_tiles_pred, metas):
    tiff_dir = os.path.join(output_dir, "geotiffs")
    os.makedirs(tiff_dir, exist_ok=True)

    for idx, pred in image_tiles_pred.items():
        meta_out = metas[idx].copy()
        # Ensure rasterio can apply a palette colormap without conflicting source tags.
        meta_out.pop("photometric", None)
        meta_out.update(
            {
                "driver": "GTiff",
                "height": pred.shape[0],
                "width": pred.shape[1],
                "count": 1,
                "dtype": "uint8",
                "nodata": 0,
                "photometric": "palette",
            }
        )
        pred_np = pred.numpy().astype(np.uint8)
        tiff_path = os.path.join(tiff_dir, f"predicted_map_{idx}_colored.tif")

        with rio.open(tiff_path, "w", **meta_out) as dst:
            dst.write(pred_np, 1)
            dst.write_colormap(1, COLOR_TABLE)

    return tiff_dir

def _save_metrics_and_visualizations(output_dir, image_tiles_pred, image_tiles_true, writer):
    test_metrics = calc_test_metrics(
        image_tiles_pred,
        image_tiles_true,
        ignore_index=0,
        positive_class=2,
        negative_class=1,
    )

    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        for idx, metrics in test_metrics.items():
            f.write(f"\nImage {idx} metrics:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
                writer.add_scalar(f"metrics/{key}", value, global_step=idx)

    for idx in list(image_tiles_pred.keys())[:3]:
        pred_rgb = tensor_to_color_image(image_tiles_pred[idx])
        true_rgb = tensor_to_color_image(image_tiles_true[idx])
        combined = torch.cat((true_rgb, pred_rgb), dim=2)
        writer.add_image(f"comparison/{idx}", combined)

    return test_metrics, metrics_path


@hydra.main(version_base="1.2", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = _get_output_dir()
    logger = init_logger(
        filepath=os.path.join(output_dir, "evaluator.log"),
        rank=0,
        add_rank_suffix=False,
        use_console=False,
    )
    logger.info("Evaluator started")
    logger.info("Output directory: %s", output_dir)
    logger.info("Device: %s", device)
    logger.info("Model checkpoint: %s", cfg.paths.trained_model)
    logger.info("Eval modalities: %s", EVAL_MODALITIES)
    logger.info("Patch size=%s, stride=%s", cfg.model.patch_size, cfg.model.stride)

    writer = SummaryWriter(log_dir=output_dir)
    try:
        logger.info("Building test dataloader")
        test_dataloader = _build_test_loader(cfg)
        logger.info("Total test patches: %d", len(test_dataloader.dataset))

        logger.info("Loading encoder/decoder")
        encoder, decoder = _load_models(cfg, device)

        logger.info("Running inference and collecting patches")
        tile_reconstruction, padding, metas = _collect_patch_outputs(test_dataloader, encoder, decoder, device)
        if not tile_reconstruction:
            raise RuntimeError(
                "No test patches were generated. Check paths.modalities/label_dir and input data availability."
            )
        logger.info("Collected patches for %d image tile(s)", len(tile_reconstruction))

        image_tiles_true, image_tiles_pred = _reconstruct_tiles(tile_reconstruction, cfg.model.patch_size)
        _remove_padding(image_tiles_true, padding)
        _remove_padding(image_tiles_pred, padding)
        _mask_background_predictions(image_tiles_pred, image_tiles_true)
        logger.info("Reconstructed and postprocessed %d image tile(s)", len(image_tiles_pred))

        tiff_dir = _save_geotiffs(output_dir, image_tiles_pred, metas)
        logger.info("Saved %d GeoTIFF(s) to %s", len(image_tiles_pred), tiff_dir)

        test_metrics, metrics_path = _save_metrics_and_visualizations(output_dir, image_tiles_pred, image_tiles_true, writer)
        logger.info("Saved metrics to %s", metrics_path)
        logger.info("Per-image metrics: %s", test_metrics)
        logger.info("TensorBoard event file written under %s", output_dir)
        logger.info("Evaluator completed successfully")
    except Exception:
        logger.exception("Evaluator failed")
        raise
    finally:
        writer.close()
        logger.info("Closed TensorBoard writer")

if __name__ == "__main__":
    main()
