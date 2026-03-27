import torch.nn as nn

from damage_mapping.models.Encoder_Prithvi import PrithviEncoder
from damage_mapping.models.Encoder_TerraMind import TerraMindEncoder


def build_encoder(encoder_cfg) -> nn.Module:
    encoder_name = str(getattr(encoder_cfg, "name", "Terramind")).strip().lower()
    build_kwargs = dict(getattr(encoder_cfg, "build_kwargs", {}) or {})

    common_kwargs = {
        "version": encoder_cfg.version,
        "pretrained": bool(getattr(encoder_cfg, "pretrained", True)),
        "modalities": list(encoder_cfg.modalities),
        **build_kwargs,
    }

    if encoder_name in {"terramind", "terra_mind"}:
        return TerraMindEncoder(**common_kwargs)

    if encoder_name == "prithvi":
        return PrithviEncoder(**common_kwargs)

    raise ValueError(
        f"Unsupported encoder '{encoder_cfg.name}'. Expected one of: 'Terramind' or 'Prithvi'."
    )
