from typing import Sequence

import torch
import torch.nn as nn
from terratorch.models import necks


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """
    U-Net style segmentation decoder for TerraMind token features.

    Expects a list of token tensors from the encoder, selects multi-scale indices,
    reshapes tokens to images, then decodes to pixel logits.
    """

    def __init__(
        self,
        num_classes: int = 3,
        token_dim: int = 768,
        indices: Sequence[int] = (3, 5, 7, 9, 11),
        decoder_channels: Sequence[int] = (64, 128, 256, 512, 1024),
        remove_cls_token: bool = False,
    ) -> None:
        super().__init__()

        if len(indices) != len(decoder_channels):
            raise ValueError("`indices` and `decoder_channels` must have the same length.")
        if len(indices) < 2:
            raise ValueError("Decoder requires at least two feature scales.")

        self.indices = list(indices)
        self.decoder_channels = list(decoder_channels)
        n_levels = len(self.indices)
        channel_list = [token_dim] * n_levels

        self.select_indices = necks.SelectIndices(channel_list=channel_list, indices=self.indices)
        self.reshape_tokens = necks.ReshapeTokensToImage(
            channel_list=channel_list,
            remove_cls_token=remove_cls_token,
        )

        self.projections = nn.ModuleList(
            [nn.Conv2d(token_dim, out_ch, kernel_size=1) for out_ch in self.decoder_channels]
        )

        self.upsamplers = nn.ModuleList(
            [
                nn.Identity()
                if i == n_levels - 1
                else nn.Upsample(scale_factor=2 ** (n_levels - 1 - i), mode="bilinear", align_corners=False)
                for i in range(n_levels)
            ]
        )

        self.bottom = ConvBlock(self.decoder_channels[-1], self.decoder_channels[-1])
        self.up_blocks = nn.ModuleList()
        curr_channels = self.decoder_channels[-1]
        for skip_channels in reversed(self.decoder_channels[:-1]):
            self.up_blocks.append(
                UpBlock(
                    in_channels=curr_channels,
                    skip_channels=skip_channels,
                    out_channels=skip_channels,
                )
            )
            curr_channels = skip_channels

        self.head = nn.Conv2d(curr_channels, num_classes, kernel_size=1)

    def forward(self, embeddings):
        features = self.select_indices(embeddings)
        features = self.reshape_tokens(features)

        projected = []
        for feature, projection, upsample in zip(features, self.projections, self.upsamplers):
            projected.append(upsample(projection(feature)))

        x = self.bottom(projected[-1])
        skips = projected[:-1]
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip)

        return self.head(x)
