"""1H encoder: dilated residual CNN over the raw spectrum, integral-aware.

A right-sized replacement for the generic `cnn` backbone, borrowing what the
13C raster encoder learned:

- The two physical scales of a 1H spectrum are multiplet fine structure
  (~0.02 ppm, a few bins) and envelopes (~1 ppm, hundreds of bins); dilated
  residual blocks cover both without a 50M-parameter stack.
- The embedding is unbounded. The `cnn` backbone ends in a Sigmoid, boxing the
  latent into [0, 1] before the projection head ever sees it.
- Integrals must survive normalisation. Spectra arrive max-normalised, so one
  tall singlet rescales everything and peak *areas* -- the proton counts -- are
  no longer comparable across spectra. A second input channel carries the
  cumulative integral, cumsum(y)/sum(y): a monotone 0..1 curve whose step at
  each signal is that signal's fraction of all protons, invariant to the
  max-scaling of the intensity channel.
"""

import torch
import torch.nn as nn

from secs.models.base import ModalityEncoder
from secs.models.encoders.c_nmr.raster_cnn import ResBlock1D
from secs.models.registry import register_encoder


class CumulativeIntegral(nn.Module):
    """(B, 1, N) intensities -> (B, 2, N): the spectrum and its running integral."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = x.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.cat([x, x.cumsum(dim=-1) / total], dim=1)


class DilatedCNN1D(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        channels: tuple[int, ...] = (64, 128, 192, 256, 384, 512),
        blocks_per_stage: int = 2,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.integral = CumulativeIntegral()

        self.stem = nn.Sequential(
            nn.Conv1d(2, channels[0], 7, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.SiLU(inplace=True),
        )

        stages = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            stage = [nn.Conv1d(in_ch, out_ch, kernel_size, stride=2, padding=kernel_size // 2, bias=False)]
            for i in range(blocks_per_stage):
                stage.append(ResBlock1D(out_ch, kernel_size, dilation=dilations[i % len(dilations)], dropout=dropout))
            stages.append(nn.Sequential(*stage))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.final_norm = nn.Sequential(nn.BatchNorm1d(in_ch), nn.SiLU(inplace=True))

        # Average and max pooling answer different questions -- "how much signal
        # is in this band" and "is this motif present anywhere" -- and a
        # retrieval embedding wants both.
        self.head = nn.Sequential(
            nn.Linear(2 * in_ch, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.integral(x)
        x = self.stem(x)
        x = self.stages(x)
        x = self.final_norm(x)
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)
        return self.head(pooled)


@register_encoder("h_nmr", "dilated_cnn")
class HNmrDilatedCNNEncoder(ModalityEncoder):
    """Dilated residual 1D CNN over the raw 1H spectrum with a cumulative-integral channel."""

    def __init__(self, ckpt_path: str | None = None, freeze_encoder: bool = False, **backbone_kwargs) -> None:
        super().__init__(ckpt_path=ckpt_path, freeze_encoder=freeze_encoder)
        self.encoder = DilatedCNN1D(**backbone_kwargs)
        self.output_dim = self.encoder.embed_dim
        self._finalize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
