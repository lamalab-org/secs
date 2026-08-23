"""1H encoder: dilated-CNN stem, attention neck with relative ppm bias.

`dilated_cnn` reads the two physical scales of a 1H spectrum locally; what it
cannot do cheaply is relate features far apart -- the triplet at 0.98 and the
quartet at 0.81 are one ethyl group, a doublet at 6.4 and one at 7.1 with the
same spacing are one coupled pair. A pure CNN only sees such pairs in its deepest,
coarsest layers. Here the conv stem downsamples the raw spectrum to a few hundred
positions and a short transformer relates them, with two kinds of position
information the 13C encoders showed to matter:

- **absolute**: each position gets a Fourier embedding of its ppm value, since
  chemical shift is chemistry (4.1 ppm means something), and convolutions are
  translation-equivariant and would otherwise not know where they are;
- **relative**: every attention logit carries a learned bias over the ppm gap
  between two positions, bucketed linearly at small gaps and logarithmically at
  large ones, so "0.17 ppm apart" is the same feature at 1.2 and at 4.3 ppm.

The input is still only the raw intensity vector (plus the cumulative-integral
channel computed from it); nothing is pre-extracted.
"""

import torch
import torch.nn as nn

from secs.models.base import ModalityEncoder
from secs.models.encoders.c_nmr.raster_cnn import ResBlock1D
from secs.models.encoders.c_nmr.relative import PooledByQuery, RelAttentionBlock, ShiftRelativeBias
from secs.models.encoders.c_nmr.transformer import FourierShiftEmbedding
from secs.models.encoders.h_nmr.dilated_cnn import CumulativeIntegral
from secs.models.registry import register_encoder


class ConvAttention1D(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        ppm_min: float = -0.5,
        ppm_max: float = 12.0,
        channels: tuple[int, ...] = (32, 64, 128, 192, 256, 256),
        blocks_per_stage: int = 2,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4),
        dim: int = 256,
        depth: int = 3,
        n_heads: int = 8,
        n_freqs: int = 32,
        rel_buckets: int = 24,
        rel_lin_ppm: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ppm_min, self.ppm_max = ppm_min, ppm_max
        self.integral = CumulativeIntegral()

        # --- conv stem: len(channels) - 1 stride-2 stages, 5000 points -> ~157 positions.
        # Kept slim at full resolution and downsampled 5x: attention memory goes with
        # positions^2 and the stem's activations with channels x length, and both are
        # what decides whether batch 768 fits.
        self.stem = nn.Sequential(
            nn.Conv1d(2, channels[0], 7, padding=3, bias=False), nn.BatchNorm1d(channels[0]), nn.SiLU(inplace=True)
        )
        stages, in_ch = [], channels[0]
        for out_ch in channels[1:]:
            stage = [nn.Conv1d(in_ch, out_ch, kernel_size, stride=2, padding=kernel_size // 2, bias=False)]
            for i in range(blocks_per_stage):
                stage.append(ResBlock1D(out_ch, kernel_size, dilation=dilations[i % len(dilations)], dropout=dropout))
            stages.append(nn.Sequential(*stage))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.to_tokens = nn.Sequential(nn.BatchNorm1d(in_ch), nn.SiLU(inplace=True), nn.Conv1d(in_ch, dim, 1))

        # --- attention neck over the downsampled positions
        self.pos = FourierShiftEmbedding(dim, n_freqs=n_freqs, max_ppm=ppm_max - ppm_min)
        self.rel_bias = ShiftRelativeBias(n_heads, n_half=rel_buckets, lin_ppm=rel_lin_ppm, max_ppm=ppm_max - ppm_min)
        self.blocks = nn.ModuleList([RelAttentionBlock(dim, n_heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        self.pool = PooledByQuery(dim, n_heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def positions_ppm(self, n_positions: int, device) -> torch.Tensor:
        """ppm of each downsampled position: the bin centres of an even split of the axis."""
        edges = torch.linspace(self.ppm_min, self.ppm_max, n_positions + 1, device=device)
        return (edges[:-1] + edges[1:]) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.integral(x)
        x = self.stem(x)
        x = self.stages(x)
        tokens = self.to_tokens(x).transpose(1, 2)  # (B, P, dim)

        ppm = self.positions_ppm(tokens.shape[1], tokens.device)
        # the Fourier embedding is relative to the axis start so its frequencies span the window
        tokens = tokens + self.pos((ppm - self.ppm_min).unsqueeze(0))
        bias = self.rel_bias(ppm.unsqueeze(0))  # (1, H, P, P), broadcast over the batch
        for block in self.blocks:
            tokens = block(tokens, bias)
        tokens = self.norm(tokens)

        no_pad = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        pooled = torch.cat([self.pool(tokens, no_pad), tokens.mean(dim=1)], dim=-1)
        return self.head(pooled)


@register_encoder("h_nmr", "conv_attention")
class HNmrConvAttentionEncoder(ModalityEncoder):
    """Dilated-CNN stem plus relative-position attention neck over the raw 1H spectrum."""

    def __init__(self, ckpt_path: str | None = None, freeze_encoder: bool = False, **backbone_kwargs) -> None:
        super().__init__(ckpt_path=ckpt_path, freeze_encoder=freeze_encoder)
        self.encoder = ConvAttention1D(**backbone_kwargs)
        self.output_dim = self.encoder.embed_dim
        self._finalize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
