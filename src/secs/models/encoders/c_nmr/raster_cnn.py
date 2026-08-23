"""13C encoder that rasterises the peak list back into a spectrum and convolves.

The set transformer and this one fail in different directions, which is the
reason to have both.

A set transformer reasons about peaks as discrete objects: drop one and the
representation changes discontinuously. That is exactly the regime the data is
in -- the 8-sigma peak picker deletes weak carbons, so the same molecule yields
different-length peak lists run to run, and quaternaries go missing first.

A convolution over a ppm axis degrades gracefully instead. A missing peak
removes a bump from a channel that is still carrying its neighbours, and what
the filters see -- "two lines about 1 ppm apart near 128 ppm", "a crowded
aromatic envelope", "nothing at all above 160" -- survives losing any one of
them. Regional density is also a real signal that a set encoder has to work to
recover and a convolution gets for free.

The raster is deliberately soft: each peak is a small Gaussian, not a spike, so
that a peak moving by less than a bin changes the input smoothly, and so that
the picker's centroid bias on merged lines does not read as a different feature.
"""

import torch
import torch.nn as nn

from secs.models.base import ModalityEncoder
from secs.models.registry import register_encoder


class PeakRasteriser(nn.Module):
    """(shifts, mask) -> (B, 1, n_bins) soft-binned spectrum.

    Two channels of information survive: where the peaks are, and how many there
    are (the raster integrates to the peak count, so an average-pooling CNN can
    read the count off the DC component instead of having it divided out).
    """

    def __init__(self, n_bins: int = 4096, min_ppm: float = -5.0, max_ppm: float = 230.0, sigma_bins: float = 2.0):
        super().__init__()
        self.n_bins, self.min_ppm, self.max_ppm = n_bins, min_ppm, max_ppm
        width = max(3, round(6 * sigma_bins) | 1)  # odd, +-3 sigma
        pos = torch.arange(width, dtype=torch.float32) - width // 2
        kernel = torch.exp(-0.5 * (pos / sigma_bins) ** 2)
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.view(1, 1, -1))
        self.pad = width // 2

    def forward(self, shifts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = shifts.shape[0]
        scale = self.n_bins / (self.max_ppm - self.min_ppm)
        idx = ((shifts - self.min_ppm) * scale).long().clamp(0, self.n_bins - 1)

        weight = mask.to(shifts.dtype)
        grid = shifts.new_zeros(b, self.n_bins)
        grid.scatter_add_(1, idx, weight)

        grid = grid.unsqueeze(1)
        return nn.functional.conv1d(grid, self.kernel.to(grid.dtype), padding=self.pad)


class ResBlock1D(nn.Module):
    """Pre-activation residual block; dilation widens the ppm window cheaply."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.body = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False),
        )

    def forward(self, x):
        return x + self.body(x)


class RasterCNN(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        n_bins: int = 4096,
        min_ppm: float = -5.0,
        max_ppm: float = 230.0,
        sigma_bins: float = 2.0,
        channels: tuple[int, ...] = (64, 128, 192, 256, 384, 512),
        blocks_per_stage: int = 2,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.raster = PeakRasteriser(n_bins=n_bins, min_ppm=min_ppm, max_ppm=max_ppm, sigma_bins=sigma_bins)

        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], 7, padding=3, bias=False),
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

    def forward(self, shifts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.raster(shifts, mask)
        x = self.stem(x)
        x = self.stages(x)
        x = self.final_norm(x)
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)
        return self.head(pooled)


@register_encoder("c_nmr", "raster_cnn")
class CNmrRasterCNNEncoder(ModalityEncoder):
    """Dilated residual 1D CNN over a soft-rasterised 13C peak list."""

    def __init__(self, ckpt_path: str | None = None, freeze_encoder: bool = False, **backbone_kwargs) -> None:
        super().__init__(ckpt_path=ckpt_path, freeze_encoder=freeze_encoder)
        self.encoder = RasterCNN(**backbone_kwargs)
        self.output_dim = self.encoder.embed_dim
        self._finalize()

    def forward(self, inputs, mask=None):
        shifts, mask = inputs if mask is None else (inputs, mask)
        if self.frozen:
            with torch.no_grad():
                return self.encoder(shifts, mask)
        return self.encoder(shifts, mask)
