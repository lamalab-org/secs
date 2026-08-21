import math

import torch
import torch.nn as nn

from secs.models.base import ModalityEncoder
from secs.models.registry import register_encoder


class FourierShiftEmbedding(nn.Module):
    """Embed a continuous ppm shift with fixed Fourier features + MLP.

    Sinusoidal features over a *continuous* scalar (not an integer index), so
    nearby ppm values get nearby embeddings and the model sees true peak
    spacing. Frequencies are deterministic (geometric/logspace, a fixed buffer,
    not random); only the MLP is learned.
    """

    def __init__(self, dim: int, n_freqs: int = 64, max_ppm: float = 218.0):
        super().__init__()
        assert dim % 2 == 0
        self.max_ppm = max_ppm
        # geometric spread of frequencies over the ppm range
        freqs = torch.logspace(0, math.log10(max_ppm / 0.05), n_freqs)
        self.register_buffer("freqs", freqs * (2 * math.pi / max_ppm))  # (n_freqs,)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freqs, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, shifts):  # (B, P) -> (B, P, dim)
        proj = shifts.unsqueeze(-1) * self.freqs  # (B, P, n_freqs)
        feats = torch.cat([proj.sin(), proj.cos()], dim=-1)  # (B, P, 2*n_freqs)
        return self.mlp(feats)


class PeakTokenizer(nn.Module):
    """Turn (shifts, mask) into a token sequence. Padding tokens are zeroed
    and masked in attention."""

    def __init__(self, dim: int, n_freqs: int = 64, max_ppm: float = 218.0):
        super().__init__()
        self.pos = FourierShiftEmbedding(dim, n_freqs=n_freqs, max_ppm=max_ppm)
        # a learned bias added to every real peak token (marks "a peak is here")
        self.peak_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, shifts, mask):  # (B,P),(B,P) -> tokens (B,P,dim)
        tok = self.pos(shifts) + self.peak_token
        return tok * mask.unsqueeze(-1)  # zero out padding


class PeakSetTransformer(nn.Module):
    """Set-transformer style encoder over 13C peaks for contrastive alignment.

    peaks (shifts, mask)
        -> continuous ppm embedding (+ peak-presence token)
        -> transformer encoder (permutation-equivariant, padding-masked)
        -> masked mean over the real peak tokens  ->  (B, embed_dim)

    The backbone stops at the pooled representation. Projection to the shared
    contrastive space is `ProjectionHead`, configured per modality under
    `model.projection_heads` -- keeping a second projection stack in here as
    well meant four linear layers after pooling and two places to configure
    the same thing. Normalisation belongs to the consumer too: InfoNCE and
    `cosine_similarity` both normalise their inputs.

    Readout is a masked mean rather than a [CLS] token or a learned query.
    Both of those are a softmax over the set, which can saturate onto one or
    two peaks and forces the whole spectrum through a single learned routing
    step; for an unordered set of ~10-40 peaks where the entire distribution
    carries the signal, the mean is the better inductive bias and costs no
    parameters. A content-free [CLS] also looks identical in every sample, so
    peaks attend to it as a cheap no-op and it drains attention mass from the
    peak-peak comparisons that matter (the ViT/LLM attention-sink effect).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        n_freqs: int = 64,
        max_ppm: float = 218.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.tokenizer = PeakTokenizer(embed_dim, n_freqs=n_freqs, max_ppm=max_ppm)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, shifts, mask):
        x = self.tokenizer(shifts, mask)  # (B, P, D)

        pad = ~mask  # True = pad
        # A fully padded row would mask every position, and softmax over an
        # all -inf row is NaN. The [CLS] token used to guarantee one live
        # position; keep the first one live instead. Its value is then
        # excluded from the mean below, so it cannot contribute.
        empty = pad.all(dim=1)
        if empty.any():
            pad = pad.clone()
            pad[empty, 0] = False

        x = self.transformer(x, src_key_padding_mask=pad)
        x = self.encoder_norm(x)

        # Masked mean over real peaks only; clamp so an empty row gives zeros
        # rather than a division by zero.
        keep = mask.unsqueeze(-1).to(x.dtype)  # (B, P, 1)
        return (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)


@register_encoder("c_nmr", "transformer", default=True)
class CNmrTransformerEncoder(ModalityEncoder):
    """Set-transformer encoder over a 13C peak list. forward takes (shifts, mask)."""

    def __init__(self, ckpt_path: str | None = None, freeze_encoder: bool = False, **backbone_kwargs) -> None:
        super().__init__(ckpt_path=ckpt_path, freeze_encoder=freeze_encoder)
        self.encoder = PeakSetTransformer(**backbone_kwargs)
        self.output_dim = self.encoder.embed_dim
        self._finalize()

    def forward(self, inputs, mask=None):
        if mask is None:
            shifts, mask = inputs
        else:
            shifts = inputs
        if self.frozen:
            with torch.no_grad():
                return self.encoder(shifts, mask)
        return self.encoder(shifts, mask)
