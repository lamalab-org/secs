"""13C peak-set encoder with relative-shift attention.

Two things the plain `transformer` backbone cannot do, both of which matter for
carbon spectra:

1. **It cannot see peak spacing directly.** `FourierShiftEmbedding` is an
   *absolute* encoding. Sinusoidal features do carry relative offsets in their
   dot products -- that is the usual argument for absolute encodings -- but only
   as long as the dot product is taken on the sinusoids themselves. The plain
   backbone pushes them through a two-layer MLP first, which destroys that
   structure before attention ever sees it. Yet what identifies a substructure
   in a 13C spectrum is largely a *pattern of gaps*: a para-disubstituted ring
   is four lines in a characteristic arrangement, and that arrangement means the
   same thing wherever it sits.

   Here every attention logit gets an additive learned bias that depends only on
   the ppm difference between the two peaks, bucketed on a signed log scale so
   sub-ppm separations (a residual solvent multiplet, an unresolved pair) get
   their own resolution while 100 ppm apart is one coarse bucket.

2. **It throws away the peak count.** Masked-mean pooling divides by the number
   of peaks, so a 5-peak and a 40-peak spectrum come out at the same scale. The
   peak count is very close to the number of distinct carbons -- the single most
   informative scalar about the molecule. Here it is fed in explicitly.

Pooling is by a learned query (PMA, from the Set Transformer), concatenated with
the masked mean and a count embedding.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from secs.models.base import ModalityEncoder
from secs.models.encoders.c_nmr.transformer import FourierShiftEmbedding
from secs.models.registry import register_encoder

NEG_INF = float("-inf")


def relative_buckets(delta: torch.Tensor, n_half: int, lin_ppm: float, max_ppm: float) -> torch.Tensor:
    """Signed ppm difference -> bucket index in [0, 2 * n_half].

    Linear below `lin_ppm`, logarithmic above it. Two carbons 0.2 ppm apart and
    two 0.4 ppm apart are chemically very different situations; 140 ppm apart
    and 150 ppm apart are the same situation. A uniform bucketing would spend
    its resolution in the wrong place.
    """
    sign = torch.sign(delta)
    a = delta.abs()

    n_lin = n_half // 2
    n_log = n_half - n_lin

    lin_idx = (a / lin_ppm * n_lin).floor()
    # +1 so the first log bucket starts where the linear ones stop
    ratio = torch.log(a.clamp(min=lin_ppm) / lin_ppm) / math.log(max_ppm / lin_ppm)
    log_idx = n_lin + (ratio * n_log).floor()

    idx = torch.where(a < lin_ppm, lin_idx, log_idx).clamp(0, n_half - 1)
    return (sign * idx).long() + n_half


class ShiftRelativeBias(nn.Module):
    """Per-head additive attention bias, a lookup on the bucketed ppm gap.

    A table lookup rather than an MLP over difference features on purpose: the
    (B, P, P, n_features) tensor an MLP needs is an order of magnitude larger
    than the (B, heads, P, P) logits themselves, which at batch 768 is the
    difference between a few hundred MB and several GB.
    """

    def __init__(self, n_heads: int, n_half: int = 24, lin_ppm: float = 2.0, max_ppm: float = 218.0):
        super().__init__()
        self.n_half, self.lin_ppm, self.max_ppm = n_half, lin_ppm, max_ppm
        self.table = nn.Embedding(2 * n_half + 1, n_heads)
        nn.init.zeros_(self.table.weight)

    def forward(self, shifts: torch.Tensor) -> torch.Tensor:  # (B,P) -> (B,H,P,P)
        delta = shifts.unsqueeze(-1) - shifts.unsqueeze(-2)
        idx = relative_buckets(delta, self.n_half, self.lin_ppm, self.max_ppm)
        return self.table(idx).permute(0, 3, 1, 2)


class RelAttentionBlock(nn.Module):
    """Pre-norm transformer block whose attention takes an additive bias."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, p, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, p, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(b, p, d)
        x = x + self.drop(self.proj(out))
        return x + self.mlp(self.norm2(x))


class PooledByQuery(nn.Module):
    """Pooling by multihead attention from a single learned query (PMA)."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=pad, need_weights=False)
        return self.norm(out.squeeze(1))


class RelativePeakTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        n_freqs: int = 64,
        max_ppm: float = 218.0,
        n_rel_buckets: int = 24,
        rel_linear_ppm: float = 2.0,
        max_peaks: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_peaks = max_peaks

        self.pos = FourierShiftEmbedding(embed_dim, n_freqs=n_freqs, max_ppm=max_ppm)
        self.peak_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.rel_bias = ShiftRelativeBias(num_heads, n_half=n_rel_buckets, lin_ppm=rel_linear_ppm, max_ppm=max_ppm)

        self.blocks = nn.ModuleList([RelAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.pool = PooledByQuery(embed_dim, num_heads, dropout=dropout)

        # peak count, straight in: it is nearly the number of distinct carbons
        self.count_embed = nn.Embedding(max_peaks + 1, embed_dim)
        self.merge = nn.Sequential(nn.Linear(3 * embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
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

    def forward(self, shifts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pad = ~mask
        # An all-padding row would make every logit -inf and every output NaN.
        empty = pad.all(dim=1)
        if empty.any():
            pad = pad.clone()
            pad[empty, 0] = False
        keep = (~pad).unsqueeze(-1).to(shifts.dtype)

        x = (self.pos(shifts) + self.peak_token) * keep

        bias = self.rel_bias(shifts)  # (B,H,P,P)
        attn_mask = bias.masked_fill(pad[:, None, None, :], NEG_INF)

        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.encoder_norm(x)

        n = keep.sum(dim=1)  # (B,1)
        mean = (x * keep).sum(dim=1) / n.clamp(min=1.0)
        pooled = self.pool(x, pad)
        count = self.count_embed(n.squeeze(-1).long().clamp(0, self.max_peaks))
        return self.merge(torch.cat([pooled, mean, count], dim=-1))


@register_encoder("c_nmr", "relative")
class CNmrRelativeTransformerEncoder(ModalityEncoder):
    """Peak-set transformer with relative-shift attention bias and count-aware pooling."""

    def __init__(self, ckpt_path: str | None = None, freeze_encoder: bool = False, **backbone_kwargs) -> None:
        super().__init__(ckpt_path=ckpt_path, freeze_encoder=freeze_encoder)
        self.encoder = RelativePeakTransformer(**backbone_kwargs)
        self.output_dim = self.encoder.embed_dim
        self._finalize()

    def forward(self, inputs, mask=None):
        shifts, mask = inputs if mask is None else (inputs, mask)
        if self.frozen:
            with torch.no_grad():
                return self.encoder(shifts, mask)
        return self.encoder(shifts, mask)
