import math

import torch
from loguru import logger
import torch.nn as nn
import torch.nn.functional as F


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
        freqs = torch.logspace(0, math.log10(max_ppm / 2.0), n_freqs)
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
    """Turn (shifts, mask) into a token sequence with a learned [CLS]-style
    presence signal. Padding tokens are zeroed and masked in attention."""

    def __init__(self, dim: int, n_freqs: int = 64, max_ppm: float = 218.0):
        super().__init__()
        self.pos = FourierShiftEmbedding(dim, n_freqs=n_freqs, max_ppm=max_ppm)
        # a learned bias added to every real peak token (marks "a peak is here")
        self.peak_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, shifts, mask):  # (B,P),(B,P) -> tokens (B,P,dim)
        tok = self.pos(shifts) + self.peak_token
        tok = tok * mask.unsqueeze(-1)  # zero out padding
        return tok


class AttentionPool(nn.Module):
    """Learned query attends over peak tokens -> one vector. Respects padding."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):  # x:(B,P,D), mask:(B,P) True=pad
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))


class PeakSetBackbone(nn.Module):
    """Set-transformer style encoder over 13C peaks for contrastive alignment.

    peaks (shifts, mask)
        -> continuous ppm embedding (+ peak-presence token)
        -> transformer encoder (permutation-equivariant, padding-masked)
        -> attention pooling (padding-aware)
        -> projection head -> L2-normalized embedding

    No convs, no downsampling, no BatchNorm: none fit a sparse permutation-
    invariant peak set. LayerNorm throughout (stable on sparse input).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        proj_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        n_freqs: int = 64,
        max_ppm: float = 218.0,
    ):
        super().__init__()
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
        self.pool = AttentionPool(embed_dim, num_heads=num_heads, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
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

    def forward(self, shifts, mask, normalize: bool = True):
        # transformer/pool want True = PAD; our mask has True = real peak
        pad = ~mask
        x = self.tokenizer(shifts, mask)
        x = self.transformer(x, src_key_padding_mask=pad)
        x = self.encoder_norm(x)
        x = self.pool(x, key_padding_mask=pad)
        x = self.head(x)
        if normalize:
            x = F.normalize(x, dim=-1)
        return x


class cNmrEncoder(nn.Module):
    """API-compatible wrapper. forward now takes (shifts, mask) instead of a
    dense spectrum tensor.

    Args:
        ckpt_path:      optional checkpoint (raw state_dict or wrapped in
                        'state_dict'/'model'/'encoder'; leading 'encoder.'
                        prefix stripped).
        freeze_encoder: freeze backbone params and keep it in eval mode.
    """

    def __init__(self, ckpt_path: str | None = None, freeze_encoder: bool = False, **backbone_kwargs) -> None:
        super().__init__()
        self.encoder = PeakSetBackbone(**backbone_kwargs)
        self.frozen = freeze_encoder
        if ckpt_path is not None:
            self._load_ckpt(ckpt_path)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def _load_ckpt(self, ckpt_path: str) -> None:
        state = torch.load(ckpt_path, map_location="cpu")
        for key in ("state_dict", "model", "encoder"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
        cleaned = {k[len("encoder.") :] if k.startswith("encoder.") else k: v for k, v in state.items()}
        missing, unexpected = self.encoder.load_state_dict(cleaned, strict=False)
        if missing:
            logger.warning(f"cNmrEncoder: {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            logger.warning(f"cNmrEncoder: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, inputs, mask=None, normalize: bool = True):
        if mask is None:
            shifts, mask = inputs
        else:
            shifts = inputs
        if self.frozen:
            with torch.no_grad():
                return self.encoder(shifts, mask, normalize=normalize)
        return self.encoder(shifts, mask, normalize=normalize)
