import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem(nn.Module):
    """
    Light convolutional stem for sparse 13C spectra.

    Design choices (vs. the original):
      - NO early MaxPool and NO stride-2 stem conv. Isolated 13C peaks must
        survive to the transformer; the original stem downsampled 4x before
        any block ran, smearing/dropping single-bin peaks.
      - Gentle, controlled downsampling via a couple of stride-2 stages only.
      - Depthwise-separable convs keep it cheap while giving each output token
        a receptive field over a small ppm neighborhood (local peak shape).
    """

    def __init__(self, in_channels: int, dims=(64, 128, 256), strides=(1, 2, 2)):
        super().__init__()
        layers = []
        c_prev = in_channels
        for c, s in zip(dims, strides):
            layers += [
                nn.Conv1d(c_prev, c, kernel_size=7, stride=s, padding=3, bias=False),
                nn.BatchNorm1d(c),
                nn.SiLU(inplace=True),
                nn.Conv1d(c, c, kernel_size=5, stride=1, padding=2, groups=c, bias=False),
                nn.BatchNorm1d(c),
                nn.SiLU(inplace=True),
            ]
            c_prev = c
        self.net = nn.Sequential(*layers)
        self.out_channels = dims[-1]
        self.total_stride = 1
        for s in strides:
            self.total_stride *= s

    def forward(self, x):  # (B, C_in, L) -> (B, C_out, L/total_stride)
        return self.net(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal PE over the sequence (ppm) axis."""

    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x):  # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


class AttentionPool(nn.Module):
    """
    Learnable attention pooling: a query token attends over all sequence
    positions and produces one vector. Unlike AdaptiveAvgPool this LEARNS
    which peaks matter and preserves positional information (via PE on the
    tokens), which matters for 13C where position is signal.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):  # (B, T, D) -> (B, D)
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


class ContrastiveCNmrBackbone(nn.Module):
    """
    13C-NMR encoder for CLIP-style contrastive alignment with molecule embeddings.

    Pipeline:  spectrum (B,1,L)
        -> light conv stem (local peak shape, gentle downsampling)
        -> tokens + positional encoding
        -> transformer encoder (global peak-relationship reasoning)
        -> attention pooling (learned, position-aware)
        -> projection head -> L2-normalized embedding
    """

    def __init__(
        self,
        input_length: int = 8192,
        input_channels: int = 1,
        embed_dim: int = 256,
        proj_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        stem_dims=(64, 128, 256),
        stem_strides=(1, 2, 2),
    ):
        super().__init__()

        self.stem = ConvStem(input_channels, dims=stem_dims, strides=stem_strides)
        if self.stem.out_channels != embed_dim:
            self.proj_in = nn.Conv1d(self.stem.out_channels, embed_dim, 1)
        else:
            self.proj_in = nn.Identity()

        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=input_length)

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
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.stem(x)
        x = self.proj_in(x)
        x = x.transpose(1, 2)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.encoder_norm(x)
        x = self.pool(x)
        x = self.head(x)
        if normalize:
            x = F.normalize(x, dim=-1)
        return x


class cNmrEncoder(nn.Module):
    """
    API-compatible wrapper (matches the original cNmrEncoder signature).

    Args:
        ckpt_path:      optional path to a checkpoint to load into the backbone.
                        Accepts either a raw state_dict or a dict containing
                        'state_dict'/'model'/'encoder' keys; a leading 'encoder.'
                        prefix on keys is stripped automatically.
        freeze_encoder: if True, freezes all backbone params (requires_grad=False)
                        and keeps it in eval mode so BN/dropout don't update.
    """

    def __init__(
        self,
        ckpt_path: str | None = None,
        freeze_encoder: bool = False,
        **backbone_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = ContrastiveCNmrBackbone(**backbone_kwargs)
        self.frozen = freeze_encoder

        if ckpt_path is not None:
            self._load_ckpt(ckpt_path)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def _load_ckpt(self, ckpt_path: str) -> None:
        state = torch.load(ckpt_path, map_location="cpu")
        # unwrap common container keys
        for key in ("state_dict", "model", "encoder"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
        # strip a leading "encoder." prefix if the ckpt was saved from the wrapper
        cleaned = {}
        for k, v in state.items():
            cleaned[k[len("encoder."):] if k.startswith("encoder.") else k] = v
        missing, unexpected = self.encoder.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[cNmrEncoder] missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[cNmrEncoder] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    def train(self, mode: bool = True):
        """Keep a frozen encoder in eval mode even when the parent is train()'d."""
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                return self.encoder(x, normalize=normalize)
        return self.encoder(x, normalize=normalize)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = cNmrEncoder(ckpt_path=None, freeze_encoder=False).to(device)
    model.eval()
    x = torch.randn(4, 1, 4096, device=device)
    with torch.no_grad():
        z = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(z.shape)}  (L2-normalized)")
    print(f"Row norms: {z.norm(dim=-1)}")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # frozen check
    frozen = cNmrEncoder(ckpt_path=None, freeze_encoder=True).to(device)
    n_trainable = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
    print(f"Frozen trainable params: {n_trainable}")