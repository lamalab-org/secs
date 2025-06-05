import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Enhanced 1D Residual block with configurable expansion"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
        expansion: int = 4,
        use_se: bool = True,
        activation: str = "relu",
    ):
        super().__init__()

        # Bottleneck design with expansion
        mid_channels = out_channels // expansion if expansion > 1 else out_channels

        # Choose activation function
        if activation == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif activation == "swish":
            act_fn = nn.SiLU(inplace=True)
        elif activation == "gelu":
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU(inplace=True)

        # Bottleneck layers
        layers = []
        if expansion > 1:
            # 1x1 conv to reduce channels
            layers.extend([nn.Conv1d(in_channels, mid_channels, 1, bias=False), nn.BatchNorm1d(mid_channels), act_fn])
        else:
            mid_channels = in_channels

        # 3x3 conv
        layers.extend(
            [
                nn.Conv1d(mid_channels, mid_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(mid_channels),
                act_fn,
                nn.Dropout(dropout),
            ]
        )

        # 1x1 conv to expand channels
        if expansion > 1:
            layers.extend([nn.Conv1d(mid_channels, out_channels, 1, bias=False), nn.BatchNorm1d(out_channels)])

        self.conv_layers = nn.Sequential(*layers)

        # Squeeze-and-Excitation
        self.se = None
        if use_se:
            se_channels = max(1, out_channels // 16)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(out_channels, se_channels, 1),
                act_fn,
                nn.Conv1d(se_channels, out_channels, 1),
                nn.Sigmoid(),
            )

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels)
            )

        self.act_fn = act_fn

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv_layers(x)

        # Apply SE if present
        if self.se is not None:
            out = out * self.se(out)

        out += residual
        return self.act_fn(out)


class ScalableCNNEncoder(nn.Module):
    """Highly scalable CNN encoder with compound scaling"""

    def __init__(
        self,
        input_length: int = 10000,
        input_channels: int = 1,
        latent_dim: int = 1024,
        # Scaling parameters
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        resolution_mult: float = 1.0,
        compound_coeff: float = 0,
        base_channels: list[int] | None = None,
        blocks_per_stage: list[int] | None = None,
        strides: list[int] | None = None,
        dropout: float = 0.1,
        use_attention: bool = True,
        attention_heads: int = 8,
        use_se: bool = True,
        activation: str = "swish",
        stem_type: str = "conv",
    ):
        super().__init__()

        # Apply compound scaling (EfficientNet style)
        if compound_coeff > 0:
            width_mult *= 1.1**compound_coeff
            depth_mult *= 1.2**compound_coeff
            resolution_mult *= 1.15**compound_coeff

        # Default configurations
        if base_channels is None:
            base_channels = [32, 64, 128, 256, 512]
        if blocks_per_stage is None:
            blocks_per_stage = [2, 3, 4, 6, 3]
        if strides is None:
            strides = [1, 2, 2, 2, 2]

        # Scale channels by width multiplier and ensure they're divisible by common factors
        def make_divisible(v, divisor=8):
            """Make channel count divisible by divisor for efficiency"""
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        self.channels = [make_divisible(max(8, int(ch * width_mult))) for ch in base_channels]

        # Scale depth by depth multiplier
        self.blocks_per_stage = [max(1, int(blocks * depth_mult)) for blocks in blocks_per_stage]

        # Scale input resolution
        self.input_length = int(input_length * resolution_mult)
        self.use_attention = use_attention

        # Stem (initial feature extraction)
        if stem_type == "conv":
            self.stem = nn.Sequential(
                nn.Conv1d(input_channels, self.channels[0], 7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(self.channels[0]),
                nn.SiLU(inplace=True) if activation == "swish" else nn.ReLU(inplace=True),
                nn.MaxPool1d(3, stride=2, padding=1),
            )
        else:  # patch embedding style
            patch_size = 16
            self.stem = nn.Sequential(
                nn.Conv1d(input_channels, self.channels[0], patch_size, stride=patch_size // 2),
                nn.BatchNorm1d(self.channels[0]),
                nn.SiLU(inplace=True) if activation == "swish" else nn.ReLU(inplace=True),
            )

        # Build stages
        self.stages = nn.ModuleList()
        in_channels = self.channels[0]

        for _, (out_channels, num_blocks, stride) in enumerate(
            zip(self.channels[1:], self.blocks_per_stage[1:], strides[1:], strict=False)
        ):
            stage = []

            # First block with stride
            stage.append(
                ResidualBlock1D(in_channels, out_channels, stride=stride, dropout=dropout, use_se=use_se, activation=activation)
            )

            # Remaining blocks
            for _ in range(num_blocks - 1):
                stage.append(
                    ResidualBlock1D(out_channels, out_channels, stride=1, dropout=dropout, use_se=use_se, activation=activation)
                )

            self.stages.append(nn.Sequential(*stage))
            in_channels = out_channels

        # Multi-head self-attention (optional)
        final_channels = self.channels[-1]
        if use_attention:
            # Ensure num_heads divides embed_dim evenly
            max_heads = final_channels // 64 if final_channels >= 64 else 1
            actual_heads = min(attention_heads, max_heads)

            # Find largest divisor of final_channels that's <= actual_heads
            for h in range(actual_heads, 0, -1):
                if final_channels % h == 0:
                    actual_heads = h
                    break

            self.attention = nn.MultiheadAttention(
                embed_dim=final_channels, num_heads=actual_heads, dropout=dropout, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(final_channels)

        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier head with multiple scales
        head_dim = final_channels * 2  # Double for richer representation
        self.head = nn.Sequential(
            nn.Linear(final_channels, head_dim),
            nn.BatchNorm1d(head_dim),
            nn.SiLU(inplace=True) if activation == "swish" else nn.ReLU(inplace=True),
            nn.Dropout(dropout * 2),  # Higher dropout in head
            nn.Linear(head_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.SiLU(inplace=True) if activation == "swish" else nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid(),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using modern techniques"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        # Optional attention
        if self.use_attention:
            # Reshape for attention: (batch, seq_len, features)
            x_att = x.transpose(1, 2)  # (B, L, C)

            # Self-attention with residual connection
            x_att_out, _ = self.attention(x_att, x_att, x_att)
            x_att = self.attention_norm(x_att + x_att_out)

            x = x_att.transpose(1, 2)  # Back to (B, C, L)

        # Global pooling and classification
        x = self.global_pool(x).squeeze(-1)  # (B, C)
        return self.head(x)


# Predefined scaling configurations
def get_scaled_model(scale: str = "small", **kwargs):
    """Get predefined scaled models"""

    configs = {
        "tiny": {
            "width_mult": 0.5,
            "depth_mult": 0.5,
            "base_channels": [16, 32, 64, 128, 256],
            "blocks_per_stage": [1, 2, 2, 3, 2],
            "dropout": 0.05,
            "use_attention": False,
            "use_se": False,
        },
        "small": {
            "width_mult": 0.75,
            "depth_mult": 0.75,
            "base_channels": [32, 64, 128, 256, 512],
            "blocks_per_stage": [2, 2, 3, 4, 2],
            "dropout": 0.1,
            "use_attention": True,
            "attention_heads": 4,
        },
        "medium": {
            "width_mult": 1.0,
            "depth_mult": 1.0,
            "base_channels": [64, 128, 256, 512, 1024],
            "blocks_per_stage": [2, 3, 4, 6, 3],
            "dropout": 0.15,
            "use_attention": True,
            "attention_heads": 8,
        },
        "large": {
            "width_mult": 1.5,
            "depth_mult": 1.25,
            "base_channels": [64, 128, 256, 512, 1024],
            "blocks_per_stage": [3, 4, 6, 8, 4],
            "dropout": 0.2,
            "use_attention": True,
            "attention_heads": 16,
        },
        "xlarge": {
            "width_mult": 2.0,
            "depth_mult": 1.5,
            "base_channels": [64, 128, 256, 512, 1024],
            "blocks_per_stage": [3, 4, 8, 12, 6],
            "dropout": 0.25,
            "use_attention": True,
            "attention_heads": 32,
            "activation": "swish",
        },
        "efficient_b0": {"compound_coeff": 0},
        "efficient_b1": {"compound_coeff": 0.5},
        "efficient_b2": {"compound_coeff": 1.0},
        "efficient_b3": {"compound_coeff": 1.5},
        "efficient_b4": {"compound_coeff": 2.0},
        "efficient_b5": {"compound_coeff": 2.5},
        "efficient_b6": {"compound_coeff": 3.0},
        "efficient_b7": {"compound_coeff": 3.5},
    }

    config = configs.get(scale, configs["medium"])
    config.update(kwargs)

    return ScalableCNNEncoder(**config)


class hNmrCNNEncoder(nn.Module):
    def __init__(self, ckpt_path: str | None = None, freeze_encoder: bool = False) -> None:
        super().__init__()
        self.encoder = get_scaled_model("xlarge")
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if ckpt_path:
            self.encoder.load_state_dict(torch.load(ckpt_path, map_location="cuda")["state_dict"], strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
