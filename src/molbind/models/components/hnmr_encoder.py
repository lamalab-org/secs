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
    """Highly scalable CNN encoder with compound scaling and an optional initial residual connection."""

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
        # --- NEW: Initial Residual Connection Parameters ---
        use_initial_residual: bool = True,
        residual_projection_dim: int = 128,
    ):
        super().__init__()

        # --- NEW: Store residual connection flag ---
        self.use_initial_residual = use_initial_residual

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
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        self.channels = [make_divisible(max(8, int(ch * width_mult))) for ch in base_channels]
        self.blocks_per_stage = [max(1, int(blocks * depth_mult)) for blocks in blocks_per_stage]
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
            stage.append(
                ResidualBlock1D(in_channels, out_channels, stride=stride, dropout=dropout, use_se=use_se, activation=activation)
            )
            for _ in range(num_blocks - 1):
                stage.append(
                    ResidualBlock1D(out_channels, out_channels, stride=1, dropout=dropout, use_se=use_se, activation=activation)
                )
            self.stages.append(nn.Sequential(*stage))
            in_channels = out_channels

        # Multi-head self-attention (optional)
        final_channels = self.channels[-1]
        if use_attention:
            max_heads = final_channels // 64 if final_channels >= 64 else 1
            actual_heads = min(attention_heads, max_heads)
            for h in range(actual_heads, 0, -1):
                if final_channels % h == 0:
                    actual_heads = h
                    break
            self.attention = nn.MultiheadAttention(
                embed_dim=final_channels, num_heads=actual_heads, dropout=dropout, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(final_channels)

        # This path processes the raw input to create a feature vector to be concatenated
        # with the deep features from the main CNN body.
        self.initial_residual_processor = None
        if self.use_initial_residual:
            # We use pooling to reduce the long sequence to a manageable size, then a linear layer.
            pooled_length = 256  # A hyperparameter to control the size after pooling
            self.initial_residual_processor = nn.Sequential(
                nn.AdaptiveAvgPool1d(pooled_length),
                nn.Flatten(),
                nn.Linear(input_channels * pooled_length, residual_projection_dim),
                nn.LayerNorm(residual_projection_dim),
                nn.SiLU() if activation == "swish" else nn.ReLU(),
            )

        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # The input dimension to the head now depends on whether we use the initial residual.
        head_input_dim = final_channels
        if self.use_initial_residual:
            head_input_dim += residual_projection_dim

        head_dim = head_input_dim * 2  # Double for richer representation
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, head_dim),
            nn.BatchNorm1d(head_dim),
            nn.SiLU(inplace=True) if activation == "swish" else nn.ReLU(inplace=True),
            nn.Dropout(dropout * 2),
            nn.Linear(head_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.SiLU(inplace=True) if activation == "swish" else nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using modern techniques"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- NEW: Store initial input for the residual path ---
        initial_x = x if self.use_initial_residual else None

        # Main CNN Path
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)

        # Optional attention
        if self.use_attention:
            x_att = x.transpose(1, 2)
            x_att_out, _ = self.attention(x_att, x_att, x_att)
            x_att = self.attention_norm(x_att + x_att_out)
            x = x_att.transpose(1, 2)

        # Global pooling for deep features
        deep_features = self.global_pool(x).squeeze(-1)  # (B, C_final)

        # --- NEW: Process and combine features ---
        if self.use_initial_residual and self.initial_residual_processor is not None:
            # Process the original input through the residual path
            residual_features = self.initial_residual_processor(initial_x)  # (B, residual_projection_dim)
            # Concatenate deep features with the initial residual features
            combined_features = torch.cat((deep_features, residual_features), dim=1)
        else:
            combined_features = deep_features

        # Pass combined features to the final head
        return self.head(combined_features)


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
            "use_initial_residual": True,  # Enable for tiny model
        },
        "small": {
            "width_mult": 0.75,
            "depth_mult": 0.75,
            "base_channels": [32, 64, 128, 256, 512],
            "blocks_per_stage": [2, 2, 3, 4, 2],
            "dropout": 0.1,
            "use_attention": True,
            "attention_heads": 4,
            "use_initial_residual": True,
        },
        "medium": {
            "width_mult": 1.0,
            "depth_mult": 1.0,
            "base_channels": [64, 128, 256, 512, 512],
            "blocks_per_stage": [2, 3, 4, 6, 3],
            "dropout": 0.15,
            "use_attention": True,
            "attention_heads": 8,
            "use_initial_residual": True,
        },
        "large": {
            "width_mult": 1.5,
            "depth_mult": 1.25,
            "base_channels": [64, 128, 256, 512, 1024],
            "blocks_per_stage": [3, 4, 6, 8, 4],
            "dropout": 0.2,
            "use_attention": True,
            "attention_heads": 16,
            "use_initial_residual": True,
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
            "use_initial_residual": True,  # Enable by default for xlarge
        },
        "efficient_b0": {"compound_coeff": 0, "use_initial_residual": True},
        "efficient_b1": {"compound_coeff": 0.5, "use_initial_residual": True},
        "efficient_b2": {"compound_coeff": 1.0, "use_initial_residual": True},
        "efficient_b3": {"compound_coeff": 1.5, "use_initial_residual": True},
        "efficient_b4": {"compound_coeff": 2.0, "use_initial_residual": True},
        "efficient_b5": {"compound_coeff": 2.5, "use_initial_residual": True},
        "efficient_b6": {"compound_coeff": 3.0, "use_initial_residual": True},
        "efficient_b7": {"compound_coeff": 3.5, "use_initial_residual": True},
    }

    config = configs.get(scale, configs["medium"])
    config.update(kwargs)

    return ScalableCNNEncoder(**config)


class hNmrCNNEncoder(nn.Module):
    # --- MODIFIED: Added flag to control the initial residual connection ---
    def __init__(self, ckpt_path: str | None = None, freeze_encoder: bool = False, use_initial_residual: bool = False) -> None:
        super().__init__()
        # Pass the flag to the model factory
        self.encoder = get_scaled_model("xlarge", use_initial_residual=use_initial_residual)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


if __name__ == "__main__":
    # Example usage
    # The hNmrCNNEncoder now enables the initial residual connection by default.
    # You can disable it by passing use_initial_residual=False
    model = hNmrCNNEncoder(ckpt_path=None, freeze_encoder=False, use_initial_residual=True).to("cuda")
    model.eval()
    print(model)  # Print the model to see the new `initial_residual_processor`

    input_tensor = torch.randn(4, 1, 10000)  # Batch size 4, 1 channel, length 10000
    output = model(input_tensor.to("cuda"))
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # Should be (4, 1024) for latent_dim=1024

    # Example of disabling the feature
    print("\n--- Model without initial residual connection ---")
    model_no_residual = hNmrCNNEncoder(use_initial_residual=False).to("cuda")
    model_no_residual.eval()
    output_no_residual = model_no_residual(input_tensor.to("cuda"))
    print(f"Output shape (no residual): {output_no_residual.shape}")
