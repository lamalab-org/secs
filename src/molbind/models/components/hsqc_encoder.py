import torch
import torch.nn as nn


class ResidualBlock2D(nn.Module):
    """Enhanced 2D Residual block with configurable expansion"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,  # Assumed to be square
        stride: int | tuple[int, int] = 1,
        dropout: float = 0.1,
        expansion: int = 4,
        use_se: bool = True,
        activation: str = "relu",
    ):
        super().__init__()

        # Choose activation function
        if activation == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif activation == "swish":
            act_fn = nn.SiLU(inplace=True)
        elif activation == "gelu":
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU(inplace=True)  # Default to ReLU
        self.final_act_fn = act_fn  # Store for final activation after residual sum

        _stride_tuple = (stride, stride) if isinstance(stride, int) else stride
        # Kernel_size for Conv2d can be an int (for square) or a tuple
        _kernel_arg = kernel_size  # Pass as int or tuple directly to Conv2d
        _padding_arg = kernel_size // 2 if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2)

        conv_layers_modules = []
        if expansion > 1:
            bottleneck_channels = out_channels // expansion
            # 1x1 conv to reduce/project channels
            conv_layers_modules.extend(
                [
                    nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(bottleneck_channels),
                    act_fn,
                ]
            )
            # KxK conv
            conv_layers_modules.extend(
                [
                    nn.Conv2d(
                        bottleneck_channels,
                        bottleneck_channels,
                        kernel_size=_kernel_arg,
                        stride=_stride_tuple,
                        padding=_padding_arg,
                        bias=False,
                    ),
                    nn.BatchNorm2d(bottleneck_channels),
                    act_fn,
                    nn.Dropout(dropout),  # Original uses nn.Dropout
                ]
            )
            # 1x1 conv to expand channels
            conv_layers_modules.extend(
                [
                    nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),  # BN before residual add
                ]
            )
        else:  # expansion == 1 (Basic block like, single main conv)
            conv_layers_modules.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=_kernel_arg,
                        stride=_stride_tuple,
                        padding=_padding_arg,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    act_fn,
                    nn.Dropout(dropout),
                ]
            )

        self.conv_path = nn.Sequential(*conv_layers_modules)

        # Squeeze-and-Excitation
        self.se_module = None
        if use_se:
            se_channels = max(1, out_channels // 16)
            self.se_module = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Global pool for 2D
                nn.Conv2d(out_channels, se_channels, kernel_size=1),  # 1x1 conv
                act_fn,  # Re-use main activation
                nn.Conv2d(se_channels, out_channels, kernel_size=1),  # 1x1 conv
                nn.Sigmoid(),
            )

        # Skip connection
        self.skip_connection = nn.Sequential()
        is_strided = _stride_tuple[0] != 1 or _stride_tuple[1] != 1
        if is_strided or (in_channels != out_channels):
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=_stride_tuple, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_connection(x)
        out = self.conv_path(x)

        if self.se_module is not None:
            scale = self.se_module(out)  # Output shape (B, C, 1, 1)
            out = out * scale  # Broadcast scale over H, W

        out += residual
        return self.final_act_fn(out)


class ScalableCNNEncoder2D(nn.Module):
    """Highly scalable 2D CNN encoder with compound scaling"""

    def __init__(
        self,
        input_height: int = 512,
        input_width: int = 512,
        input_channels: int = 1,
        latent_dim: int = 1024,
        # Scaling parameters
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        resolution_mult: float = 1.0,
        compound_coeff: float = 0,
        base_channels: list[int] | None = None,
        blocks_per_stage: list[int] | None = None,
        strides: list[int] | None = None,  # Strides for each stage, applied as (s,s)
        dropout: float = 0.1,
        use_attention: bool = True,
        attention_heads: int = 8,
        use_se: bool = True,
        activation: str = "swish",
        stem_type: str = "conv",  # "conv" or "patch"
        block_expansion: int = 4,  # Expansion factor for residual blocks
    ):
        super().__init__()

        if compound_coeff > 0:
            width_mult *= 1.1**compound_coeff
            depth_mult *= 1.2**compound_coeff
            resolution_mult *= 1.15**compound_coeff

        if base_channels is None:
            base_channels = [64, 128, 256, 512, 1024]  # Default from original "medium"
        if blocks_per_stage is None:
            blocks_per_stage = [2, 3, 4, 6, 3]  # Default from original "medium"
        if strides is None:
            strides = [1, 2, 2, 2, 2]  # Default from original "medium"

        def make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        self.channels = [make_divisible(max(8, int(ch * width_mult))) for ch in base_channels]
        self.blocks_per_stage = [max(1, int(blocks * depth_mult)) for blocks in blocks_per_stage]
        self.strides = strides  # Strides are not scaled by depth_mult in original

        self.input_height = int(input_height * resolution_mult)
        self.input_width = int(input_width * resolution_mult)
        self.use_attention = use_attention

        if activation == "swish":
            act_fn_general = nn.SiLU(inplace=True)
        elif activation == "relu":
            act_fn_general = nn.ReLU(inplace=True)
        elif activation == "gelu":
            act_fn_general = nn.GELU()
        else:
            act_fn_general = nn.SiLU(inplace=True)

        stem_out_channels = self.channels[0]
        if stem_type == "conv":
            self.stem = nn.Sequential(
                nn.Conv2d(input_channels, stem_out_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(stem_out_channels),
                act_fn_general,
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:  # patch embedding style
            patch_size = 16
            self.stem = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    stem_out_channels,
                    kernel_size=patch_size,
                    stride=patch_size // 2,
                    padding=patch_size // 4,
                    bias=False,
                ),  # Added bias=False for consistency
                nn.BatchNorm2d(stem_out_channels),
                act_fn_general,
            )

        self.stages = nn.ModuleList()
        current_in_channels = stem_out_channels
        # Loop structure assumes self.channels[0] is stem output,
        # self.blocks_per_stage[0] and self.strides[0] are for the first stage after stem.
        # The original code's loop `zip(self.channels[1:], self.blocks_per_stage[1:], strides[1:])`
        # implies that the 0-th element of blocks_per_stage and strides in the config is ignored.
        # We will follow that logic for consistency.
        for i in range(len(self.channels) - 1):  # Number of stages = len(self.channels) - 1
            stage_out_channels = self.channels[i + 1]
            # Use blocks_per_stage[i+1] and strides[i+1] if following original indexing logic
            # where config lists have a dummy 0-th element or are 1-indexed for stages.
            # If config lists are 0-indexed for stages (more natural):
            #   num_blocks = self.blocks_per_stage[i]
            #   stage_stride = self.strides[i]
            # Given the original config structure, let's assume the lists are 0-indexed for stages,
            # and self.channels[0] is stem, self.channels[1] is stage 0 output, etc.
            num_blocks = self.blocks_per_stage[i]  # Use i-th element for i-th stage
            stage_stride = self.strides[i]  # Use i-th element for i-th stage

            stage_layers = []
            stage_layers.append(
                ResidualBlock2D(
                    current_in_channels,
                    stage_out_channels,
                    stride=stage_stride,
                    dropout=dropout,
                    use_se=use_se,
                    activation=activation,
                    expansion=block_expansion,
                )
            )
            for _ in range(num_blocks - 1):
                stage_layers.append(
                    ResidualBlock2D(
                        stage_out_channels,
                        stage_out_channels,
                        stride=1,
                        dropout=dropout,
                        use_se=use_se,
                        activation=activation,
                        expansion=block_expansion,
                    )
                )
            self.stages.append(nn.Sequential(*stage_layers))
            current_in_channels = stage_out_channels

        final_channels = current_in_channels  # Should be self.channels[-1]
        self.attention = None
        self.attention_norm = None
        if use_attention:
            max_h = final_channels // 64 if final_channels >= 64 else 1
            actual_h = min(attention_heads, max_h)
            if actual_h > 0:
                # Find largest divisor of final_channels that's <= actual_h
                for h_candidate in range(actual_h, 0, -1):
                    if final_channels % h_candidate == 0:
                        actual_h = h_candidate
                        break
                if final_channels % actual_h == 0:  # Ensure it's divisible
                    self.attention = nn.MultiheadAttention(
                        embed_dim=final_channels, num_heads=actual_h, dropout=dropout, batch_first=True
                    )
                    self.attention_norm = nn.LayerNorm(final_channels)
                else:
                    self.use_attention = False  # Could not set up attention
            else:
                self.use_attention = False

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        head_input_dim = final_channels
        head_bottleneck_dim = head_input_dim * 2
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, head_bottleneck_dim),
            nn.BatchNorm1d(head_bottleneck_dim),
            act_fn_general,
            nn.Dropout(dropout * 2),
            nn.Linear(head_bottleneck_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            act_fn_general,
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid(),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d | nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage_module in self.stages:
            x = stage_module(x)

        if self.use_attention and self.attention is not None:
            B, C, H, W = x.shape
            x_att = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
            att_output, _ = self.attention(x_att, x_att, x_att)
            x_att = self.attention_norm(x_att + att_output)  # Post-norm
            x = x_att.transpose(1, 2).unflatten(2, (H, W))

        x = self.global_pool(x)
        x = x.flatten(start_dim=1)  # (B, C)
        return self.head(x)


def get_scaled_model_2d(scale: str = "small", **kwargs):
    configs = {
        "tiny": {
            "width_mult": 0.5,
            "depth_mult": 0.5,
            "base_channels": [16, 32, 64, 128],
            "blocks_per_stage": [1, 2, 2, 1],
            "strides": [1, 2, 2, 2],
            "dropout": 0.05,
            "use_attention": False,
            "use_se": False,
            "block_expansion": 2,
        },
        "small": {
            "width_mult": 0.75,
            "depth_mult": 0.75,
            "base_channels": [32, 48, 96, 192, 384],
            "blocks_per_stage": [2, 2, 3, 3, 3],
            "strides": [1, 2, 2, 2, 2],
            "dropout": 0.1,
            "use_attention": True,
            "attention_heads": 4,
            "block_expansion": 4,
        },
        "medium": {
            "width_mult": 1.0,
            "depth_mult": 1.0,
            "base_channels": [64, 128, 256, 512, 768],
            "blocks_per_stage": [2, 3, 4, 6, 3],
            "strides": [1, 2, 2, 2, 2],
            "dropout": 0.15,
            "use_attention": True,
            "attention_heads": 8,
            "block_expansion": 4,
        },
        "large": {
            "width_mult": 1.5,
            "depth_mult": 1.25,
            "base_channels": [64, 128, 256, 512, 1024],
            "blocks_per_stage": [3, 4, 6, 8, 4],
            "strides": [1, 2, 2, 2, 2],
            "dropout": 0.2,
            "use_attention": True,
            "attention_heads": 16,
            "block_expansion": 4,
        },
        "xlarge": {
            "width_mult": 2.0,
            "depth_mult": 1.5,
            "base_channels": [80, 160, 320, 640, 1280],
            "blocks_per_stage": [3, 4, 8, 12, 6],
            "strides": [1, 2, 2, 2, 2],
            "dropout": 0.25,
            "use_attention": True,
            "attention_heads": 16,
            "activation": "swish",
            "block_expansion": 6,
        },  # Example, heads might need adjustment
        # EfficientNet-style compound scaling. These primarily set compound_coeff.
        # Other params (base_channels, etc.) will default to "medium" if not overridden by compound scaling logic or kwargs.
        "efficient_b0": {
            "compound_coeff": 0,
            "base_channels": [32, 40, 80, 112, 192],
            "blocks_per_stage": [1, 2, 2, 3, 3],
            "strides": [1, 2, 2, 1, 2],
            "block_expansion": 6,
            "activation": "swish",
        },  # Example B0-like
        "efficient_b1": {"compound_coeff": 0.5},
        "efficient_b2": {"compound_coeff": 1.0},
        "efficient_b3": {"compound_coeff": 1.5},
        "efficient_b4": {"compound_coeff": 2.0},
        "efficient_b5": {"compound_coeff": 2.5},
        "efficient_b6": {"compound_coeff": 3.0},
        "efficient_b7": {"compound_coeff": 3.5},
    }
    # Refined config handling:
    # Start with a base (e.g., "medium" or a specific one for "efficient_bX")
    # Then apply scale-specific overrides, then kwargs.
    if scale.startswith("efficient_b"):
        # For EfficientNet, usually specific base params are defined for B0, then scaled.
        # Here, we'll use "efficient_b0" as a base for other Bx if they only set compound_coeff.
        # Or, more simply, use "medium" as a general base that compound_coeff modifies.
        base_config_name = "efficient_b0" if scale == "efficient_b0" else "medium"
        final_config = configs[base_config_name].copy()
        if scale != base_config_name and scale in configs:  # Apply specific Bx overrides if any (e.g. compound_coeff)
            final_config.update(configs[scale])
    else:
        final_config = configs.get(scale, configs["medium"].copy())

    final_config.update(kwargs)

    # Ensure input_height/width are present
    if "input_height" not in final_config:
        final_config["input_height"] = 512
    if "input_width" not in final_config:
        final_config["input_width"] = 512

    return ScalableCNNEncoder2D(**final_config)


class HSQCEncoder(nn.Module):
    def __init__(self, ckpt_path: str | None = None, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = get_scaled_model_2d(
            scale="xlarge",
            input_height=512,
            input_width=512,
            input_channels=1,  # For a single-channel 512x512 image
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if ckpt_path:
            # map_location='cpu' is safer for loading then moving to device
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # Check if state_dict is nested (e.g. under "state_dict" or "model")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

            self.encoder.load_state_dict(state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input x: (Batch, 1, 512, 512)
        return self.encoder(x)
