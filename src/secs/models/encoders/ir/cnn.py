import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig

from secs.models.base import ModalityEncoder
from secs.models.registry import register_encoder
from secs.utils import rename_keys_with_prefix, select_device


@register_encoder("ir", "cnn", default=True)
class IrCNNEncoder(ModalityEncoder):
    """1D CNN over a 1600-bin IR spectrum."""

    def __init__(self, ckpt_path: str | None = None, freeze_encoder: bool = False) -> None:
        super().__init__(ckpt_path=ckpt_path, freeze_encoder=freeze_encoder)

        cfg = DictConfig(
            {
                "length_in": 1600,
                "channels_in": 1,
                "channels_out_1": 45,
                "channels_out_2": 45,
                "channels_out_3": 90,
                "channels_out_4": 90,
                "latent_dim": 512,
                "num_conv_layers": 3,
                "num_fc_layers": 2,
                "conv_kernel_dim_1": 10,
                "conv_stride_1": 1,
                "conv_padding_1": 0,
                "conv_dilation": 1,
                "pool_kernel_dim_1": 5,
                "pool_stride_1": 5,
                "pool_padding_1": 0,
                "pool_dilation": 1,
                "conv_kernel_dim_2": 5,
                "conv_stride_2": 1,
                "conv_padding_2": 0,
                "pool_kernel_dim_2": 3,
                "pool_stride_2": 2,
                "pool_padding_2": 0,
                "conv_kernel_dim_3": 5,
                "conv_stride_3": 1,
                "conv_padding_3": 0,
                "pool_kernel_dim_3": 2,
                "pool_stride_3": 1,
                "pool_padding_3": 0,
                "fc_dim_1": 512,
                "fc_dim_2": 512,
                "conv_kernel_dim_4": 4,
                "conv_stride_4": 1,
            }
        )
        logger.info(cfg)
        self.cfg = cfg
        # Dynamically calculate output dimensions
        self.out_dims = self._calculate_out_dims()

        # Create layers dynamically based on config
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        in_channels = cfg.channels_in
        for i in range(cfg.num_conv_layers):
            out_channels = cfg[f"channels_out_{i + 1}"]
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    cfg[f"conv_kernel_dim_{i + 1}"],
                    stride=cfg[f"conv_stride_{i + 1}"],
                    padding=cfg[f"conv_padding_{i + 1}"],
                    dilation=cfg.conv_dilation,
                )
            )
            self.norm_layers.append(nn.BatchNorm1d(out_channels))
            self.pool_layers.append(
                nn.AvgPool1d(
                    cfg[f"pool_kernel_dim_{i + 1}"],
                    stride=cfg[f"pool_stride_{i + 1}"],
                    padding=cfg[f"pool_padding_{i + 1}"],
                    # dilation=cfg.pool_dilation,
                )
            )
            in_channels = out_channels

        self.fc_layers = nn.ModuleList()

        self.last_conv_out_dim = self._conv_out_dim(
            self.out_dims[-1],
            cfg.conv_kernel_dim_4,
            cfg.conv_stride_4,
            cfg.conv_padding_3,
            cfg.conv_dilation,
        )
        self.last_conv = nn.Conv1d(
            cfg.channels_out_3,
            cfg.channels_out_4,
            cfg.conv_kernel_dim_4,
            stride=cfg.conv_stride_4,
            padding=cfg.conv_padding_3,
            dilation=cfg.conv_dilation,
        )
        fc_in_dim = self.last_conv_out_dim * cfg.channels_out_4
        for i in range(cfg.num_fc_layers - 1):
            fc_out_dim = cfg[f"fc_dim_{i + 1}"]
            self.fc_layers.append(nn.Linear(fc_in_dim, fc_out_dim))
            self.fc_layers.append(nn.BatchNorm1d(fc_out_dim))
            self.fc_layers.append(nn.ReLU())
            fc_in_dim = fc_out_dim
        self.fc_layers.append(nn.Linear(fc_in_dim, cfg.latent_dim))

        self.output_dim = cfg.latent_dim
        self._finalize()

    def _calculate_out_dims(self):
        out_dims = [self.cfg.length_in]
        for i in range(self.cfg.num_conv_layers):
            out_dims.append(
                self._conv_out_dim(
                    out_dims[-1],
                    self.cfg[f"conv_kernel_dim_{i + 1}"],
                    self.cfg[f"conv_stride_{i + 1}"],
                    self.cfg[f"conv_padding_{i + 1}"],
                    self.cfg.conv_dilation,
                )
            )
            out_dims.append(
                self._conv_out_dim(
                    out_dims[-1],
                    self.cfg[f"pool_kernel_dim_{i + 1}"],
                    self.cfg[f"pool_stride_{i + 1}"],
                    self.cfg[f"pool_padding_{i + 1}"],
                    self.cfg.pool_dilation,
                )
            )
        return out_dims

    @staticmethod
    def _conv_out_dim(dim_in, kernel_size, stride, padding, dilation):
        return (dim_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def load_checkpoint(self, ckpt_path: str, strict: bool = True) -> None:  # noqa: ARG002
        # The layers live on the encoder itself here, not under `self.encoder`.
        self.load_state_dict(rename_keys_with_prefix(torch.load(ckpt_path, map_location=select_device())["state_dict"]))

    def freeze(self) -> None:
        self.frozen = True
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        return nn.Module.train(self, mode and not self.frozen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, norm, pool in zip(self.conv_layers, self.norm_layers, self.pool_layers, strict=False):
            x = pool(F.relu(norm(conv(x))))
        x = self.last_conv(x)
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers[:-1]:
            x = layer(x)

        return torch.sigmoid(self.fc_layers[-1](x))
