import torch  # noqa: I002
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

"""
Reference to the original paper:
    An end-to-end deep learning framework for translating mass spectra to de-novo molecules:
    https://www.nature.com/articles/s42004-023-00932-3
Reference to the original implementation:
    Model: https://github.com/KavrakiLab/Spec2Mol/blob/master/model1D2Conv.py
    Config: https://github.com/KavrakiLab/Spec2Mol/blob/master/train_MS_encoder.py
    Data: --NOT-PROVIDED--COMMERCIAL-DATASET--
"""


def conv_out_dim(
    length_in: int, kernel: int, stride: int, padding: int, dilation: int
) -> int:
    return (length_in + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


class Net1D(nn.Module):
    def __init__(self, cfg: DictConfig, length_in: int) -> None:
        super().__init__()
        out_1 = conv_out_dim(
            length_in,
            cfg.conv_kernel_dim_1,
            cfg.conv_stride_1,
            cfg.conv_padding_1,
            cfg.conv_dilation,
        )
        out_2 = conv_out_dim(
            out_1,
            cfg.pool_kernel_dim_1,
            cfg.pool_stride_1,
            cfg.pool_padding_1,
            cfg.pool_dilation,
        )
        out_3 = conv_out_dim(
            out_2,
            cfg.conv_kernel_dim_2,
            cfg.conv_stride_2,
            cfg.conv_padding_2,
            cfg.conv_dilation,
        )
        out_4 = conv_out_dim(
            out_3,
            cfg.pool_kernel_dim_2,
            cfg.pool_stride_2,
            cfg.pool_padding_2,
            cfg.pool_dilation,
        )
        self.cnn_out = cfg.channels_out * out_4
        self.conv1 = nn.Conv1d(
            cfg.channels_in,
            cfg.channels_med_1,
            cfg.conv_kernel_dim_1,
            stride=cfg.conv_stride_1,
            padding=cfg.conv_padding_1,
            dilation=cfg.conv_dilation,
        )
        self.norm1 = nn.BatchNorm1d(cfg.channels_med_1)
        self.pool1 = nn.MaxPool1d(
            cfg.pool_kernel_dim_1,
            stride=cfg.pool_stride_1,
            padding=cfg.pool_padding_1,
            dilation=cfg.pool_dilation,
        )
        self.pool2 = nn.MaxPool1d(
            cfg.pool_kernel_dim_2,
            stride=cfg.pool_stride_2,
            padding=cfg.pool_padding_2,
            dilation=cfg.pool_dilation,
        )
        self.conv2 = nn.Conv1d(
            cfg.channels_med_1,
            cfg.channels_out,
            cfg.conv_kernel_dim_2,
            stride=cfg.conv_stride_2,
            padding=cfg.conv_padding_2,
            dilation=cfg.conv_dilation,
        )
        self.norm2 = nn.BatchNorm1d(cfg.channels_out)
        self.fc1 = nn.Linear(cfg.channels_out * out_4, cfg.fc_dim_1)
        self.fc2 = nn.Linear(cfg.fc_dim_1, cfg.emb_dim)
        self.norm3 = nn.BatchNorm1d(cfg.fc_dim_1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = x.view(-1, self.cnn_out)
        x = F.relu(self.norm3(self.fc1(x)))
        return torch.tanh(self.fc2(x))
