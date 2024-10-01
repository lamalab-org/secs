from pathlib import Path
from typing import (
    Callable,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models.dimenet import (
    DimeNet,
    OutputBlock,
    triplets,
)
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import scatter
from transformers import AutoConfig, AutoModel, RobertaForCausalLM

from molbind.models.components.base_encoder import (
    BaseModalityEncoder,
    FingerprintEncoder,
    GraphEncoder,
)
from molbind.utils import rename_keys_with_prefix, select_device


class SmilesEncoder(BaseModalityEncoder):
    def __init__(self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs) -> None:
        super().__init__("ibm/MoLFormer-XL-both-10pct", freeze_encoder, pretrained, **kwargs)

    def _initialize_encoder(self):
        self.encoder = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        if len(x) == 2:
            token_ids, attention_mask = x
        else:
            token_ids, attention_mask = x[0], x[1]
        output = self.encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
        )
        return output.pooler_output


class SelfiesEncoder(BaseModalityEncoder):
    def __init__(self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs) -> None:
        super().__init__("HUBioDataLab/SELFormer", freeze_encoder, pretrained, **kwargs)


class IUPACNameEncoder(BaseModalityEncoder):
    def __init__(self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs) -> None:
        super().__init__("FacebookAI/roberta-base", freeze_encoder, pretrained, **kwargs)

    def _initialize_encoder(self) -> None:
        config = AutoConfig.from_pretrained(self.model_name)
        self.encoder = RobertaForCausalLM.from_pretrained(self.model_name, config=config)
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False


class DescriptionEncoder:
    pass


class CustomFingerprintEncoder(FingerprintEncoder):
    def __init__(
        self,
        input_dims: list[int],
        output_dims: list[int],
        latent_dim: int,
        ckpt_path: str,
    ) -> None:
        super().__init__(input_dims, output_dims, latent_dim)
        # load weights from the pre-trained model
        self.load_state_dict(rename_keys_with_prefix(torch.load(ckpt_path, map_location=select_device())["state_dict"]))

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class CustomGraphEncoder(GraphEncoder):
    def __init__(
        self,
        ckpt_path: str,
        drop_ratio: float = 0,
        num_layer: int = 5,
        feat_dim: int = 512,
        pool: str = "mean",
        emb_dim: int = 300,
    ) -> None:
        super().__init__(
            drop_ratio=drop_ratio,
            num_layer=num_layer,
            feat_dim=feat_dim,
            pool=pool,
            emb_dim=emb_dim,
        )
        # load weights from the pre-trained model
        # check format of the ckpt file
        suffix = Path(ckpt_path).suffix
        if suffix == ".pth":
            logger.info("Loading graph model from a `.pth` file")
            self.load_state_dict(
                rename_keys_with_prefix(torch.load(ckpt_path, map_location=select_device())),
                strict=True,
            )
        elif suffix == ".ckpt":
            logger.info("Loading graph model from `.ckpt` file")
            self.load_state_dict(
                rename_keys_with_prefix(torch.load(ckpt_path, map_location=select_device())["state_dict"]),
                strict=True,
            )
        self.drop_ratio = drop_ratio

    def forward(self, data: Data) -> tuple:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
        # global pooling
        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        return self.out_lin(h)


class OutputBlockMolBind(OutputBlock):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
        output_initializer: str = "zeros",
    ) -> None:
        super().__init__(
            num_radial=num_radial,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            act=act,
            output_initializer=output_initializer,
        )

    def forward(
        self,
        x: Tensor,
        rbf: Tensor,
        i: Tensor,
        num_nodes: int | None = None,
    ) -> Tensor:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce="sum")
        # copy value of x for molbind
        for lin in self.lins:
            x = self.act(lin(x))
        return x


class StructureEncoder(DimeNet):
    """
    Wrapper around DimeNet model from `torch_geometric.nn.models` to allow for
    easy configuration via Hydra.
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: str | Callable = "swish",
        output_initializer: str = "zeros",
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            num_bilinear=num_bilinear,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
            output_initializer=output_initializer,
        )


class CustomStructureEncoder(StructureEncoder):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: str | Callable = "swish",
        output_initializer: str = "zeros",
        ckpt_path: str | None = None,
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            num_bilinear=num_bilinear,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
            output_initializer=output_initializer,
        )

        self.output_blocks = torch.nn.ModuleList(
            [
                OutputBlockMolBind(
                    num_radial=num_radial,
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    num_layers=num_output_layers,
                    act=activation_resolver(act),
                    output_initializer=output_initializer,
                )
                for _ in range(num_blocks + 1)
            ]
        )
        self.load_state_dict(rename_keys_with_prefix(torch.load(ckpt_path, map_location=select_device())["state_dict"]))

    def forward(
        self,
        batch: Data,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            batch (Data): The input data with all the necessary attributes.
        """

        z, pos, batch = batch.z, batch.pos, batch.batch
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edge_index, num_nodes=z.size(0))
        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        pos_ji, pos_ki = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_i]
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks, self.output_blocks[1:]):
            x = interaction_block(
                x=x,
                rbf=rbf,
                sbf=sbf,
                idx_kj=idx_kj,
                idx_ji=idx_ji,
            )
            P += output_block(x=x, rbf=rbf, i=i, num_nodes=pos.size(0))
        return scatter(P, batch, dim=0, reduce="sum")


class ImageEncoder(nn.Module):
    def __init__(self, ckpt_path: str) -> None:
        super().__init__()
        self.cfg = [
            [128, 7, 3, 4],
            [256, 5, 1, 1],
            [384, 5, 1, 1],
            "M",
            [384, 3, 1, 1],
            [384, 3, 1, 1],
            "M",
            [512, 3, 1, 1],
            [512, 3, 1, 1],
            [512, 3, 1, 1],
            "M",
        ]
        self.features = self.make_layers()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)

    def make_layers(self, batch_norm: bool = False) -> nn.Sequential:
        """
        :param batch_norm: boolean of batch normalization should be used in-between conv2d and relu activation.
                        Defaults to False
        :return: torch.nn.Sequential module as feature-extractor
        """
        layers: list[nn.Module] = []

        in_channels = 1
        for v in self.cfg:
            if v == "A":
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                if v == "M":
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    units, kern_size, stride, padding = v
                    conv2d = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=units,
                        kernel_size=kern_size,
                        stride=stride,
                        padding=padding,
                    )
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(units), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = units
        return nn.Sequential(*layers)

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        x = self.features(x)
        return self.flatten(self.pool(x))


class cNmrEncoder(FingerprintEncoder):
    def __init__(
        self,
        input_dims: list[int],
        output_dims: list[int],
        latent_dim: int,
        ckpt_path: str | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__(input_dims, output_dims, latent_dim)
        # load weights from the pre-trained model
        if ckpt_path is not None:
            self.load_state_dict(rename_keys_with_prefix(torch.load(ckpt_path, map_location=select_device())["state_dict"]))
            logger.info("Loaded weights from pre-trained model for C-NMR")
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class IrEncoder(FingerprintEncoder):
    def __init__(
        self,
        input_dims: list[int],
        output_dims: list[int],
        latent_dim: int,
        ckpt_path: str | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__(input_dims, output_dims, latent_dim)
        # load weights from the pre-trained model
        if ckpt_path is not None:
            self.load_state_dict(rename_keys_with_prefix(torch.load(ckpt_path, map_location=select_device())["state_dict"]))
            logger.info("Loaded weights from pre-trained model for IR")
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class MassSpecPositiveEncoder(FingerprintEncoder):
    def __init__(
        self,
        input_dims: list[int],
        output_dims: list[int],
        latent_dim: int,
        ckpt_path: str | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__(input_dims, output_dims, latent_dim)
        # load weights from the pre-trained model
        if ckpt_path is not None:
            self.load_state_dict(rename_keys_with_prefix(torch.load(ckpt_path, map_location=select_device())["state_dict"]))
            logger.info("Loaded weights from pre-trained model for Mass Spec Positive")
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class MassSpecNegativeEncoder(FingerprintEncoder):
    def __init__(
        self,
        input_dims: list[int],
        output_dims: list[int],
        latent_dim: int,
        ckpt_path: str | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__(input_dims, output_dims, latent_dim)
        # load weights from the pre-trained model
        if ckpt_path is not None:
            self.load_state_dict(rename_keys_with_prefix(torch.load(ckpt_path, map_location=select_device())["state_dict"]))
            logger.info("Loaded weights from pre-trained model for Mass Spec Negative")
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class hNmrEncoder(FingerprintEncoder):
    def __init__(
        self,
        input_dims: list[int],
        output_dims: list[int],
        latent_dim: int,
        ckpt_path: str | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__(input_dims, output_dims, latent_dim)
        # load weights from the pre-trained model
        if ckpt_path is not None:
            self.load_state_dict(rename_keys_with_prefix(torch.load(ckpt_path, map_location=select_device())["state_dict"]))
            logger.info("Loaded weights from pre-trained model for H-NMR")

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


def conv_out_dim(length_in: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (length_in + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


class hNmrCNNEncoder(nn.Module):
    def __init__(self, ckpt_path: str | None = None, freeze_encoder: bool = False) -> None:
        super().__init__()
        cfg = DictConfig(
            {
                "length_in": 10000,
                "channels_in": 1,
                "channels_med_1": 2,
                "channels_out": 4,
                "latent_dim": 512,
                "conv_layer": {"kernel": 200, "stride": 1, "padding": 100, "pool_factor": 50},
                "pool_layer": {"kernel": 50, "stride": 50, "padding": 0, "pool_factor": 50},
                "fc_dim_1": 512,
                "lr": 0.001,
                "conv_kernel_dim_1": 200,
                "conv_stride_1": 1,
                "conv_padding_1": 100,
                "conv_dilation": 1,
                "pool_kernel_dim_1": 50,
                "pool_stride_1": 50,
                "pool_padding_1": 0,
                "pool_dilation": 1,
                "conv_kernel_dim_2": 200,
                "conv_stride_2": 1,
                "conv_padding_2": 100,
                "pool_kernel_dim_2": 50,
                "pool_stride_2": 50,
                "pool_padding_2": 0,
                "emb_dim": 512,
            }
        )

        out_1 = conv_out_dim(
            cfg.length_in,
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

        if ckpt_path is not None:
            self.load_state_dict(rename_keys_with_prefix(torch.load(ckpt_path, map_location=select_device())["state_dict"]))
            logger.info("Loaded weights from pre-trained model for H-NMR CNN")
        if freeze_encoder:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = x.view(-1, self.cnn_out)
        x = F.relu(self.norm3(self.fc1(x)))
        return torch.tanh(self.fc2(x))
