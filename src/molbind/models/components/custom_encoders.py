from typing import (  # noqa: I002, UP035
    Callable,
    List,
    Optional,
    Union,
)

import torch
import torch.nn.functional as F
from loguru import logger
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
from transformers import AutoConfig, RobertaForCausalLM

from molbind.models.components.base_encoder import (
    BaseModalityEncoder,
    FingerprintEncoder,
    GraphEncoder,
)
from molbind.utils import rename_keys_with_prefix, select_device


class SmilesEncoder(BaseModalityEncoder):
    def __init__(
        self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs
    ) -> None:
        super().__init__(
            "seyonec/ChemBERTa-zinc-base-v1", freeze_encoder, pretrained, **kwargs
        )


class SelfiesEncoder(BaseModalityEncoder):
    def __init__(
        self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs
    ) -> None:
        super().__init__("HUBioDataLab/SELFormer", freeze_encoder, pretrained, **kwargs)


class IUPACNameEncoder(BaseModalityEncoder):
    def __init__(
        self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs
    ) -> None:
        super().__init__(
            "FacebookAI/roberta-base", freeze_encoder, pretrained, **kwargs
        )

    def _initialize_encoder(self) -> None:
        config = AutoConfig.from_pretrained(self.model_name)
        self.encoder = RobertaForCausalLM.from_pretrained(
            self.model_name, config=config
        )
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False


class DescriptionEncoder:
    pass


class IREncoder(BaseModalityEncoder):
    pass


class NMREncoder(BaseModalityEncoder):
    pass


class CustomFingerprintEncoder(FingerprintEncoder):
    def __init__(
        self,
        input_dims: List[int],  # noqa: UP006
        output_dims: List[int],  # noqa: UP006
        latent_dim: int,
        ckpt_path: str,
    ) -> None:
        super().__init__(input_dims, output_dims, latent_dim)
        # load weights from the pre-trained model
        self.load_state_dict(
            rename_keys_with_prefix(
                torch.load(ckpt_path, map_location=select_device())["state_dict"]
            )
        )

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
        self.load_state_dict(
            rename_keys_with_prefix(
                torch.load(ckpt_path, map_location=select_device())["state_dict"]
            )
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
        num_nodes: Optional[int] = None,
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
        act: Union[str, Callable] = "swish",
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
        act: Union[str, Callable] = "swish",
        output_initializer: str = "zeros",
        ckpt_path: Optional[str] = None,
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
        self.load_state_dict(
            rename_keys_with_prefix(
                torch.load(ckpt_path, map_location=select_device())["state_dict"]
            )
        )

    def forward(
        self,
        batch: Data,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            batch (Data): The input data with all the necessary attributes.
        """

        z, pos, batch = batch.z, batch.pos, batch.batch
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors
        )

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0)
        )
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
        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:]
        ):
            x = interaction_block(
                x=x,
                rbf=rbf,
                sbf=sbf,
                idx_kj=idx_kj,
                idx_ji=idx_ji,
            )
            output_block_ = output_block(x=x, rbf=rbf, i=i, num_nodes=pos.size(0))
            P += output_block_
        return scatter(P, batch, dim=0, reduce="sum")