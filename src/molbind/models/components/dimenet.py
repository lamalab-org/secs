from typing import Callable, Literal, Optional, Union  # noqa: I002

import torch
from torch import Tensor
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models.dimenet import (
    DimeNet,
    OutputBlock,
    triplets,
)
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter


class OutputBlockMolBind(OutputBlock):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
        output_initializer: str = "zeros",
        model: Literal["dimenet", "molbind"] = "dimenet",
    ) -> None:
        self.model = model
        super().__init__(
            num_radial,
            hidden_channels,
            out_channels,
            num_layers,
            output_initializer,
            act,
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
        embedding = x.clone()
        for lin in self.lins:
            x = self.act(lin(x))
        if self.model == "dimenet":
            return self.lin(x)
        elif self.model == "molbind":
            return x, embedding
        raise ValueError("Model must be 'dimenet' or 'molbind'.")


class DimeNetMolBind(DimeNet):
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
        model: Literal["dimenet", "molbind"] = "dimenet",
        ckpt_path: Optional[str] = None,
    ):
        self.model = model
        super().__init__(
            hidden_channels,
            out_channels,
            num_blocks,
            num_bilinear,
            num_spherical,
            num_radial,
            cutoff,
            max_num_neighbors,
            envelope_exponent,
            num_before_skip,
            num_after_skip,
            num_output_layers,
            act,
            output_initializer,
        )

        if num_spherical < 2:
            raise ValueError("'num_spherical' should be greater than 1")
        del self.output_blocks
        self.output_blocks = torch.nn.ModuleList(
            [
                OutputBlockMolBind(
                    num_radial,
                    hidden_channels,
                    out_channels,
                    num_output_layers,
                    act,
                    output_initializer,
                    model=self.model,
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
        z: Tensor,
        pos: Tensor,
        batch: OptTensor = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
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
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            if self.model == "dimenet":
                P = P + output_block(
                    x=x, rbf=rbf, i=i, num_nodes=pos.size(0), model=self.model
                )
            elif self.model == "molbind":
                output_block_ = output_block(
                    x=x, rbf=rbf, i=i, num_nodes=pos.size(0), model=self.model
                )
                P = P + output_block_[0]
        if self.model == "molbind":
            return output_block_[1]
        if batch is None:
            return P.sum(dim=0)
        else:
            return scatter(P, batch, dim=0, reduce="sum")
