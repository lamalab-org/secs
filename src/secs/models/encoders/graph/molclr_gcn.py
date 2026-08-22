"""MolCLR's GCN over molecular graphs, the sibling backbone of `molclr_gin`.

Same atom/bond vocabulary, same readout contract and the same MolCLR checkpoint
layout as the GIN; only the message passing differs. The vocabulary has to stay
in step with `secs.utils.graph`, which builds the graphs: the extra atom type is
the mask token, the extra bond type is the self-loop.
"""

import math
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops, degree

from secs.models.base import ModalityEncoder
from secs.models.encoders.graph.molclr_gin import (
    num_atom_type,
    num_bond_direction,
    num_bond_type,
    num_chirality_tag,
)
from secs.models.registry import register_encoder


def gcn_norm(edge_index, num_nodes: int):
    """Symmetric degree normalisation D^-1/2 A D^-1/2 for an edge list."""
    row, col = edge_index[0], edge_index[1]
    deg = degree(col, num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    return deg_inv_sqrt[row] * deg_inv_sqrt[col]


class GCNConv(MessagePassing):
    """MolCLR's GCN layer: a shared linear map plus a scalar bond bias per edge."""

    def __init__(self, emb_dim, aggr="add", normalize=False):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.normalize = normalize

        self.weight = Parameter(torch.empty(emb_dim, emb_dim))
        self.bias = Parameter(torch.empty(emb_dim))
        self.reset_parameters()

        # a single scalar per bond, as in MolCLR: the edge only shifts the message
        self.edge_embedding1 = nn.Embedding(num_bond_type, 1)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, 1)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        # MolCLR computes the D^-1/2 A D^-1/2 normalisation and then drops it on
        # the floor, so its layer is a plain neighbour sum. `normalize` defaults
        # to that quirk: the released weights and BatchNorm running stats were
        # trained through the unnormalised path, and switching it on under them
        # rescales the messages by ~4-7x. Turn it on when training from scratch.
        norm = gcn_norm(edge_index, x.size(0)) if self.normalize else None
        x = x @ self.weight

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm) + self.bias

    def message(self, x_j, edge_attr, norm):
        message = x_j + edge_attr
        return message if norm is None else norm.view(-1, 1) * message


class GCNet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        feat_dim (int): dimensionality of the pooled graph feature
        drop_ratio (float): dropout rate
        pool (str): graph-level readout over node states ("mean", "max", "add")
        normalize (bool): apply the GCN degree normalisation. Off by default,
            matching MolCLR, whose released checkpoints were trained without it.
        readout (str): "feat" returns the pooled feature (feat_dim), "projected"
            returns MolCLR's contrastive projection of it (feat_dim // 2). The
            projection head is only built when it is asked for, so no parameter
            goes unused (DDP refuses to step over those).
    Output:
        graph representations
    """

    POOLS: ClassVar[dict[str, object]] = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool}

    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool="mean", readout="feat", normalize=False):
        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of convolutions
        self.gnns = nn.ModuleList()
        for _ in range(num_layer):
            self.gnns.append(GCNConv(emb_dim, aggr="add", normalize=normalize))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool not in self.POOLS:
            raise ValueError(f"Unknown pool '{pool}'. Available: {sorted(self.POOLS)}")
        self.pool = self.POOLS[pool]

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        if readout not in {"feat", "projected"}:
            raise ValueError(f"Unknown readout '{readout}'. Available: ['feat', 'projected']")
        self.readout = readout
        self.out_dim = feat_dim if readout == "feat" else feat_dim // 2
        self.out_lin = (
            nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim, self.feat_dim // 2),
            )
            if readout == "projected"
            else None
        )

    def forward(self, data):
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

        # A dataset handed over one graph at a time (no collater) has no `batch`;
        # everything then belongs to a single graph.
        batch = data.batch if data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.pool(h, batch)
        h = self.feat_lin(h)

        return h if self.out_lin is None else self.out_lin(h)


@register_encoder("graph", "molclr_gcn")
class GraphGCNEncoder(ModalityEncoder):
    """MolCLR GCN over a batched molecular graph. forward takes a PyG `Data`/`Batch`."""

    def __init__(self, ckpt_path: str | None = None, freeze_encoder: bool = False, **backbone_kwargs) -> None:
        super().__init__(ckpt_path=ckpt_path, freeze_encoder=freeze_encoder)
        self.encoder = GCNet(**backbone_kwargs)
        self.output_dim = self.encoder.out_dim
        self._finalize()

    def forward(self, data: Data) -> Tensor:
        if self.frozen:
            with torch.no_grad():
                return self.encoder(data)
        return self.encoder(data)
