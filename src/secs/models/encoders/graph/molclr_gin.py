"""MolCLR's GIN over molecular graphs, wired up as the `graph` modality encoder.

The atom/bond vocabulary here has to stay in step with `secs.utils.graph`, which
builds the graphs: the extra atom type is the mask token, the extra bond type is
the self-loop.
"""

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops

from secs.models.base import ModalityEncoder
from secs.models.registry import register_encoder

num_atom_type = 119  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5  # including aromatic and self-loop edge
num_bond_direction = 3


class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        feat_dim (int): dimensionality of the pooled graph feature
        drop_ratio (float): dropout rate
        pool (str): graph-level readout over node states ("mean", "max", "add")
        readout (str): "feat" returns the pooled feature (feat_dim), "projected"
            returns MolCLR's contrastive projection of it (feat_dim // 2). The
            projection head is only built when it is asked for, so no parameter
            goes unused (DDP refuses to step over those).
    Output:
        graph representations
    """

    POOLS: ClassVar[dict[str, object]] = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool}

    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool="mean", readout="feat"):
        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for _ in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

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


@register_encoder("graph", "molclr_gin", default=True)
class GraphGINEncoder(ModalityEncoder):
    """MolCLR GIN over a batched molecular graph. forward takes a PyG `Data`/`Batch`."""

    def __init__(self, ckpt_path: str | None = None, freeze_encoder: bool = False, **backbone_kwargs) -> None:
        super().__init__(ckpt_path=ckpt_path, freeze_encoder=freeze_encoder)
        self.encoder = GINet(**backbone_kwargs)
        self.output_dim = self.encoder.out_dim
        self._finalize()

    def forward(self, data: Data) -> Tensor:
        if self.frozen:
            with torch.no_grad():
                return self.encoder(data)
        return self.encoder(data)
