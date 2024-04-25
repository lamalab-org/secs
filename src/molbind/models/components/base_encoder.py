import math  # noqa: I002
from typing import List, Tuple  # noqa: UP035

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from transformers import AutoModelForCausalLM

from molbind.models.components.head import ProjectionHead


def xavier_init(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        if len(param.shape) > 1:
            nn.init.xavier_uniform_(param)
    return model


class BaseModalityEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze_encoder: bool = False,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        self.pretrained = pretrained
        self.encoder = None
        self._initialize_encoder()

    def _initialize_encoder(self):
        if self.pretrained:
            self.encoder = AutoModelForCausalLM.from_pretrained(self.model_name)
            if self.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        else:
            self.encoder = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.encoder = xavier_init(self.encoder)

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tensor:  # noqa: UP006
        token_ids, attention_mask = x
        output = self.encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden_state = output.hidden_states[-1]

        return self._non_pad_token_embed_averaging(last_hidden_state, attention_mask)

    def _non_pad_token_embed_averaging(
        self, last_hidden_state: Tensor, attention_mask: Tensor
    ) -> Tensor:
        attention_mask = attention_mask.float().unsqueeze(-1)
        sum_ = (last_hidden_state * attention_mask).sum(dim=1)
        norm = attention_mask.squeeze(-1).sum(dim=1).unsqueeze(1)
        return sum_ / norm


class FingerprintEncoder(nn.Module):
    def __init__(
        self,
        input_dims: List[int],  # noqa: UP006
        output_dims: List[int],  # noqa: UP006
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = ProjectionHead(dims=input_dims, activation="leakyrelu")
        # Output layers for mu and log_var
        self.fc_mu = nn.Linear(input_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(input_dims[-1], latent_dim)
        # decoder
        self.decoder = ProjectionHead(dims=output_dims, activation="leakyrelu")

    def encode(self, x: Tensor):
        return self.encoder(x)

    def decode(self, x: Tensor):
        return self.decoder(x)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # noqa: UP006
        latent_state = self.encode(x)
        mu = self.fc_mu(latent_state)
        log_var = self.fc_log_var(latent_state)
        output = self.decode(latent_state)
        return mu, log_var, output



def gcn_norm(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super().__init__()
        self.emb_dim = emb_dim
        self.aggr = aggr

        num_bond_type = 5  # including aromatic and self-loop edge
        num_bond_direction = 3
        self.weight = Parameter(torch.Tensor(emb_dim, emb_dim))
        self.bias = Parameter(torch.Tensor(emb_dim))
        self.reset_parameters()

        self.edge_embedding1 = nn.Embedding(num_bond_type, 1)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, 1)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def reset_parameters(self):
        # glorot(self.weight)
        # zeros(self.bias)
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        edge_index, __ = gcn_norm(edge_index)

        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_attr):
        # return x_j if edge_attr is None else edge_attr.view(-1, 1) * x_j
        return x_j if edge_attr is None else edge_attr + x_j

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)


class GCN(nn.Module):
    def __init__(
        self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool="mean"
    ):
        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        num_atom_type = 119  # including the extra mask tokens
        num_chirality_tag = 3
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for _ in range(num_layer):
            self.gnns.append(GCNConv(emb_dim, aggr="add"))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == "mean":
            self.pool = global_mean_pool
        elif pool == "add":
            self.pool = global_add_pool
        elif pool == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Not defined pooling!")

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            # nn.Softplus(),
            nn.Linear(self.feat_dim, self.feat_dim // 2),
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

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)

        return h, out
