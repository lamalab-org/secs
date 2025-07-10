import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
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

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        token_ids, attention_mask = x
        output = self.encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden_state = output.hidden_states[-1]

        return self._non_pad_token_embed_averaging(last_hidden_state, attention_mask)

    def _non_pad_token_embed_averaging(self, last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        attention_mask = attention_mask.float().unsqueeze(-1)
        sum_ = (last_hidden_state * attention_mask).sum(dim=1)
        norm = attention_mask.squeeze(-1).sum(dim=1).unsqueeze(1)
        return sum_ / norm


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

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

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
        return torch.sparse.mm(adj_t, x, reduce=self.aggr)
