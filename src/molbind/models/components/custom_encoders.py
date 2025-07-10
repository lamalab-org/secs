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
from transformers import BertModel, BertTokenizer

from molbind.models.components.base_encoder import (
    BaseModalityEncoder,
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
        token_ids, attention_mask = x if len(x) == 2 else (x[0], x[1])
        output = self.encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
        )
        return output.pooler_output


class PolymerNameEncoder(BaseModalityEncoder):
    def __init__(self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs) -> None:
        super().__init__("FacebookAI/roberta-base", freeze_encoder, pretrained, **kwargs)

    def _initialize_encoder(self) -> None:
        config = AutoConfig.from_pretrained(self.model_name)
        self.encoder = RobertaForCausalLM.from_pretrained(self.model_name, config=config)
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
