import torch.nn as nn
from typing import Dict, Literal
from molbind.models.components.head import ProjectionHead
from molbind.models.components.base_encoder import BaseModalityEncoder
from transformers import AutoModelForCausalLM


class SmilesEncoder(BaseModalityEncoder):
    def __init__(self,
                projection_head_type=Literal["linear", "non-linear"],
                **kwargs):

        super().__init__(projection_head_type, pretrained=True, **kwargs)

    def build_encoder(self):
        return AutoModelForCausalLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")


class SelfiesEncoder(BaseModalityEncoder):
    def __init__(self,
                freeze_encoder: bool = False,
                pretrained=True,
                **kwargs):
        super().__init__(projection_head_type, pretrained, **kwargs)

    def build_encoder(self):
        if self.pretrained:
            return AutoModelForCausalLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        else:
            return AutoModelForCausalLM.from_config(self.config)

    def build_projection_head(self, projection_head_type, **kwargs):
        # TODO: implement
        raise NotImplementedError
