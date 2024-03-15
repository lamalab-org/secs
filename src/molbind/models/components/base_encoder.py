from typing import Dict, Literal
from transformers import AutoModelForCausalLM
import torch.nn as nn


class BaseModalityEncoder(nn.Module):
    def __init__(self,
                projection_head_type : Literal["linear", "non-linear"] = "non-linear",
                pretrained=True,
                **kwargs):
       self.pretrained = pretrained

       self.encoder = self.build_encoder()
       self.pooler = self.build_pooler(**kwargs)
       self.projection_head = self.build_projection_head(
        projection_head_type, **kwargs)

    def build_encoder(self):
        raise NotImplementedError

    def build_pooler(self, **kwargs):
        # raise NotImplementedError
        return nn.Identity()

    def build_projection_head(self, projection_head_type, **kwargs):
        # raise NotImplementedError
        return nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        x = self.pooler(x)
        x = self.projection_head(x)

        return x