import torch.nn as nn
from typing import Dict, Literal
from molbind.models.components.head import ProjectionHead
from molbind.models.components.base_encoder import BaseModalityEncoder
from transformers import AutoModelForCausalLM


class SmilesEncoder(BaseModalityEncoder):
    def __init__(self, 
                projection_head_type=Literal["linear", "non-linear"],
                pretrained=True,
                **kwargs):
        super().__init__(projection_head_type, pretrained, **kwargs)
        self.encoder = self.build_encoder()
    
    def build_encoder(self):
        return AutoModelForCausalLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    
    def forward(self, x):
        x = self.encoder(x)
        return x
    
    
class SelfiesEncoder(BaseModalityEncoder):
    def __init__(self, 
                freeze_encoder: bool = False,
                pretrained=True,
                **kwargs):
        super().__init__(projection_head_type, pretrained, **kwargs)
        self.encoder = self.build_encoder()
    
    def build_encoder(self):
        if self.pretrained:
            return AutoModelForCausalLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        else:
            return AutoModelForCausalLM.from_config(self.config)
    
    def forward(self, x):
        x = self.encoder(x)
        # pool embeddings 
        
        x = self.projection_head(x)
        # pooling
        return x