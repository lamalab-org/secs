from molbind.models.components.base_encoder import BaseModalityEncoder
from typing import Dict, Literal, Hint
from transformers import AutoModelForCausalLM
import torch.nn as nn


class SmilesEncoder(BaseModalityEncoder):
    def __init__(self, 
                projection_head_type=Hint["linear", "non-linear"],
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
                projection_head_type=Hint["linear", "non-linear"],
                pretrained=True,
                **kwargs):
        super().__init__(projection_head_type, pretrained, **kwargs)
        self.encoder = self.build_encoder()
        self.projection_head = self.build_projection_head(config.projection_parameters)
        self.pooling_function = "max/min/last/first"
    
    def build_encoder(self):
        return AutoModelForCausalLM.from_pretrained("HUBioDataLab/SELFormer")
    
    def forward(self, x):
        x = self.encoder(x)
        # pool embeddings 
        
        x = self.projection_head(x)
        # pooling
        return x