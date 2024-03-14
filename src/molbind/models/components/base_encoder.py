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

    
    def build_encoder(self):
        pass
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.projection_head(x)
        # pooling
        return x