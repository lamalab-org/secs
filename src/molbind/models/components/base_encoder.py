from typing import Dict, Literal
from transformers import AutoModelForCausalLM
import torch.nn as nn
    
    
class BaseModalityEncoder(nn.Module):
    def __init__(self, 
                freeze_encoder: bool = False,
                pretrained=True,
                *args,
                **kwargs):
        super(BaseModalityEncoder, self).__init__()
        self.pretrained = pretrained
        self.freeze_encoder = freeze_encoder

    def build_encoder(self):
        pass
    
    def forward(self, x):
        return self.encoder(x)