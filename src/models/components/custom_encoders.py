import torch.nn as nn
from typing import Dict, Literal, Hint
from molbind.models.components.head import ProjectionHead


class BaseModalityEncoder(nn.Module):
    def __init__(self, 
                projection_head_type : Hint["linear", "non-linear"] = "non-linear",
                pretrained=True,
                **kwargs):
       self.pretrained = pretrained
        
        # some from HuggingFace, others from scratch
        # restore weights = True/False
        # freeze weights = True/False
        # projection head type = linear/non-linear
        self.encoder = self.build_encoder() 

    
    def build_encoder(self):
        pass
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.projection_head(x)
        # pooling
        return x