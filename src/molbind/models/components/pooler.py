import torch.nn as nn
from typing import Hint


class EmbedPooler(nn.Module):
    def __init__(self, pooling_function: Hint["max", "min", "last", "first"] = "max"):
        super(EmbedPooler, self).__init__()
    
    def forward(self, x):
        if self.pooling_function == "max":
            return x.max(dim=1)
        elif self.pooling_function == "min":
            return x.min(dim=1)
        elif self.pooling_function == "last":
            return x[:,-1]
        elif self.pooling_function == "first":
            return x[:,0]
        else:
            raise ValueError("Pooling function not implemented.")