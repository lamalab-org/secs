from typing import Dict, Tuple
import torch
from transformers import AutoModelForCausalLM
from molbind.models.components.base_encoder import BaseModalityEncoder
from molbind.utils.utils import reinitialize_weights


class SmilesEncoder(BaseModalityEncoder):
    def __init__(self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs):
        super().__init__(
            "seyonec/ChemBERTa-zinc-base-v1", freeze_encoder, pretrained, **kwargs
        )


class SelfiesEncoder(BaseModalityEncoder):
    def __init__(self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs):
        super().__init__("HUBioDataLab/SELFormer", freeze_encoder, pretrained, **kwargs)


class IUPACNameEncoder(BaseModalityEncoder):
    pass


class IREncoder(BaseModalityEncoder):
    pass


class GraphEncoder(BaseModalityEncoder):
    pass


class NMREncoder(BaseModalityEncoder):
    pass
