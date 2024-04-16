from typing import List

import torch
from torch import Tensor

from molbind.models.components.base_encoder import (
    BaseModalityEncoder,
    FingerprintEncoder,
)
from molbind.utils.utils import remove_keys_with_prefix


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


class CustomFingerprintEncoder(FingerprintEncoder):
    def __init__(
        self,
        input_dim: List[int],
        output_dim: List[int],
        latent_dim: int,
        ckpt_path: str,
    ):
        super().__init__(input_dim, output_dim, latent_dim)
        # load weights from the pre-trained model
        self.load_state_dict(
            remove_keys_with_prefix(torch.load(ckpt_path)["state_dict"])
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)
