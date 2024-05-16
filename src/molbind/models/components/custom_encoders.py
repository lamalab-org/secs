from typing import List  # noqa: UP035, I002

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import OPTForCausalLM

from molbind.models.components.base_encoder import (
    BaseModalityEncoder,
    FingerprintEncoder,
    GraphEncoder,
)
from molbind.utils import rename_keys_with_prefix, select_device


class SmilesEncoder(BaseModalityEncoder):
    def __init__(
        self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs
    ) -> None:
        super().__init__(
            "seyonec/ChemBERTa-zinc-base-v1", freeze_encoder, pretrained, **kwargs
        )


class SelfiesEncoder(BaseModalityEncoder):
    def __init__(
        self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs
    ) -> None:
        super().__init__("HUBioDataLab/SELFormer", freeze_encoder, pretrained, **kwargs)


class IUPACNameEncoder(BaseModalityEncoder):
    pass


class IREncoder(BaseModalityEncoder):
    pass


class NMREncoder(BaseModalityEncoder):
    def __init__(self, freeze_encoder: bool = False, pretrained: bool = True):
        super().__init__("facebook/galactica-125m", freeze_encoder, pretrained)

    def _initialize_encoder(self) -> None:
        self.encoder = OPTForCausalLM.from_pretrained(self.model_name)
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    # def _non_pad_token_embed_averaging(
    #     self, last_hidden_state: Tensor, attention_mask: Tensor
    # ) -> Tensor:
    #     attention_mask = attention_mask.float().unsqueeze(-1)
    #     sum_ = (last_hidden_state * attention_mask).sum(dim=1)
    #     norm = attention_mask.squeeze(-1).sum(dim=1).unsqueeze(1)
    #     return sum_ / norm


class CustomFingerprintEncoder(FingerprintEncoder):
    def __init__(
        self,
        input_dims: List[int],  # noqa: UP006
        output_dims: List[int],  # noqa: UP006
        latent_dim: int,
        ckpt_path: str,
    ) -> None:
        super().__init__(input_dims, output_dims, latent_dim)
        # load weights from the pre-trained model
        self.load_state_dict(
            rename_keys_with_prefix(
                torch.load(ckpt_path, map_location=select_device())["state_dict"]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class CustomGraphEncoder(GraphEncoder):
    def __init__(
        self,
        ckpt_path: str,
        drop_ratio: float = 0,
        num_layer: int = 5,
        feat_dim: int = 512,
        pool: str = "mean",
        emb_dim: int = 300,
    ) -> None:
        super().__init__(
            drop_ratio=drop_ratio,
            num_layer=num_layer,
            feat_dim=feat_dim,
            pool=pool,
            emb_dim=emb_dim,
        )
        # load weights from the pre-trained model
        self.load_state_dict(
            rename_keys_with_prefix(
                torch.load(ckpt_path, map_location=select_device())["state_dict"]
            )
        )
        self.drop_ratio = drop_ratio

    def forward(self, data: tuple) -> Tensor:
        xis, xjs = data
        ris, zis = self.forward_single(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = self.forward_single(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        # return averaged projection features
        return (zis + zjs) / 2

    def forward_single(self, data) -> tuple:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
        # global pooling
        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)
        return h, out
