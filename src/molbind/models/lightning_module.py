from typing import Dict

import torch
from info_nce import InfoNCE
from pytorch_lightning import LightningModule
from torch import Tensor

from molbind.models.model import MolBind


class MolBindModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = MolBind(cfg=cfg)
        self.config = cfg
        self.loss = InfoNCE(
            temperature=cfg.model.loss.temperature, negative_mode="unpaired"
        )

    def forward(self, batch: Dict):
        return self.model(batch)

    def _info_nce_loss(self, z1: Tensor, z2: Tensor):
        return self.loss(z1, z2)

    def _multimodal_loss(self, embeddings_dict: Dict, prefix: str):
        modality_pair = [*embeddings_dict]
        loss = self._info_nce_loss(
            embeddings_dict[modality_pair[0]], embeddings_dict[modality_pair[1]]
        )
        self.log(f"{prefix}_loss", loss)
        return loss

    def training_step(self, batch: Dict):
        embeddings_dict = self.forward(batch)
        return self._multimodal_loss(embeddings_dict, "train")

    def validation_step(self, batch: Dict):
        embeddings_dict = self.forward(batch)
        return self._multimodal_loss(embeddings_dict, "valid")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.optimizer.lr,
            weight_decay=self.config.model.optimizer.weight_decay,
        )
