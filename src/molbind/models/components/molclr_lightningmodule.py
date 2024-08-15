from __future__ import annotations

import torch
import torch.nn.functional as F
from info_nce import InfoNCE
from lightning import LightningModule
from omegaconf import DictConfig
from torch import Tensor

from molbind.models.components.base_encoder import GraphEncoder


class GCNModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_config = cfg.model
        self.loss_config = cfg.loss
        self.batch_size = cfg.data.batch_size
        self.model = GraphEncoder(**self.model_config)
        self.criterion = InfoNCE(
            temperature=self.loss_config.temperature, negative_mode="unpaired"
        )
        # log hyperparameters
        self.log(name="batch_size", value=self.batch_size, batch_size=self.batch_size)
        self.log("learning_rate", self.cfg.optimizer.lr)

    def forward(self, batch):
        xis, xjs = batch
        ris, zis = self.model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = self.model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        return zis, zjs

    def _info_nce(self, batch, prefix: str = "train") -> Tensor:
        xis, xjs = batch
        _, zis = self.model(xis)  # [N,C]

        # get the representations and the projections
        _, zjs = self.model(xjs)  # [N,C]
        loss = self.criterion(zis, zjs)
        self.log(f"{prefix}_loss", loss)
        return loss

    def training_step(self, batch) -> Tensor:
        return self._info_nce(batch, prefix="train")

    def validation_step(self, batch) -> Tensor:
        return self._info_nce(batch, prefix="val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
