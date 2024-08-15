from __future__ import annotations

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch import Tensor
from torch.nn.functional import l1_loss
from torch_geometric.data import Data

from molbind.models.components.custom_encoders import StructureEncoder


class StructureEncoderModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.model = StructureEncoder(**cfg.model.encoder)
        self.config = cfg
        self.loss = l1_loss

    def forward(self, data: Data) -> Tensor:
        return self.model(z=data.z, pos=data.pos, batch=data.batch)

    def _compute_loss(self, output: Tensor, target: Tensor, prefix) -> Tensor:
        loss = self.loss(output, target.unsqueeze(1))
        self.log(f"{prefix}_loss", loss, batch_size=output.shape[0])
        return loss

    def training_step(self, batch: Data) -> Tensor:
        output = self.model(z=batch.z, pos=batch.pos, batch=batch.batch)
        return self._compute_loss(output, batch.y, "train")

    def validation_step(self, batch: Data) -> Tensor:
        output = self.model(z=batch.z, pos=batch.pos, batch=batch.batch)
        return self._compute_loss(output, batch.y, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.config.model.optimizer.lr)
