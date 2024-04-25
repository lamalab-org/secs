import torch  # noqa: I002
import torch.nn.functional as F
from lightning import LightningModule
from omegaconf import DictConfig
from torch import Tensor

from molbind.models.components.base_encoder import GCN
from molbind.models.components.molclr_loss import NTXentLoss


class GCNModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_config = cfg.model
        self.loss_config = cfg.loss
        self.model = GCN(**self.model_config)
        self.criterion = NTXentLoss(**self.loss_config)

    def forward(self, batch):
        _, xis_xjs = batch
        xis, xjs = xis_xjs[0], xis_xjs[1]
        _, zis = self.model(xis)
        _, zjs = self.model(xjs)
        return zis, zjs

    def _nt_xent_loss(self, batch):
        _, xis_xjs = batch
        xis, xjs = xis_xjs[0], xis_xjs[1]

        _, zis = self.model(xis)  # [N,C]

        # get the representations and the projections
        _, zjs = self.model(xjs)  # [N,C]
        return self.criterion(zis, zjs)

    def training_step(self, batch) -> Tensor:
        return self._nt_xent_loss(batch)

    def validation_step(self, batch) -> Tensor:
        return self._nt_xent_loss(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
