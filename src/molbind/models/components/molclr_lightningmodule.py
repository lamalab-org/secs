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
        self.batch_size = cfg.data.batch_size
        self.model = GCN(**self.model_config)
        self.criterion = NTXentLoss(**self.loss_config)
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

    def _nt_xent_loss(self, batch, prefix: str = "train") -> Tensor:
        xis, xjs = batch
        _, zis = self.model(xis)  # [N,C]

        # get the representations and the projections
        _, zjs = self.model(xjs)  # [N,C]
        loss = self.criterion(zis, zjs)
        self.log(f"{prefix}_loss", loss)
        return loss

    def training_step(self, batch) -> Tensor:
        return self._nt_xent_loss(batch, prefix="train")

    def validation_step(self, batch) -> Tensor:
        return self._nt_xent_loss(batch, prefix="val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
