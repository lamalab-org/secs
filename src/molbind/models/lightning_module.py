from info_nce import InfoNCE
from pytorch_lightning import LightningModule
from molbind.models.model import MolBind
import torch


class MolBindModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = MolBind(cfg=cfg)
        self.config = cfg
        self.loss = InfoNCE(temperature=cfg.loss.temperature, negative_mode="unpaired")

    def forward(self, input):
        return self.model(input)

    def _info_nce_loss(self, z1, z2):
        return self.loss(z1, z2)

    def _multimodal_loss(self, embeddings_dict, prefix):
        modality_pair = [*embeddings_dict.keys()]
        loss = self._info_nce_loss(
            embeddings_dict[modality_pair[0]], embeddings_dict[modality_pair[1]]
        )
        self.log(f"{prefix}_loss", loss)
        return loss

    def training_step(self, input):
        embeddings_dict = self.forward(input)
        return self._multimodal_loss(embeddings_dict, "train")

    def validation_step(self, input):
        embeddings_dict = self.forward(input)
        return self._multimodal_loss(embeddings_dict, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.optimizer.lr,
            weight_decay=self.config.model.optimizer.weight_decay,
        )