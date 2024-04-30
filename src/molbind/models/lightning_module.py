from typing import Dict  # noqa: UP035, I002

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
        self.batch_size = cfg.data.batch_size
        self.central_modality = cfg.data.central_modality

    def forward(self, batch: Dict) -> Dict:  # noqa: UP006
        try:
            forward_pass = self.model(batch)
        except Exception:
            dict_input_data = self._treat_graph_batch(batch)
            forward_pass = self.model(dict_input_data)
        return forward_pass

    def _info_nce_loss(self, z1: Tensor, z2: Tensor) -> float:
        return self.loss(z1, z2)

    def _multimodal_loss(self, embeddings_dict: Dict, prefix: str) -> float:  # noqa: UP006
        modality_pair = [*embeddings_dict]
        loss = self._info_nce_loss(
            embeddings_dict[modality_pair[0]], embeddings_dict[modality_pair[1]]
        )
        self.log(f"{prefix}_loss", loss)
        return loss

    def training_step(self, batch: Dict):  # noqa: UP006
        embeddings_dict = self.forward(batch)
        return self._multimodal_loss(embeddings_dict, "train")

    def validation_step(self, batch: Dict) -> Tensor:  # noqa: UP006
        embeddings_dict = self.forward(batch)
        return self._multimodal_loss(embeddings_dict, "valid")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.optimizer.lr,
            weight_decay=self.config.model.optimizer.weight_decay,
        )

    def _treat_graph_batch(self, batch):
        # this adjusts the shape of the central modality data to be compatible with the model
        if not hasattr(batch[0][0], "input_ids"):
            central_modality_data = batch[0][0].central_modality_data.reshape(
                self.batch_size, -1
            )
        else:
            central_modality_data = (
                batch[0][0].input_ids.reshape(self.batch_size, -1),
                batch[0][0].attention_mask.reshape(self.batch_size, -1)
            )
        return {
            self.central_modality: central_modality_data,
            "graph": batch[0],
        }
