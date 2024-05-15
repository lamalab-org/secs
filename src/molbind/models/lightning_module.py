from typing import Dict, List  # noqa: UP035, I002

import torch
from info_nce import InfoNCE
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn.functional import cosine_similarity
from torchmetrics.retrieval import (
    RetrievalAUROC,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalPrecision,
    RetrievalRecall,
)

# from molbind.metrics.retrieval import compute_top_k_retrieval
from molbind.models import MolBind
from molbind.utils import select_device


class MolBindModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.model = MolBind(cfg=cfg)
        self.config = cfg
        self.loss = InfoNCE(
            temperature=cfg.model.loss.temperature, negative_mode="unpaired"
        )
        self.batch_size = cfg.data.batch_size
        self.central_modality = cfg.data.central_modality
        self.store_embeddings = []

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
        self.log(f"{prefix}_loss", loss, batch_size=self.batch_size)
        # compute cosine similarity between the embeddings of the central modality
        # and the other modality
        similarity = torch.nn.functional.cosine_similarity(
            embeddings_dict[modality_pair[0]], embeddings_dict[modality_pair[1]], dim=1
        )
        self.log(
            f"{prefix}_{modality_pair[0]}_{modality_pair[1]}_similarity",
            similarity.mean(),
            batch_size=self.batch_size,
        )
        # compute retrieval metrics
        k_list = [1, 5, 10]

        self.retrieval_metrics(
            embeddings_dict[modality_pair[0]],
            embeddings_dict[modality_pair[1]],
            *modality_pair,
            k_list,
        )
        return loss

    def training_step(self, batch: Dict):  # noqa: UP006
        embeddings_dict = self.forward(batch)
        return self._multimodal_loss(embeddings_dict, "train")

    def validation_step(self, batch: Dict) -> Tensor:  # noqa: UP006
        embeddings_dict = self.forward(batch)
        return self._multimodal_loss(embeddings_dict, "valid")

    def test_step(self, batch: Dict) -> Tensor:  # noqa: UP006
        embeddings_dict = self.forward(batch)
        self.store_embeddings.append(embeddings_dict)
        return self._multimodal_loss(embeddings_dict, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.optimizer.lr,
            weight_decay=self.config.model.optimizer.weight_decay,
        )

    def retrieval_metrics(
        self,
        embeddings_central_mod: Tensor,
        embeddings_other_mod: Tensor,
        central_modality: str,
        other_modality: str,
        k_list: List[int],  # noqa: UP006
    ) -> None:
        metrics = [
            RetrievalMRR,
            RetrievalMAP,
            RetrievalPrecision,
            RetrievalRecall,
            RetrievalAUROC,
        ]
        metric_names = [metric.__name__ for metric in metrics]
        # compute cosine similarity matrix between embeddings of the central modality and the other modality
        self.cos_sim = cosine_similarity(
            embeddings_central_mod.unsqueeze(1),
            embeddings_other_mod.unsqueeze(0),
            dim=2,
        )
        # preds, target, indexes
        flatten_cos_sim = self.cos_sim.flatten()
        indexes = torch.arange(embeddings_central_mod.shape[0]).repeat(
            embeddings_central_mod.shape[0]
        )
        target = (
            torch.eye(embeddings_central_mod.shape[0], dtype=torch.long)
            .flatten()
            .to(select_device())
        )

        for k_val, metric, metric_name in zip(k_list, metrics, metric_names):
            metric_to_log = metric(top_k=k_val)
            metric_to_log.update(flatten_cos_sim, target, indexes)
            self.log(
                f"{central_modality}_{other_modality}_{metric_name}_top_{k_val}",
                metric_to_log.compute(),
                batch_size=self.batch_size,
            )

    def _treat_graph_batch(self, batch: Dict) -> Dict:  # noqa: UP006
        # this adjusts the shape of the central modality data to be compatible with the model
        if not hasattr(batch[0][0], "input_ids"):
            central_modality_data = batch[0][0].central_modality_data.reshape(
                self.batch_size, -1
            )
        else:
            central_modality_data = (
                batch[0][0].input_ids.reshape(self.batch_size, -1),
                batch[0][0].attention_mask.reshape(self.batch_size, -1),
            )
        return {
            self.central_modality: central_modality_data,
            "graph": batch[0],
        }
