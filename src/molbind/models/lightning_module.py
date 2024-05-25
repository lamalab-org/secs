import contextlib  # noqa: I002
from pathlib import Path
from typing import Dict, List  # noqa: UP035
from uuid import uuid1

import torch
from info_nce import InfoNCE
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn.functional import cosine_similarity
from torch.optim import Optimizer
from torchmetrics.retrieval import (
    RetrievalMRR,
    RetrievalRecall,
)

from molbind.models import MolBind
from molbind.utils import rename_keys_with_prefix, select_device


class MolBindModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.model = MolBind(cfg=cfg)

        if hasattr(cfg, "ckpt_path") and cfg.ckpt_path is not None:
            with contextlib.suppress(FileNotFoundError):
                self.model.load_state_dict(
                    rename_keys_with_prefix(torch.load(cfg.ckpt_path)["state_dict"])
                )
                logger.info("Successfully loaded model from checkpoint.")
        else:
            logger.info("No checkpoint path found. Training from scratch.")
        self.config = cfg
        self.loss = InfoNCE(
            temperature=cfg.model.loss.temperature, negative_mode="unpaired"
        )
        self.batch_size = cfg.data.batch_size
        self.central_modality = cfg.data.central_modality
        self.tracker = 0

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
        self.log(f"{prefix}_loss", loss, batch_size=self.batch_size, sync_dist=True)
        # compute retrieval metrics
        k_list = [1, 5]
        if prefix in ["valid", "test"]:
            self.retrieval_metrics(
                embeddings_dict[modality_pair[0]],
                embeddings_dict[modality_pair[1]],
                *modality_pair,
                k_list,
                prefix=prefix,
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
        # self.store_embeddings.append(embeddings_dict)
        # store embeddings to file

        file_template = "{}/{}_embeddings_{}.pth"
        directory_for_embeddings = self.config.store_embeddings_directory
        # create a directory for embeddings if it does not exist
        if self.tracker == 0:
            if not Path(directory_for_embeddings).exists():
                Path.mkdir(directory_for_embeddings)
            else:
                logger.info(f"Directory {directory_for_embeddings} already exists.")
                logger.info("Creating new directory for embeddings..")
                new_directory_for_embeddings = directory_for_embeddings + uuid1()
                Path(new_directory_for_embeddings).mkdir()
                logger.info(f"New directory created at {new_directory_for_embeddings}")
        # convert to numpy array and then store to .pth file
        for modality, embeddings in embeddings_dict.items():
            torch.save(
                embeddings,
                file_template.format(directory_for_embeddings, modality, self.tracker),
            )
        self.tracker += 1
        return self._multimodal_loss(embeddings_dict, "test")

    def configure_optimizers(self) -> Optimizer:
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
        prefix: str,
    ) -> None:
        """
        Example:

        .unsqueeze(1) modality 1 embeddings of shape (Batch_Size, Embedding_Size) is equivalent
        to Tensor[:, None, :].
        .unsqueeze(0) modality 2 embeddings of shape (Batch_Size, Embedding_Size) is equivalent
        to Tensor[None, :, :].

        This allows to compute the matrix of cosine similarities between all pairs of embeddings
        across two tensors containing embeddings for different modalities.

        preds, targets, indexes are tensors of shape (Batch_Size*Batch_size)

        Example on a 2x2 matrix

        metric = RetrievalMRR(top_k=1)
        preds = torch.tensor([0.56, 0.3, 0.2, 0.7])
        preds = torch.tensor([[0.56, 0.3], [0.2, 0.7]]).flatten()
        preds shape change: (Batch_Size, Batch_Size) -> (Batch_Size*Batch_Size)
        # True corresponds to diagonal elements in our case
        # for larger examples we can use torch.eye(Batch_Size).flatten()
        target = torch.tensor([True, False, False, True])
        # These are query ids. Metrics are computed after grouping by query id and then averaging.
        # For large matrices we can use torch.repeat_interleave(torch.arange(Batch_Size))
        indexes = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        metric.update(preds, target, indexes)
        metric.compute()
        # tensor(1.) output: since the cosine similarity on the diagonals is the highest in their corresponding row
        """

        metrics = [
            RetrievalMRR,
            RetrievalRecall,
        ]
        metric_names = [metric.__name__ for metric in metrics]
        # reference: https://medium.com/@dhruvbird/all-pairs-cosine-similarity-in-pytorch-867e722c8572
        cos_sim = cosine_similarity(
            embeddings_central_mod.unsqueeze(1),
            embeddings_other_mod.unsqueeze(
                0
            ),  # adding a third dim allows to compute pairwise cosine sim.
            dim=2,
        )
        # preds, target, indexes
        flatten_cos_sim = cos_sim.flatten().to(
            select_device()
        )  # (Batch Size*Batch Size)

        # the metric calculations are grouped by indexes and then averaged
        # repeat interleave creates tensors of the form [0, 0, 1, 1, 2, 2]
        indexes = (
            torch.arange(embeddings_central_mod.shape[0])
            .repeat_interleave(embeddings_central_mod.shape[0])
            .to(select_device())
        )
        # Diagonal elements are the true querries, the rest are false querries
        target = (
            torch.eye(embeddings_central_mod.shape[0], dtype=torch.long)
            .flatten()
            .to(select_device())
        )

        for metric, metric_name in zip(metrics, metric_names):
            for k_val in k_list:
                metric_to_log = metric(top_k=k_val)
                metric_to_log.update(flatten_cos_sim, target, indexes)
                self.log(
                    f"{prefix}_{central_modality}_{other_modality}_{metric_name}_top_{k_val}",
                    metric_to_log.compute(),
                    batch_size=self.batch_size,
                    sync_dist=True,
                )

    def _treat_graph_batch(self, batch: Dict) -> Dict:  # noqa: UP006
        # this adjusts the shape of the central modality data to be compatible with the model
        if not hasattr(batch[0], "input_ids"):
            central_modality_data = batch[0].central_modality_data.reshape(
                self.batch_size, -1
            )
        else:
            central_modality_data = (
                batch[0].input_ids.reshape(self.batch_size, -1),
                batch[0].attention_mask.reshape(self.batch_size, -1),
            )
        return {
            self.central_modality: central_modality_data,
            "graph": batch[0],
        }
