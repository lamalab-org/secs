import contextlib
import os
from pathlib import Path
from uuid import uuid1

import torch
from info_nce import InfoNCE
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn.functional import cosine_similarity
from torch.optim import Optimizer
from torch_geometric.data import Data
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
        self.world_size = int(
            os.environ.get(
                "WORLD_SIZE", cfg.trainer.gpus_per_node * cfg.trainer.num_nodes
            )
        )
        if hasattr(cfg, "ckpt_path") and cfg.ckpt_path is not None:
            with contextlib.suppress(FileNotFoundError):
                self.model.load_state_dict(
                    rename_keys_with_prefix(torch.load(cfg.ckpt_path)["state_dict"]),
                    strict=False,
                )
                logger.info(
                    f"Successfully loaded model from checkpoint: {cfg.ckpt_path}"
                )
        else:
            logger.info("No checkpoint path found. Training from scratch.")

        self.config = cfg
        self.loss = InfoNCE(
            temperature=cfg.model.loss.temperature, negative_mode="unpaired"
        )
        if cfg.model.loss.sparse:
            self.l1_loss = torch.nn.L1Loss()
        self.per_device_batch_size = cfg.data.batch_size
        self.batch_size = self.per_device_batch_size * self.world_size
        self.central_modality = cfg.data.central_modality
        self.tracker = 0

        logger.info(f"Per device batch size: {self.per_device_batch_size}")
        logger.info(f"Loss batch size: {self.batch_size}")

    def forward(self, batch: tuple[Data] | dict) -> dict:
        if isinstance(batch, tuple) and isinstance(batch[0], Data):
            dict_input_data = self._treat_graph_batch(batch[0])
            forward_pass = self.model(dict_input_data)
        else:
            forward_pass = self.model(batch)
        return forward_pass

    def _info_nce_loss(self, z1: Tensor, z2: Tensor) -> float:
        if self.world_size > 1:
            # shape (World_Size, Batch_Size, Embedding_Size)
            all_z1 = self.all_gather(z1, sync_grads=True)
            all_z2 = self.all_gather(z2, sync_grads=True)
            # flattened shape (World_Size*Batch_Size, Embedding_Size)
            all_z1 = all_z1.flatten(0, 1)
            all_z2 = all_z2.flatten(0, 1)
            return self.loss(all_z1, all_z2)
        else:
            return self.loss(z1, z2)

    def _multimodal_loss(
        self, embeddings_dict: dict[str, Tensor], prefix: str
    ) -> float:
        modality_pair = [*embeddings_dict]
        central_to_modality_loss = self._info_nce_loss(
            embeddings_dict[modality_pair[0]], embeddings_dict[modality_pair[1]]
        )
        if self.config.model.loss.symmetric:
            modality_to_central_loss = self._info_nce_loss(
                embeddings_dict[modality_pair[1]], embeddings_dict[modality_pair[0]]
            )
            loss = (central_to_modality_loss + modality_to_central_loss) / 2
        else:
            loss = central_to_modality_loss.detach().clone()
        if self.config.model.loss.sparse:
            loss = loss + self.config.model.loss.l1_loss_weight * self.l1_loss(
                embeddings_dict[modality_pair[0]], embeddings_dict[modality_pair[1]]
            )
        #  check if loss is nan
        if torch.isnan(loss):
            logger.error(f"Loss is nan for {prefix} batch.")
        self.log(
            f"{prefix}_loss",
            loss,
            batch_size=self.batch_size,
            sync_dist=self.world_size > 1,
        )
        # compute retrieval metrics
        k_list = [1, 5]
        if prefix in ["valid", "test", "predict"]:
            self.retrieval_metrics(
                embeddings_dict[modality_pair[0]],
                embeddings_dict[modality_pair[1]],
                *modality_pair,
                k_list,
                prefix=prefix,
            )
        return loss

    def training_step(self, batch: dict) -> Tensor:
        embeddings_dict = self.forward(batch)
        return self._multimodal_loss(embeddings_dict, "train")

    def validation_step(self, batch: dict) -> Tensor:
        embeddings_dict = self.forward(batch)
        return self._multimodal_loss(embeddings_dict, "valid")

    def predict_step(self, batch: dict) -> Tensor:
        return self.forward(batch)

    def test_step(self, batch: dict) -> Tensor:
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
                directory_for_embeddings = directory_for_embeddings + "_" + str(uuid1())
                Path(directory_for_embeddings).mkdir()
                logger.info(f"New directory created at {directory_for_embeddings}")
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
        k_list: list[int],
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
        if self.world_size > 1:
            # both all gather calls return tensors of shape (World_Size, Batch_Size, Embedding_Size)
            all_embeddings_central_mod = self.all_gather(
                embeddings_central_mod, sync_grads=True
            )
            all_embeddings_other_mod = self.all_gather(
                embeddings_other_mod, sync_grads=True
            )
            all_embeddings_central_mod = all_embeddings_central_mod.flatten(0, 1)
            all_embeddings_other_mod = all_embeddings_other_mod.flatten(0, 1)
        else:
            all_embeddings_central_mod = embeddings_central_mod.detach().clone()
            all_embeddings_other_mod = embeddings_other_mod.detach().clone()
        device = select_device()

        # reference: https://medium.com/@dhruvbird/all-pairs-cosine-similarity-in-pytorch-867e722c8572
        # adding a third dim allows to compute pairwise cosine sim.
        cos_sim = cosine_similarity(
            all_embeddings_central_mod.unsqueeze(1),
            all_embeddings_other_mod.unsqueeze(0),
            dim=2,
        ).to(device)
        # preds, target, indexes
        flatten_cos_sim = cos_sim.flatten().to(device)  # (Batch Size*Batch Size)

        # the metric calculations are grouped by indexes and then averaged
        # repeat interleave creates tensors of the form [0, 0, 1, 1, 2, 2]
        indexes = (
            torch.arange(all_embeddings_central_mod.shape[0])
            .repeat_interleave(all_embeddings_other_mod.shape[0])
            .to(device)
        )
        # Diagonal elements are the true querries, the rest are false querries
        target = (
            torch.eye(all_embeddings_central_mod.shape[0], dtype=torch.long)
            .flatten()
            .to(device)
        )
        assert target.sum() == all_embeddings_central_mod.shape[0]
        for k_val in k_list:
            for metric, metric_name in zip(metrics, metric_names):
                metric_to_log = metric(top_k=k_val)
                metric_to_log.update(flatten_cos_sim, target, indexes=indexes)
                self.log(
                    f"{prefix}_{central_modality}_{other_modality}_{metric_name}_top_{k_val}",
                    metric_to_log.compute(),
                    batch_size=self.per_device_batch_size * self.world_size,
                    sync_dist=self.world_size > 1,
                )

    def _treat_graph_batch(self, batch: Data) -> dict:
        # pos is only in "structure" modality and otherwise "graph"
        modality = batch.modality[0]
        # this adjusts the shape of the central modality data to be compatible with the model
        batch_size = len(batch.central_modality)
        if not hasattr(batch, "input_ids"):
            central_modality_data = batch.central_modality_data.reshape(batch_size, -1)
        else:
            central_modality_data = (
                batch.input_ids.reshape(batch_size, -1),
                batch.attention_mask.reshape(batch_size, -1),
            )
        return {
            self.central_modality: central_modality_data,
            modality: batch,
        }
