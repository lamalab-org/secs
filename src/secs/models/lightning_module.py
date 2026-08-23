import math
import os

import torch
import torch.nn as nn
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

from secs.models.model import MolBind
from secs.utils import select_device


class SECSModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.model = MolBind(cfg=cfg)
        self.world_size = int(os.environ.get("WORLD_SIZE", cfg.trainer.gpus_per_node * cfg.trainer.num_nodes))
        self.config = cfg
        self.loss = InfoNCE(temperature=cfg.model.loss.temperature, negative_mode="unpaired")

        self.learnable_temperature = bool(getattr(cfg.model.loss, "learnable_temperature", False))
        if self.learnable_temperature:
            self.log_inv_temperature = nn.Parameter(torch.tensor(math.log(1.0 / cfg.model.loss.temperature)))
            self.max_inv_temperature = float(getattr(cfg.model.loss, "max_inv_temperature", 100.0))

        self.per_device_batch_size = cfg.data.batch_size
        self.batch_size = self.per_device_batch_size * self.world_size
        self.central_modality = cfg.data.central_modality

        logger.info(f"Per device batch size: {self.per_device_batch_size}")
        logger.info(f"Loss batch size: {self.batch_size}")

        self._load_checkpoint(getattr(cfg, "ckpt_path", None))

    def _load_checkpoint(self, ckpt_path: str | None) -> None:
        """Loads a checkpoint."""
        if not ckpt_path or ckpt_path.strip().lower() in {"none", "null"}:
            logger.info("No checkpoint path found. Training from scratch.")
            return
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        except FileNotFoundError:
            logger.warning(f"Checkpoint {ckpt_path} not found. Training from scratch.")
            return
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint: {ckpt_path}")
        for name, keys in (("missing", missing), ("unexpected", unexpected)):
            if keys:
                logger.warning(f"Checkpoint has {len(keys)} {name} keys (e.g. {keys[:3]})")

    def forward(self, batch: tuple | dict) -> dict:
        return self.model(batch)

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

    def _multimodal_loss(self, embeddings_dict: dict[str, Tensor], prefix: str) -> float:
        if self.learnable_temperature:
            inv_t = self.log_inv_temperature.exp().clamp(max=self.max_inv_temperature)
            self.loss.temperature = 1.0 / inv_t
            self.log(f"{prefix}_temperature", 1.0 / inv_t.detach(), batch_size=self.batch_size, sync_dist=self.world_size > 1)

        # modality_pair[0] is the central modality (smiles), [1] the spectral one
        modality_pair = [*embeddings_dict]
        # queries = spectrum, keys = smiles: same direction as retrieval at inference
        modality_to_central_loss = self._info_nce_loss(embeddings_dict[modality_pair[1]], embeddings_dict[modality_pair[0]])
        if self.config.model.loss.symmetric:
            central_to_modality_loss = self._info_nce_loss(embeddings_dict[modality_pair[0]], embeddings_dict[modality_pair[1]])
            loss = (modality_to_central_loss + central_to_modality_loss) / 2
        else:
            loss = modality_to_central_loss
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

    def predict_step(self, batch: dict | tuple[Tensor, Tensor]) -> Tensor:
        if isinstance(batch, tuple):
            return self.forward(batch)
        return self.model.encode_modality(batch, self.central_modality)

    def configure_optimizers(self) -> Optimizer:
        groups = [
            {
                "params": list(self.model.parameters()),
                "weight_decay": self.config.model.optimizer.weight_decay,
            }
        ]
        if self.learnable_temperature:
            groups.append({"params": [self.log_inv_temperature], "weight_decay": 0.0})
        return torch.optim.AdamW(groups, lr=self.config.model.optimizer.lr)

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
        This allows to compute the matrix of cosine similarities between all pairs of embeddings
        across two tensors containing embeddings for different modalities.

        preds, targets, indexes are tensors of shape (Batch_Size*Batch_size)
        """

        metrics = [
            RetrievalMRR,
            RetrievalRecall,
        ]
        metric_names = [metric.__name__ for metric in metrics]
        if self.world_size > 1:
            # both all gather calls return tensors of shape (World_Size, Batch_Size, Embedding_Size)
            all_embeddings_central_mod = self.all_gather(embeddings_central_mod, sync_grads=True)
            all_embeddings_other_mod = self.all_gather(embeddings_other_mod, sync_grads=True)
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
            torch.arange(all_embeddings_central_mod.shape[0]).repeat_interleave(all_embeddings_other_mod.shape[0]).to(device)
        )
        # Diagonal elements are the true querries, the rest are false querries
        target = torch.eye(all_embeddings_central_mod.shape[0], dtype=torch.long).flatten().to(device)
        assert target.sum() == all_embeddings_central_mod.shape[0]
        for k_val in k_list:
            for metric, metric_name in zip(metrics, metric_names, strict=False):
                metric_to_log = metric(top_k=k_val)
                metric_to_log.update(flatten_cos_sim, target, indexes=indexes)
                self.log(
                    f"{prefix}_{central_modality}_{other_modality}_{metric_name}_top_{k_val}",
                    metric_to_log.compute(),
                    batch_size=self.per_device_batch_size * self.world_size,
                    sync_dist=self.world_size > 1,
                )
