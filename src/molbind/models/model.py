from typing import Dict, Tuple, Union  # noqa: I002, UP035

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor

from molbind.models.components.head import ProjectionHead
from molbind.utils import select_device


class MolBind(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        from molbind.data.available import AVAILABLE_ENCODERS

        modalities = cfg.data.modalities
        logger.info(f"Modalities: {modalities}")
        central_modality = cfg.data.central_modality
        self.central_modality = central_modality

        # Instantiate all encoders and projection heads
        self.dict_encoders = {
            central_modality: AVAILABLE_ENCODERS[central_modality](
                **cfg.model.encoders[central_modality]
            )
        }
        self.dict_projection_heads = {
            central_modality: ProjectionHead(
                **cfg.model.projection_heads[central_modality]
            )
        }
        # Add other modalities to `dict_encoders` and `dict_projection_heads`
        for modality in modalities:
            if modality not in [*AVAILABLE_ENCODERS]:
                raise ValueError(f"Modality {modality} not supported yet.")
            self.dict_encoders[modality] = AVAILABLE_ENCODERS[modality](
                **cfg.model.encoders[modality]
            )
            self.dict_projection_heads[modality] = ProjectionHead(
                **cfg.model.projection_heads[modality]
            )

        # convert dicts to nn.ModuleDict
        self.dict_encoders = nn.ModuleDict(self.dict_encoders)
        self.dict_projection_heads = nn.ModuleDict(self.dict_projection_heads)

        # store the batch size
        self.batch_size = cfg.data.batch_size

    def forward(
        self,
        input_data: Dict[str, Union[Tuple[Tensor, Tensor], Tensor]],  # noqa: UP006
    ) -> Tensor:
        store_embeddings = {}
        # Input data = [data, batch_index, dataloader_index]
        if isinstance(input_data, tuple):
            input_data, _, _ = input_data
            modality = [*input_data][1]
        if isinstance(input_data, dict):
            modality = [*input_data][1]
        else:
            raise ValueError("Input data should be a `tuple` or a `dictionary`.")
        # Input data is a dictionary with (central_modality, modality) pairs (where the central modality is at index 0)
        # Store embeddings as store_embeddings[modality] = (central_modality_embedding, modality_embedding)
        # Forward through respective encoder and projection head
        central_modality_embedding = self.dict_encoders[self.central_modality].forward(
            input_data[self.central_modality]
        )
        modality_embedding = self.dict_encoders[modality].forward(input_data[modality])
        central_modality_embedding_projected = self.dict_projection_heads[
            self.central_modality
        ](central_modality_embedding)
        modality_embedding_projected = self.dict_projection_heads[modality](
            modality_embedding
        )
        # Projection heads
        store_embeddings[self.central_modality] = central_modality_embedding_projected
        store_embeddings[modality] = modality_embedding_projected
        return store_embeddings

    def load_from_checkpoint(self, path: str):
        return torch.load(path, map_location=select_device())["state_dict"]
