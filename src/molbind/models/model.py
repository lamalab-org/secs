import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor

from molbind.models.components.head import ProjectionHead


class MolBind(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        from molbind.data.available import ModalityConstants

        modalities = cfg.data.modalities
        central_modality = cfg.data.central_modality
        self.central_modality = central_modality
        logger.info(f"Non-central modalities: {modalities}")

        # Instantiate all encoders and projection heads
        dict_encoders, dict_projection_heads = {}, {}
        # Add other modalities to `dict_encoders` and `dict_projection_heads`
        for modality in [*modalities, central_modality]:
            if modality not in [*vars(ModalityConstants)]:
                raise ValueError(f"Modality {modality} not supported yet.")
            dict_encoders[modality] = ModalityConstants[modality].encoder(
                **cfg.model.encoders[modality]
            )

            if cfg.model.projection_heads[f"{modality}_is_on"]:
                dict_projection_heads[modality] = ProjectionHead(
                    **cfg.model.projection_heads[modality]
                )

        # convert dicts to nn.Moduledict
        self.dict_encoders = nn.ModuleDict(dict_encoders)
        self.dict_projection_heads = nn.ModuleDict(dict_projection_heads)

        # add requires grad to projection heads
        for modality, projection_head in self.dict_projection_heads.items():
            if cfg.model.projection_heads[f"{modality}_freeze"]:
                for param in projection_head.parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_data: dict[str, tuple[Tensor, Tensor] | Tensor],
    ) -> Tensor:
        store_embeddings = {}
        # Input data = [data, batch_index, dataloader_index]
        if isinstance(input_data, tuple):
            input_data, _, _ = input_data
            modality = [*input_data][1]
        if isinstance(input_data, dict):
            modality = [*input_data][1]
        # Input data is a dictionary with (central_modality, modality) pairs (where the central modality is at index 0)
        # Store embeddings as store_embeddings[modality] = (central_modality_embedding, modality_embedding)
        # Forward through respective encoder and projection head
        central_modality_embedding = self.dict_encoders[self.central_modality].forward(
            input_data[self.central_modality]
        )
        modality_embedding = self.dict_encoders[modality].forward(input_data[modality])

        if self.central_modality in self.dict_projection_heads:
            central_modality_embedding_projected = self.dict_projection_heads[
                self.central_modality
            ](central_modality_embedding)
            store_embeddings[self.central_modality] = (
                central_modality_embedding_projected
            )
        else:
            store_embeddings[self.central_modality] = central_modality_embedding
        if modality in self.dict_projection_heads:
            modality_embedding_projected = self.dict_projection_heads[modality](
                modality_embedding
            )
            # Projection heads
            store_embeddings[modality] = modality_embedding_projected
        else:
            store_embeddings[modality] = modality_embedding
        return store_embeddings

    def encode_modality(
        self, input_data: Tensor | tuple[Tensor, Tensor], modality: str
    ) -> Tensor:
        # forward pass through modality encoder
        embedding = self.dict_encoders[modality].forward(input_data)
        if modality in self.dict_projection_heads:
            embedding = self.dict_projection_heads[modality](embedding)
        return embedding
