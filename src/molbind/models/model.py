from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from molbind.data.available import AVAILABLE_ENCODERS
from molbind.models.components.head import ProjectionHead


class MolBind(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        modalities = cfg.data.modalities
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

    def forward(
        self, input_data: Dict[str, Union[Tuple[Tensor, Tensor], Tensor]]
    ) -> Tensor:
        store_embeddings = {}
        # input_data = [data, batch_index, dataloader_index]
        input_data, _, _ = input_data
        # input_data is a dictionary with (smiles, modality) pairs (where the central modality is at index 0)
        modality = [*input_data][1]
        # store embeddings as store_embeddings[modality] = (smiles_embedding, modality_embedding)
        # forward through respective encoder
        smiles_embedding = self.dict_encoders[self.central_modality].forward(
            input_data[self.central_modality]
        )
        modality_embedding = self.dict_encoders[modality].forward(input_data[modality])
        central_modality_embedding_projected = self.dict_projection_heads[
            self.central_modality
        ](smiles_embedding)
        modality_embedding_projected = self.dict_projection_heads[modality](
            modality_embedding
        )
        # projection heads
        store_embeddings[self.central_modality] = central_modality_embedding_projected
        store_embeddings[modality] = modality_embedding_projected
        return store_embeddings

    def load_from_checkpoint(self, path: str):
        return torch.load(path)
