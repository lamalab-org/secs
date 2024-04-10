from molbind.models.components.custom_encoders import (
    SmilesEncoder,
    SelfiesEncoder,
    GraphEncoder,
)
from molbind.models.components.head import ProjectionHead
from torch import Tensor
import torch.nn as nn
import torch
from typing import Dict, Union, Tuple


AVAILABLE_ENCODERS = {
    "smiles": SmilesEncoder,
    "selfies": SelfiesEncoder,
    "graph": GraphEncoder,
    "nmr": None,
}


class MolBind(nn.Module):
    def __init__(self, cfg):
        super(MolBind, self).__init__()

        modalities = cfg.data.modalities
        # Instantiate all encoders in modalities

        self.dict_encoders = {"smiles": SmilesEncoder(**cfg.model.encoders["smiles"])}
        self.dict_projection_heads = {
            "smiles": ProjectionHead(**cfg.model.projection_heads["smiles"])
        }

        for modality in modalities:
            if modality not in [*AVAILABLE_ENCODERS]:
                raise ValueError(f"Modality {modality} not supported yet.")
            self.dict_encoders[modality] = AVAILABLE_ENCODERS[modality](
                **cfg.model.encoders[modality]
            )
            self.dict_projection_heads[modality] = ProjectionHead(
                **cfg.model.projection_heads[modality]
            )

        # convert to nn.ModuleDict
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
        smiles_embedding = self.dict_encoders["smiles"].forward(input_data["smiles"])
        modality_embedding = self.dict_encoders[modality].forward(input_data[modality])
        smiles_embedding_projected = self.dict_projection_heads["smiles"](
            smiles_embedding
        )
        modality_embedding_projected = self.dict_projection_heads[modality](
            modality_embedding
        )
        # projection head
        store_embeddings["smiles"] = smiles_embedding_projected
        store_embeddings[modality] = modality_embedding_projected
        return store_embeddings

    def load_from_checkpoint(self, path: str):
        return torch.load(path)
