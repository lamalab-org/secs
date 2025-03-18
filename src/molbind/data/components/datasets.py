import random
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torchvision import transforms

from molbind.data.utils.graph_utils import (
    get_item_for_dimenet,
    smiles_to_graph_without_augment,
)


def _fingerprint(fingerprint: list[int]) -> Tensor:
    return Tensor(fingerprint)


def _string(input: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    return input


class StringDataset(Dataset):
    def __init__(
        self,
        central_modality_data: tuple[Tensor, Tensor],
        other_modality_data: tuple[Tensor, Tensor],
        central_modality: str,
        other_modality: str,
    ) -> None:
        from molbind.data.available import ModalityConstants

        self.central_modality = central_modality
        self.other_modality = other_modality
        self.central_modality_data = central_modality_data
        self.other_modality_data = other_modality_data
        self.central_modality_data_type = ModalityConstants[central_modality].data_type
        self.other_modality_data_type = ModalityConstants[other_modality].data_type

    def __len__(self):
        return len(self.other_modality_data[0])

    def __getitem__(self, idx):
        central_data = (
            tuple(i[idx] for i in self.central_modality_data)
            if isinstance(self.central_modality_data_type, str)
            else (self.central_modality_data[0][idx], self.central_modality_data[1][idx])
        )
        other_data = (
            tuple(i[idx] for i in self.other_modality_data)
            if isinstance(self.other_modality_data_type, str)
            else (self.other_modality_data[0][idx], self.other_modality_data[1][idx])
        )

        return {
            self.central_modality: central_data,
            self.other_modality: other_data,
        }


class FingerprintMolBindDataset(Dataset):
    def __init__(
        self,
        central_modality_data: tuple[Tensor, Tensor],
        fingerprint_data: list[list[int]],
        central_modality: str,
    ) -> None:
        """Dataset for fingerprints.

        Args:
            central_modality_data (tuple[Tensor, Tensor]): pair of (central_modality, tokenized_central_modality)
            fingerprint_data (Tensor): fingerprint data
            central_modality (str): name of central modality as found in ModalityConstants
        Returns:
            None
        """
        self.central_modality_data = central_modality_data
        self.central_modality = central_modality
        self.other_modality = "fingerprint"
        self.fingerprints = fingerprint_data

    def __len__(self):
        return len(self.fingerprints)

    def __getitem__(self, idx: int) -> dict:
        return {
            self.central_modality: [i[idx] for i in self.central_modality_data],
            self.other_modality: Tensor(self.fingerprints[idx]),
        }


class GraphDataset(Dataset):
    def __init__(
        self,
        graph_data: pd.DataFrame,
        central_modality: str,
        central_modality_data: list[int] | Tensor | tuple[Tensor, Tensor],
    ) -> None:
        """Dataset for the graph modality (MolCLR).

        Args:
            graph_data (pl.DataFrame): graph data as a polars DataFrame
            central_modality (str): name of central modality as found in ModalityConstants
            central_modality_data (list[int] | Tensor | tuple[Tensor, Tensor]]): central modality data
            that is either a tensor or a tuple of tensors depending on the data type
        Returns:
            None
        """

        super().__init__()
        from molbind.data.available import (
            ModalityConstants,
            NonStringModalities,
            StringModalities,
        )

        self.central_modality = central_modality
        self.modality = "graph"
        self.smiles_list = graph_data[self.modality].to_list()
        self.central_modality_data = central_modality_data
        self.central_modality_data_type = ModalityConstants[central_modality].data_type
        # modality handler functions if a modality is the central modality
        self.central_modality_handlers = {
            StringModalities.SMILES: _string,
            StringModalities.SELFIES: _string,
            StringModalities.IUPAC_NAME: _string,
            NonStringModalities.FINGERPRINT: _fingerprint,
            NonStringModalities.IR: _fingerprint,
            NonStringModalities.C_NMR: _fingerprint,
            NonStringModalities.MASS_SPEC_POSITIVE: _fingerprint,
            NonStringModalities.MASS_SPEC_NEGATIVE: _fingerprint,
            NonStringModalities.H_NMR: _fingerprint,
            NonStringModalities.MULTI_SPEC: _fingerprint,
        }

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, index: int) -> Data:
        """
        For graph data, the central modality is added as an attribute to the graph data.
        Then the data is reshaped to a Tensor of size (batch_size, N).
        """
        data = smiles_to_graph_without_augment(self.smiles_list[index])
        data.central_modality = self.central_modality
        if not isinstance(self.central_modality_data_type, str):
            data.central_modality_data = self.central_modality_handlers[data.central_modality](self.central_modality_data[index])
        else:
            data.input_ids = self.central_modality_handlers[data.central_modality](self.central_modality_data[0][index])
            data.attention_mask = self.central_modality_handlers[data.central_modality](self.central_modality_data[1][index])
        data.modality = self.modality
        return data


class StringDatasetEmbedding(Dataset):
    def __init__(
        self,
        data: list[list[str]],
    ) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx][0]),
            torch.tensor(self.data[idx][1]),
        )
