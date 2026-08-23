from enum import Enum, StrEnum
from typing import NamedTuple

from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from transformers import PreTrainedTokenizerBase

from secs.data import GraphDataset, HSQCDataset, IrDataset, StringDataset, cNmrDataset, hNmrDataset
from secs.data.components.secs_tokenizers import SMILES_TOKENIZER


class ModalitySpec(NamedTuple):
    """
    Data-side description of a modality.
    """

    data_type: type
    dataset: type
    tokenizer: PreTrainedTokenizerBase | None = None


class StringModalities(StrEnum):
    SMILES = "smiles"


class NonStringModalities(StrEnum):
    C_NMR = "c_nmr"
    H_NMR = "h_nmr"
    IR = "ir"
    HSQC = "hsqc"
    GRAPH = "graph"


class ModalityConstants(Enum):
    """
    ModalityConstants[modality]: (data_type, dataset, tokenizer)
    """

    c_nmr = ModalitySpec(list, cNmrDataset)
    h_nmr = ModalitySpec(list, hNmrDataset)
    ir = ModalitySpec(list, IrDataset)
    smiles = ModalitySpec(str, StringDataset, SMILES_TOKENIZER)
    hsqc = ModalitySpec(list, HSQCDataset)
    graph = ModalitySpec(Data, GraphDataset)

    @property
    def data_type(self):
        return self.value[0]

    @property
    def dataset(self):
        return self.value[1]

    @property
    def tokenizer(self):
        return self.value[2]

    @property
    def loader(self) -> type:
        """
        The DataLoader class this modality's samples collate with.
        """
        return GeometricDataLoader if self.data_type is Data else DataLoader


def loader_for(*modalities: str) -> type:
    """The loader a sample pairing these modalities needs."""
    loaders = {ModalityConstants[modality].loader for modality in modalities}
    return GeometricDataLoader if GeometricDataLoader in loaders else DataLoader
