from enum import Enum, StrEnum
from typing import NamedTuple

from transformers import PreTrainedTokenizerBase

from secs.data import HSQCDataset, IrDataset, StringDataset, cNmrDataset, hNmrDataset
from secs.data.components.secs_tokenizers import SMILES_TOKENIZER


class ModalitySpec(NamedTuple):
    """Data-side description of a modality.

    Which encoder a modality uses is not fixed here: several backbones can serve
    the same modality, and the config picks one through `secs.models.registry`.
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
    STRUCTURE = "structure"


class ModalityConstants(Enum):
    """
    ModalityConstants[modality]: (data_type, dataset, tokenizer)
    """

    c_nmr = ModalitySpec(list, cNmrDataset)
    h_nmr = ModalitySpec(list, hNmrDataset)
    ir = ModalitySpec(list, IrDataset)
    smiles = ModalitySpec(str, StringDataset, SMILES_TOKENIZER)
    hsqc = ModalitySpec(list, HSQCDataset)

    @property
    def data_type(self):
        return self.value[0]

    @property
    def dataset(self):
        return self.value[1]

    @property
    def tokenizer(self):
        return self.value[2]
