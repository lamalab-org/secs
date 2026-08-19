from enum import Enum, StrEnum
from typing import NamedTuple

from transformers import PreTrainedTokenizerBase

from secs.data import GeneralDataset, HSQCDataset, IrDataset, StringDataset, cNmrDataset, hNmrDataset
from secs.data.components.secs_tokenizers import SMILES_TOKENIZER
from secs.models import (
    HSQCEncoder,
    IrCNNEncoder,
    SfmEmbeddingModel,
    SmilesEncoder,
    cNmrEncoder,
    hNmrCNNEncoder,
)


class ModalitySpec(NamedTuple):
    data_type: type
    dataset: type
    encoder: type
    tokenizer: PreTrainedTokenizerBase | None = None


class StringModalities(StrEnum):
    SMILES = "smiles"
    BIGSMILES = "bigsmiles"
    POLYMER_NAME = "polymer_name"
    IUPAC_NAME = "iupac_name"
    PSMILES = "psmiles"
    GENERAL = "general"


class NonStringModalities(StrEnum):
    C_NMR = "c_nmr"
    H_NMR = "h_nmr"
    IR = "ir"
    HSQC = "hsqc"
    GRAPH = "graph"
    STRUCTURE = "structure"
    BIGSMILES = "bigsmiles"
    POLYMER_NAME = "polymer_name"
    PSMILES = "psmiles"
    GENERAL = "general"


class ModalityConstants(Enum):
    """
    ModalityConstants[modality]: (data_type, dataset, encoder, tokenizer)
    """

    c_nmr = ModalitySpec(list, cNmrDataset, cNmrEncoder)
    h_nmr = ModalitySpec(list, hNmrDataset, hNmrCNNEncoder)
    ir = ModalitySpec(list, IrDataset, IrCNNEncoder)
    smiles = ModalitySpec(str, StringDataset, SmilesEncoder, SMILES_TOKENIZER)
    hsqc = ModalitySpec(list, HSQCDataset, HSQCEncoder)
    general = ModalitySpec(list, GeneralDataset, SfmEmbeddingModel)

    @property
    def data_type(self):
        return self.value[0]

    @property
    def dataset(self):
        return self.value[1]

    @property
    def encoder(self):
        return self.value[2]

    @property
    def tokenizer(self):
        return self.value[3]
