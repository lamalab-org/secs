from enum import Enum, StrEnum

from secs.data import GeneralDataset, HSQCDataset, IrDataset, StringDataset, cNmrDataset, hNmrDataset
from secs.data.components.mb_tokenizers import SMILES_TOKENIZER
from secs.models import (
    HSQCEncoder,
    IrCNNEncoder,
    SfmEmbeddingModel,
    SmilesEncoder,
    cNmrEncoder,
    hNmrCNNEncoder,
)


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

    c_nmr = (list, cNmrDataset, cNmrEncoder, None)
    h_nmr = (list, hNmrDataset, hNmrCNNEncoder, None)
    ir = (list, IrDataset, IrCNNEncoder, None)
    smiles = (str, StringDataset, SmilesEncoder, SMILES_TOKENIZER)
    hsqc = (list, HSQCDataset, HSQCEncoder, None)
    general = (list, GeneralDataset, SfmEmbeddingModel, None)

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
