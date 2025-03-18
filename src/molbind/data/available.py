from enum import Enum, StrEnum

from molbind.data.components.datasets import (
    StringDataset,
)
from molbind.data.components.mb_tokenizers import (
    SELFIES_TOKENIZER,
    SMILES_TOKENIZER,
)
from molbind.models.components.custom_encoders import (
    IUPACNameEncoder,
    SelfiesEncoder,
    SmilesEncoder,
)


class StringModalities(StrEnum):
    SMILES = "smiles"
    SELFIES = "selfies"
    IUPAC_NAME = "iupac_name"
    DESCRIPTION = "description"
    PSMILES = "psmiles"
    BIGSMILES = "bigsmiles"


class NonStringModalities(StrEnum):
    C_NMR = "c_nmr"
    FINGERPRINT = "fingerprint"
    IMAGE = "image"
    GRAPH = "graph"
    H_NMR = "h_nmr"
    IR = "ir"
    MASS_SPEC_POSITIVE = "mass_spec_positive"
    MASS_SPEC_NEGATIVE = "mass_spec_negative"
    STRUCTURE = "structure"
    MULTI_SPEC = "multi_spec"
    H_NMR_CNN = "h_nmr_cnn"


class ModalityConstants(Enum):
    """
    ModalityConstants[modality]: (data_type, dataset, encoder, tokenizer)
    """
    iupac_name = (str, StringDataset, IUPACNameEncoder, None)
    smiles = (str, StringDataset, SmilesEncoder, SMILES_TOKENIZER)
    selfies = (str, StringDataset, SelfiesEncoder, SELFIES_TOKENIZER)
    psmiles = (str, StringDataset, SmilesEncoder, SMILES_TOKENIZER)
    bigsmiles = (str, StringDataset, SmilesEncoder, SMILES_TOKENIZER)

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
