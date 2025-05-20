from enum import Enum, StrEnum

from molbind.data.components.datasets import (
    FingerprintMolBindDataset,
    IrDataset,
    MassSpecNegativeDataset,
    MassSpecPositiveDataset,
    StringDataset,
    cNmrDataset,
    hNmrDataset,
)
from molbind.data.components.mb_tokenizers import (
    SMILES_TOKENIZER,
)
from molbind.models.components.custom_encoders import (
    CustomFingerprintEncoder,
    IrCNNEncoder,
    MassSpecNegativeEncoder,
    MassSpecPositiveEncoder,
    SmilesEncoder,
    cNmrEncoder,
    hNmrCNNEncoder,
)


class StringModalities(StrEnum):
    SMILES = "smiles"
    SELFIES = "selfies"
    IUPAC_NAME = "iupac_name"
    DESCRIPTION = "description"


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


class ModalityConstants(Enum):
    """
    ModalityConstants[modality]: (data_type, dataset, encoder, tokenizer)
    """

    c_nmr = (list, cNmrDataset, cNmrEncoder, None)
    fingerprint = (list, FingerprintMolBindDataset, CustomFingerprintEncoder, None)
    h_nmr = (list, hNmrDataset, hNmrCNNEncoder, None)
    ir = (list, IrDataset, IrCNNEncoder, None)
    mass_spec_negative = (list, MassSpecNegativeDataset, MassSpecNegativeEncoder, None)
    mass_spec_positive = (list, MassSpecPositiveDataset, MassSpecPositiveEncoder, None)
    smiles = (str, StringDataset, SmilesEncoder, SMILES_TOKENIZER)

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
