from enum import Enum, StrEnum  # noqa: I002

from networkx import Graph
from numpy import ndarray

from molbind.data.components.datasets import (
    FingerprintMolBindDataset,
    GraphDataset,
    ImageDataset,
    StringDataset,
    StructureDataset,
    cNmrDataset,
)
from molbind.data.components.mb_tokenizers import (
    DESCRIPTION_TOKENIZER,
    SELFIES_TOKENIZER,
    SMILES_TOKENIZER,
)
from molbind.models.components.custom_encoders import (
    CustomFingerprintEncoder,
    CustomGraphEncoder,
    CustomStructureEncoder,
    DescriptionEncoder,
    ImageEncoder,
    IREncoder,
    IUPACNameEncoder,
    SelfiesEncoder,
    SmilesEncoder,
    cNmrEncoder,
)


class StringModalities(StrEnum):
    SMILES = "smiles"
    SELFIES = "selfies"
    IUPAC_NAME = "iupac_name"
    DESCRIPTION = "description"
    IR = "ir"
    MASS = "mass"


class NonStringModalities(StrEnum):
    IMAGE = "image"
    STRUCTURE = "structure"
    GRAPH = "graph"
    FINGERPRINT = "fingerprint"
    C_NMR = "c_nmr"


class ModalityConstants(Enum):
    """
    ModalityConstants[modality]: (data_type, dataset, encoder, tokenizer)
    """

    c_nmr = (str, cNmrDataset, cNmrEncoder, None)
    description = (str, StringDataset, DescriptionEncoder, DESCRIPTION_TOKENIZER)
    fingerprint = (list, FingerprintMolBindDataset, CustomFingerprintEncoder, None)
    ir = (str, StringDataset, IREncoder, None)
    iupac_name = (str, StringDataset, IUPACNameEncoder, None)
    smiles = (str, StringDataset, SmilesEncoder, SMILES_TOKENIZER)
    selfies = (str, StringDataset, SelfiesEncoder, SELFIES_TOKENIZER)
    graph = (Graph, GraphDataset, CustomGraphEncoder, None)
    structure = (Graph, StructureDataset, CustomStructureEncoder, None)
    image = (ndarray, ImageDataset, ImageEncoder, None)

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
