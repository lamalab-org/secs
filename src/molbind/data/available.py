from enum import StrEnum  # noqa: I002

from networkx import Graph

from molbind.data.components.datasets import (
    FingerprintMolBindDataset,
    GraphDataset,
    StringDataset,
    StructureDataset,
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
    IUPACNameEncoder,
    NMREncoder,
    SelfiesEncoder,
    SmilesEncoder,
)

AVAILABLE_ENCODERS = {
    "description": DescriptionEncoder,
    "fingerprint": CustomFingerprintEncoder,
    "graph": CustomGraphEncoder,
    "iupac_name": IUPACNameEncoder,
    "smiles": SmilesEncoder,
    "selfies": SelfiesEncoder,
    "structure": CustomStructureEncoder,
    "nmr": NMREncoder,
}

MODALITY_DATA_TYPES = {
    "description": str,
    "fingerprint": list,
    "ir": str,
    "iupac_name": str,
    "nmr": str,
    "smiles": str,
    "selfies": str,
    "graph": Graph,
    "structure": Graph,
}

STRING_TOKENIZERS = {
    "smiles": SMILES_TOKENIZER,
    "selfies": SELFIES_TOKENIZER,
    "description": DESCRIPTION_TOKENIZER,
}

MODALITY_DATASETS = {
    "description": StringDataset,
    "fingerprint": FingerprintMolBindDataset,
    "graph": GraphDataset,
    "ir": StringDataset,
    "iupac_name": StringDataset,
    "nmr": StringDataset,
    "selfies": StringDataset,
    "smiles": StringDataset,
    "structure": StructureDataset,
}


class StringModalities(StrEnum):
    SMILES = "smiles"
    SELFIES = "selfies"
    IUPAC_NAME = "iupac_name"
    DESCRIPTION = "description"
    IR = "ir"
    NMR = "nmr"
    MASS = "mass"


class NonStringModalities(StrEnum):
    STRUCTURE = "structure"
    GRAPH = "graph"
    FINGERPRINT = "fingerprint"
