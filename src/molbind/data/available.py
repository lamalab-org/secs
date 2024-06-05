from enum import StrEnum  # noqa: I002

from networkx import Graph

from molbind.data.components.datasets import (
    FingerprintMolBindDataset,
    GraphDataset,
    StringDataset,
)
from molbind.data.components.mb_tokenizers import (
    DESCRIPTION_TOKENIZER,
    SELFIES_TOKENIZER,
    SMILES_TOKENIZER,
)
from molbind.models.components.custom_encoders import (
    CustomFingerprintEncoder,
    CustomGraphEncoder,
    DescriptionEncoder,
    IUPACNameEncoder,
    SelfiesEncoder,
    SmilesEncoder,
)

AVAILABLE_ENCODERS = {
    "smiles": SmilesEncoder,
    "selfies": SelfiesEncoder,
    "graph": CustomGraphEncoder,
    "nmr": None,
    "fingerprint": CustomFingerprintEncoder,
    "iupac_name": IUPACNameEncoder,
    "description": DescriptionEncoder,
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
}

STRING_TOKENIZERS = {
    "smiles": SMILES_TOKENIZER,
    "selfies": SELFIES_TOKENIZER,
    "description": DESCRIPTION_TOKENIZER,
}

MODALITY_DATASETS = {
    "fingerprint": FingerprintMolBindDataset,
    "description": StringDataset,
    "iupac_name": StringDataset,
    "smiles": StringDataset,
    "selfies": StringDataset,
    "graph": GraphDataset,
    "nmr": StringDataset,
    "ir": StringDataset,
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
    GRAPH = "graph"
    FINGERPRINT = "fingerprint"
