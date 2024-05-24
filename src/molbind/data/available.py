from enum import StrEnum  # noqa: I002

from networkx import Graph

from molbind.data.components.datasets import (
    FingerprintMolBindDataset,
    GraphDataset,
    StringDataset,
)
from molbind.data.components.mb_tokenizers import (
    SELFIES_TOKENIZER,
    SMILES_TOKENIZER,
    TEXT_TOKENIZER,
)
from molbind.models.components.custom_encoders import (
    CustomFingerprintEncoder,
    CustomGraphEncoder,
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
}

MODALITY_DATA_TYPES = {
    "fingerprint": list,
    "iupac_name": str,
    "smiles": str,
    "selfies": str,
    "graph": Graph,
    "nmr": str,
    "ir": str,
}

STRING_TOKENIZERS = {
    "smiles": SMILES_TOKENIZER,
    "selfies": SELFIES_TOKENIZER,
    "iupac_name": TEXT_TOKENIZER,
}

MODALITY_DATASETS = {
    "fingerprint": FingerprintMolBindDataset,
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
    IR = "ir"
    NMR = "nmr"
    MASS = "mass"


class NonStringModalities(StrEnum):
    GRAPH = "graph"
    FINGERPRINT = "fingerprint"
