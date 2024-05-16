from enum import StrEnum  # noqa: I002

from networkx import Graph

from molbind.data.components.datasets import (
    FingerprintMolBindDataset,
    GraphDataset,
    StringDataset,
)
from molbind.data.components.mb_tokenizers import (
    GALACTICA_125M_TOKENIZER,
    SELFIES_TOKENIZER,
    SMILES_TOKENIZER,
)
from molbind.models.components.custom_encoders import (
    CustomFingerprintEncoder,
    CustomGraphEncoder,
    NMREncoder,
    SelfiesEncoder,
    SmilesEncoder,
)

GALACTICA_125M_TOKENIZER.add_special_tokens({"pad_token": "<pad>"})

AVAILABLE_ENCODERS = {
    "smiles": SmilesEncoder,
    "selfies": SelfiesEncoder,
    "graph": CustomGraphEncoder,
    "nmr": NMREncoder,
    "fingerprint": CustomFingerprintEncoder,
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
    "nmr": GALACTICA_125M_TOKENIZER,
}

MODALITY_DATASETS = {
    "fingerprint": FingerprintMolBindDataset,
    "smiles": StringDataset,
    "selfies": StringDataset,
    "graph": GraphDataset,
    "nmr": StringDataset,
    "ir": StringDataset,
}


class StringModalities(StrEnum):
    SMILES = "smiles"
    SELFIES = "selfies"
    INCHI = "inchi"
    IR = "ir"
    NMR = "nmr"
    MASS = "mass"


class NonStringModalities(StrEnum):
    GRAPH = "graph"
    FINGERPRINT = "fingerprint"
