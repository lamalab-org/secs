from enum import Enum, StrEnum

from molbind.data.components.datasets import (
    StringDataset,
)
from molbind.data.components.mb_tokenizers import SMILES_TOKENIZER, POLYMER_NAME_TOKENIZER
from molbind.models.components.custom_encoders import PolymerNameEncoder, SmilesEncoder


class StringModalities(StrEnum):
    SMILES = "smiles"
    POLYMER_NAME = "polymer_name"
    PSMILES = "psmiles"
    BIGSMILES = "bigsmiles"


class ModalityConstants(Enum):
    """
    ModalityConstants[modality]: (data_type, dataset, encoder, tokenizer)
    """

    polymer_name = (str, StringDataset, PolymerNameEncoder, POLYMER_NAME_TOKENIZER)
    smiles = (str, StringDataset, SmilesEncoder, SMILES_TOKENIZER)
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
