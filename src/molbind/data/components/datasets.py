from typing import Optional, Tuple

from torch import Tensor
from torch.utils.data import Dataset


class StringDataset(Dataset):
    def __init__(
        self,
        dataset: Tuple[Tensor, Tensor],
        modality: str,
        central_modality: str = "smiles",
        context_length: Optional[int] = 256,
    ):
        """Dataset for string modalities.

        Args:
            dataset (Tuple[Tensor, Tensor]): pair of (central_modality, modality) where the central modality is at index 0.
            modality (str): name of data modality as found in MODALITY_DATA_TYPES
            context_length (int, optional): _description_. Defaults to 256.
        """
        from molbind.data.available import MODALITY_DATA_TYPES, STRING_TOKENIZERS

        assert len(dataset) == 2
        assert len(dataset[0]) == len(dataset[1])

        self.modality = modality
        self.central_modality = central_modality

        assert MODALITY_DATA_TYPES[modality] == str
        assert MODALITY_DATA_TYPES[central_modality] == str

        self.tokenized_central_modality = STRING_TOKENIZERS[central_modality](
            dataset[0],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=context_length,
        )

        self.tokenized_string = STRING_TOKENIZERS[modality](
            dataset[1],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=context_length,
        )

    def __len__(self):
        return len(self.tokenized_central_modality.input_ids)

    def __getitem__(self, idx):
        return {
            self.central_modality: (
                self.tokenized_central_modality.input_ids[idx],
                self.tokenized_central_modality.attention_mask[idx],
            ),
            self.modality: (
                self.tokenized_string.input_ids[idx],
                self.tokenized_string.attention_mask[idx],
            ),
        }


class FingerprintMolBindDataset(Dataset):
    def __init__(
        self,
        dataset: Tuple[Tensor, Tensor],
        central_modality: str = "smiles",
        context_length: Optional[int] = 256,
    ):
        """Dataset for fingerprints.

        Args:
            dataset (Tuple[Tensor, Tensor]): pair of (central_modality, fingerprint) where the central modality is at index 0.
            modality (str): name of data modality as found in MODALITY_DATA_TYPES
            context_length (int, optional): _description_. Defaults to 256.
        """
        from molbind.data.available import MODALITY_DATA_TYPES, STRING_TOKENIZERS

        assert len(dataset) == 2
        assert len(dataset[0]) == len(dataset[1])

        self.dataset = dataset
        self.modality = "fingerprint"
        self.central_modality = central_modality
        self.context_length = context_length

        assert MODALITY_DATA_TYPES[self.modality] == str
        assert MODALITY_DATA_TYPES[central_modality] == str

        self.tokenized_central_modality = STRING_TOKENIZERS[central_modality](
            dataset[0],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=context_length,
        )

        self.fingerprints = dataset[1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            self.central_modality: self.central_modality[idx],
            self.modality: self.fingerprints[idx],
        }


class GraphDataset(Dataset):
    pass


class FingerprintVAEDataset(Dataset):
    def __init__(
        self,
        dataset: Tensor,
    ):
        self.fingerprints = dataset

    def __len__(self):
        return len(self.fingerprints)

    def __getitem__(self, idx):
        return self.fingerprints[idx]
