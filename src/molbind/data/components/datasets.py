from typing import Tuple  # noqa: UP035, I002

from torch import Tensor
from torch.utils.data import Dataset


class StringDataset(Dataset):
    def __init__(
        self,
        central_modality_data: Tuple[Tensor, Tensor],  # noqa: UP006
        other_modality_data: Tuple[Tensor, Tensor],  # noqa: UP006
        central_modality: str,
        other_modality: str,
    ) -> None:
        """Dataset for string modalities.

        Args:
            central_modality_data (Tuple[Tensor, Tensor]): pair of (central_modality, tokenized_central_modality)
            other_modality_data (Tuple[Tensor, Tensor]): pair of (other_modality, tokenized_other_modality)
            central_modality (str): name of central modality as found in MODALITY_DATA_TYPES
            other_modality (str): name of other modality as found in MODALITY_DATA_TYPES
        """
        self.central_modality_data = central_modality_data
        self.central_modality = central_modality
        self.other_modality = other_modality
        self.tokenized_central_modality = central_modality_data
        self.tokenized_other_modality = other_modality_data

    def __len__(self):
        return len(self.tokenized_central_modality[0])

    def __getitem__(self, idx):
        return {
            self.central_modality: (
                self.central_modality_data[0][idx],
                self.central_modality_data[1][idx],
            ),
            self.modality: self.other_modality_data[idx],
        }


class FingerprintMolBindDataset(Dataset):
    def __init__(
        self,
        central_modality_data: Tuple[Tensor, Tensor],  # noqa: UP006
        fingerprint_data: Tensor,
        central_modality: str,
    ) -> None:
        """Dataset for fingerprints.

        Args:
            central_modality_data (Tuple[Tensor, Tensor]): pair of (central_modality, tokenized_central_modality)
            fingerprint_data (Tensor): fingerprint data
            central_modality (str): name of central modality as found in MODALITY_DATA_TYPES
        Returns:
            None
        """
        self.central_modality_data = central_modality_data
        self.central_modality = central_modality
        self.other_modality = "fingerprint"
        self.fingerprints = fingerprint_data

    def __len__(self):
        return len(self.fingerprints)

    def __getitem__(self, idx):
        return {
            self.central_modality: (
                self.central_modality_data[0][idx],
                self.central_modality_data[1][idx],
            ),
            self.other_modality: Tensor(self.fingerprints[idx]),
        }


class GraphDataset(Dataset):
    pass


class FingerprintVAEDataset(Dataset):
    def __init__(
        self,
        dataset: Tensor,
    ):
        """Dataset for fingerprints for the VAE model."""
        self.fingerprints = dataset

    def __len__(self):
        return len(self.fingerprints)

    def __getitem__(self, idx):
        return self.fingerprints[idx]
