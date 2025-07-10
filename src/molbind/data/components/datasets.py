import torch
from torch import Tensor
from torch.utils.data import Dataset


def _string(input: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    return input


class StringDataset(Dataset):
    def __init__(
        self,
        central_modality_data: tuple[Tensor, Tensor],
        other_modality_data: tuple[Tensor, Tensor],
        central_modality: str,
        other_modality: str,
    ) -> None:
        from molbind.data.available import ModalityConstants

        self.central_modality = central_modality
        self.other_modality = other_modality
        self.central_modality_data = central_modality_data
        self.other_modality_data = other_modality_data
        self.central_modality_data_type = ModalityConstants[central_modality].data_type
        self.other_modality_data_type = ModalityConstants[other_modality].data_type

        # Cache data type checks for better performance
        self.central_is_string = isinstance(self.central_modality_data_type, str)
        self.other_is_string = isinstance(self.other_modality_data_type, str)

    def __len__(self):
        return len(self.other_modality_data[0])

    def __getitem__(self, idx):
        # Optimized data access with cached type checks
        if self.central_is_string:
            central_data = tuple(i[idx] for i in self.central_modality_data)
        else:
            central_data = (self.central_modality_data[0][idx], self.central_modality_data[1][idx])

        if self.other_is_string:
            other_data = tuple(i[idx] for i in self.other_modality_data)
        else:
            other_data = (self.other_modality_data[0][idx], self.other_modality_data[1][idx])

        return {
            self.central_modality: central_data,
            self.other_modality: other_data,
        }


class StringDatasetEmbedding(Dataset):
    def __init__(
        self,
        data: list[list[str]],
    ) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx][0]),
            torch.tensor(self.data[idx][1]),
        )
