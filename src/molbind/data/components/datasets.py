from typing import List, Tuple, Union  # noqa: UP035, I002

import polars as pl
from torch import Tensor
from torch.utils.data import Dataset

from molbind.data.utils.graph_utils import smiles_to_graph


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
        from molbind.data.available import MODALITY_DATA_TYPES

        # modality pair definition
        self.central_modality = central_modality
        self.other_modality = other_modality
        # modality pair data
        self.central_modality_data = central_modality_data
        self.other_modality_data = other_modality_data
        self.central_modality_data_type = MODALITY_DATA_TYPES[central_modality]
        self.other_modality_data_type = MODALITY_DATA_TYPES[other_modality]

    def __len__(self):
        return len(self.other_modality_data[0])

    def __getitem__(self, idx):
        return {
            self.central_modality: tuple([i[idx] for i in self.central_modality_data])
            if self.central_modality_data_type == str
            else Tensor(self.central_modality_data[idx]),
            self.other_modality: tuple([i[idx] for i in self.other_modality_data])
            if self.other_modality_data_type == str
            else Tensor(self.other_modality_data)[idx],
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
            self.central_modality: [i[idx] for i in self.central_modality_data],
            self.other_modality: Tensor(self.fingerprints[idx]),
        }


class GraphDataset(Dataset):
    def __init__(
        self,
        graph_data: pl.DataFrame,
        central_modality: str,
        central_modality_data: Union[List[int], Tensor, Tuple[Tensor, Tensor]],  # noqa: UP006
    ) -> None:
        """Dataset for the graph modality (MolCLR).

        Args:
            graph_data (pl.DataFrame): graph data as a polars DataFrame
            central_modality (str): name of central modality as found in MODALITY_DATA_TYPES
            central_modality_data (Union[Tensor, Tuple[Tensor, Tensor]]): central modality data
            that is either a tensor or a tuple of tensors depending on the data type
        Returns:
            None
        """

        super().__init__()
        from molbind.data.available import (
            MODALITY_DATA_TYPES,
            NonStringModalities,
            StringModalities,
        )

        self.central_modality = central_modality
        self.modality = "graph"
        self.smiles_list = graph_data[self.modality].to_list()
        self.central_modality_data = central_modality_data
        self.central_modality_data_type = MODALITY_DATA_TYPES[central_modality]
        # modality handler functions if a modality is the central modality
        self.central_modality_handlers = {
            StringModalities.SMILES: self._string,
            StringModalities.SELFIES: self._string,
            StringModalities.INCHI: self._string,
            StringModalities.IR: self._string,
            StringModalities.NMR: self._string,
            StringModalities.MASS: self._string,
            NonStringModalities.FINGERPRINT: self._fingerprint,
        }

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, index: int) -> Tuple:  # noqa: UP006
        """
        For graph data, the central modality is added as an attribute to the graph data.
        Then the data is reshaped to a Tensor of size (batch_size, N).
        """
        data_i, data_j = smiles_to_graph(self.smiles_list[index])
        data_i.central_modality = self.central_modality
        if self.central_modality_data_type != str:
            data_i.central_modality_data = self.central_modality_handlers[
                data_i.central_modality
            ](self.central_modality_data[index])
        else:
            data_i.input_ids = self.central_modality_handlers[data_i.central_modality](
                self.central_modality_data[0][index]
            )
            data_i.attention_mask = self.central_modality_handlers[
                data_i.central_modality
            ](self.central_modality_data[1][index])
        return data_i, data_j

    def _fingerprint(self, fingerprint: List[int]) -> Tensor:  # noqa: UP006
        return Tensor(fingerprint)

    def _string(self, input: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:  # noqa: UP006
        return input


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
