from typing import List, Literal, Optional, Tuple, Union  # noqa: UP035, I002

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from molbind.data.utils.graph_utils import (
    get_item_for_dimenet,
    smiles_to_graph_without_augment,
)


def _fingerprint(fingerprint: List[int]) -> Tensor:  # noqa: UP006
    return Tensor(fingerprint)


def _string(input: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:  # noqa: UP006
    return input


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
        fingerprint_data: List[List[int]],  # noqa: UP006
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

    def __getitem__(self, idx: int) -> dict:
        return {
            self.central_modality: [i[idx] for i in self.central_modality_data],
            self.other_modality: Tensor(self.fingerprints[idx]),
        }


class GraphDataset(Dataset):
    def __init__(
        self,
        graph_data: pd.DataFrame,
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
            StringModalities.SMILES: _string,
            StringModalities.SELFIES: _string,
            StringModalities.IUPAC_NAME: _string,
            StringModalities.IR: _string,
            StringModalities.NMR: _string,
            StringModalities.MASS: _string,
            NonStringModalities.FINGERPRINT: _fingerprint,
        }

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, index: int) -> Data:
        """
        For graph data, the central modality is added as an attribute to the graph data.
        Then the data is reshaped to a Tensor of size (batch_size, N).
        """
        data = smiles_to_graph_without_augment(self.smiles_list[index])
        data.central_modality = self.central_modality
        if self.central_modality_data_type != str:
            data.central_modality_data = self.central_modality_handlers[
                data.central_modality
            ](self.central_modality_data[index])
        else:
            data.input_ids = self.central_modality_handlers[data.central_modality](
                self.central_modality_data[0][index]
            )
            data.attention_mask = self.central_modality_handlers[data.central_modality](
                self.central_modality_data[1][index]
            )
        data.modality = self.modality
        return data


class StructureDataset(Dataset):
    """
    This dataset is used for 3D coordinates data.
    It can be used both for training DimeNet and in MolBind.
    """

    def __init__(
        self,
        sdf_file_list: List[str],  # noqa: UP006
        dataset_mode: Literal["molbind", "encoder"],
        output_list: Optional[List[float]] = None,  # noqa: UP006
        **kwargs,
    ) -> None:
        from molbind.data.available import (
            MODALITY_DATA_TYPES,
            NonStringModalities,
            StringModalities,
        )

        self.sdf_file_list = sdf_file_list
        self.mode = dataset_mode
        if dataset_mode == "encoder":
            self.energies_list = output_list
        elif dataset_mode == "molbind":
            self.energies_list = [0.0] * len(sdf_file_list)
            self.central_modality = kwargs["central_modality"]
            self.other_modality = "structure"
            self.central_modality_data = kwargs["central_modality_data"]
            self.central_modality_data_type = MODALITY_DATA_TYPES[self.central_modality]
        # modality handler functions if a modality is the central modality
        self.central_modality_handlers = {
            StringModalities.SMILES: _string,
            StringModalities.SELFIES: _string,
            StringModalities.IUPAC_NAME: _string,
            StringModalities.IR: _string,
            StringModalities.NMR: _string,
            StringModalities.MASS: _string,
            NonStringModalities.FINGERPRINT: _fingerprint,
        }

    def __len__(self) -> float:
        return len(self.sdf_file_list)

    def __getitem__(self, idx: int) -> Data:
        data = get_item_for_dimenet(sdf_file=self.sdf_file_list[idx], i=idx)
        if self.mode == "encoder":
            return data
        elif self.mode == "molbind":
            data.central_modality = self.central_modality
            if self.central_modality_data_type != str:
                data.central_modality_data = self.central_modality_handlers[
                    data.central_modality
                ](self.central_modality_data[idx])
            else:
                data.input_ids = self.central_modality_handlers[data.central_modality](
                    self.central_modality_data[0][idx]
                )
                data.attention_mask = self.central_modality_handlers[
                    data.central_modality
                ](self.central_modality_data[1][idx])
            data.modality = self.other_modality
            return data
        raise ValueError(
            f"'{self.mode}' is an invalid mode. Accepted values are 'encoder' and 'molbind'"
        )


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
