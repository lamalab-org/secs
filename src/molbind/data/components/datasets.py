import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class StringDataset(Dataset):
    def __init__(
        self,
        central_modality_data: tuple[Tensor, Tensor],
        other_modality_data: tuple[Tensor, Tensor],
        central_modality: str,
        other_modality: str,
    ) -> None:
        """Dataset for string modalities.

        Args:
            central_modality_data (tuple[Tensor, Tensor]): pair of (central_modality, tokenized_central_modality)
            other_modality_data (tuple[Tensor, Tensor]): pair of (other_modality, tokenized_other_modality)
            central_modality (str): name of central modality as found in ModalityConstants
            other_modality (str): name of other modality as found in ModalityConstants
        """
        from molbind.data.available import ModalityConstants

        # modality pair definition
        self.central_modality = central_modality
        self.other_modality = other_modality
        # modality pair data
        self.central_modality_data = central_modality_data
        self.other_modality_data = other_modality_data
        self.central_modality_data_type = ModalityConstants[central_modality].data_type
        self.other_modality_data_type = ModalityConstants[other_modality].data_type

    def __len__(self):
        return len(self.other_modality_data[0])

    def __getitem__(self, idx):
        return {
            self.central_modality: tuple([i[idx] for i in self.central_modality_data])
            if isinstance(self.central_modality_data_type, str)
            else Tensor(self.central_modality_data[idx]),
            self.other_modality: tuple([i[idx] for i in self.other_modality_data])
            if isinstance(self.other_modality_data_type, str)
            else Tensor(self.other_modality_data)[idx],
        }


class FingerprintMolBindDataset(Dataset):
    def __init__(
        self,
        central_modality_data: tuple[Tensor, Tensor],
        fingerprint_data: list[list[int]],
        central_modality: str,
    ) -> None:
        """Dataset for fingerprints.

        Args:
            central_modality_data (tuple[Tensor, Tensor]): pair of (central_modality, tokenized_central_modality)
            fingerprint_data (Tensor): fingerprint data
            central_modality (str): name of central modality as found in ModalityConstants
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


class cNmrDataset(Dataset):
    def __init__(
        self,
        data: list[list[float]],
        vec_len: int = 512,
        min_value: float = 0,
        max_value: float = 300,
        **kwargs,
    ) -> None:
        self.c_nmr = data
        self.vec_len = vec_len
        self.min_value = min_value
        self.max_value = max_value
        self.central_modality = kwargs["central_modality"]
        self.other_modality = "c_nmr"
        self.central_modality_data = kwargs["central_modality_data"]

    def __len__(self):
        return len(self.c_nmr)

    def __getitem__(self, index: int) -> dict:
        return {
            self.central_modality: [i[index] for i in self.central_modality_data],
            self.other_modality: self.c_nmr_to_vec(self.c_nmr[index]),
        }

    def c_nmr_to_vec(self, nmr_shifts: list[float]) -> Tensor:
        init_vec = torch.zeros(self.vec_len, dtype=torch.float32)
        for shift in nmr_shifts:
            index = int(shift / self.max_value * self.vec_len)
            init_vec[index] = 1
        return init_vec


class IrDataset(Dataset):
    def __init__(
        self,
        data: list[list[float]],
        **kwargs,
    ) -> None:
        self.ir = data
        # self.min_value = min_value
        # self.max_value = max_value
        self.central_modality = kwargs["central_modality"]
        self.other_modality = "ir"
        self.central_modality_data = kwargs["central_modality_data"]

    def __len__(self):
        return len(self.ir)

    def __getitem__(self, index: int) -> dict:
        # convert to tensor
        ir = torch.tensor(self.ir[index], dtype=torch.float32)[100:1700].unsqueeze(0)
        return {
            self.central_modality: [i[index] for i in self.central_modality_data],
            self.other_modality: ir,
        }


class MassSpecDataset(Dataset):
    def __init__(
        self,
        data: list[list[float]],
        vec_len: int = 1024,
        max_value: float = 1000,
        **kwargs,
    ) -> None:
        self.mass_spec = data
        self.vec_len = vec_len
        self.max_value = max_value
        self.central_modality = kwargs["central_modality"]
        self.other_modality = "mass_spec"
        self.central_modality_data = kwargs["central_modality_data"]

    def __len__(self):
        return len(self.mass_spec)

    def __getitem__(self, index: int) -> dict:
        return {
            self.central_modality: [i[index] for i in self.central_modality_data],
            self.other_modality: self.mass_to_spec(self.mass_spec[index]),
        }

    def mass_to_spec(self, mass_spec: list[list[float, float]]) -> Tensor:
        """
        list[list[mass, intensity]]
        """
        init_vec = torch.zeros(self.vec_len, dtype=torch.float32)
        for mass, intensity in mass_spec:
            index = int(mass / self.max_value * self.vec_len)
            init_vec[index] = intensity
        return init_vec


class MassSpecPositiveDataset(MassSpecDataset):
    def __init__(
        self,
        data: list[list[float]],
        vec_len: int = 1024,
        max_value: float = 1000,
        **kwargs,
    ) -> None:
        super().__init__(data, vec_len, max_value, **kwargs)
        self.other_modality = "mass_spec_positive"


class MassSpecNegativeDataset(MassSpecDataset):
    def __init__(
        self,
        data: list[list[float]],
        vec_len: int = 1024,
        max_value: float = 1000,
        **kwargs,
    ) -> None:
        super().__init__(data, vec_len, max_value, **kwargs)
        self.other_modality = "mass_spec_negative"


class hNmrDataset(Dataset):
    def __init__(
        self,
        data: list[list[float]],
        vec_len: int = 512,
        architecture: str = "cnn",
        **kwargs,
    ) -> None:
        self.h_nmr = data
        self.vec_len = vec_len
        self.central_modality = kwargs["central_modality"]
        self.other_modality = "h_nmr"
        self.central_modality_data = kwargs["central_modality_data"]
        self.architecture = architecture

    def __len__(self):
        return len(self.h_nmr)

    def __getitem__(self, index: int) -> dict:
        return {
            self.central_modality: [i[index] for i in self.central_modality_data],
            self.other_modality: self.hnmr_to_vec(self.h_nmr[index]),
        }

    def hnmr_to_vec(self, nmr_shifts: list[list[float]]) -> Tensor:
        if len(nmr_shifts) == 10000:
            # normalize the data
            nmr_shifts = nmr_shifts / np.max(nmr_shifts)
            if self.architecture == "cnn":
                # add 1 channel if using CNN
                return torch.tensor(nmr_shifts, dtype=torch.float32).unsqueeze(0)
            return torch.tensor(nmr_shifts, dtype=torch.float32)
        init_vec = torch.zeros(self.vec_len, dtype=torch.float32)
        if isinstance(nmr_shifts[0], list | np.ndarray):
            for shift, _ in nmr_shifts:
                index = int(shift / 18 * self.vec_len)
                init_vec[index] = 1
        else:
            for _shift in nmr_shifts:
                shift = np.round(_shift, 2)
                index = int(shift / 18 * self.vec_len)
                if index >= self.vec_len:
                    index = self.vec_len - 1
                if index < 0:
                    index = 0
                if init_vec[index] != 0:
                    index = index + 1
                elif init_vec[index] == 0:
                    init_vec[index] = 1
        return init_vec


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
