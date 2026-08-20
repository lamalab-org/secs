from io import BytesIO

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from secs.data.components.hnmr import augment
from secs.utils import generate_hsqc_matrix
from secs.utils.elucidation import reduce_resolution_by_averaging


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
        from secs.data.modalities import ModalityConstants

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
            if self.central_modality_data_type is str
            else Tensor(self.central_modality_data[idx]),
            self.other_modality: tuple([i[idx] for i in self.other_modality_data])
            if self.other_modality_data_type is str
            else Tensor(self.other_modality_data)[idx],
        }


class FingerprintSECSDataset(Dataset):
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


class cNmrDataset(Dataset):
    """Peak-list 13C dataset that PRECOMPUTES padded (shifts, mask) in __init__.

    Every sample is padded to a fixed `max_peaks` length up front, so the
    default DataLoader collate can stack them — NO custom collate_fn needed.
    Absolute ppm is preserved (no per-sample normalization).

    Memory note: stores an (N, max_peaks) float tensor + (N, max_peaks) bool
    mask for the whole dataset. At N=400k, max_peaks=128 that's ~256 MB.
    """

    def __init__(
        self,
        data: list[list[float]],
        max_peaks: int = 128,
        min_value: float = -5.0,
        max_value: float = 230.0,
        augment: bool = False,
        **kwargs,
    ) -> None:
        self.max_peaks = max_peaks
        self.min_value = min_value
        self.max_value = max_value
        self.central_modality = kwargs["central_modality"]
        self.other_modality = "c_nmr"
        self.augment = augment

        central_data = kwargs["central_modality_data"]

        # --- clean each sample to its in-range peaks, drop empties ---
        cleaned, keep = [], []
        for i, s in enumerate(data):
            arr = np.asarray(s, dtype=np.float32)
            if arr.size:
                arr = arr[(arr >= min_value) & (arr <= max_value)]
            if arr.size:  # keep only non-empty (all-padding row -> attention NaN)
                cleaned.append(arr[:max_peaks])
                keep.append(i)

        # filter central modality in lockstep so indices stay aligned
        self.central_modality_data = [[col[i] for i in keep] for col in central_data]

        # --- precompute padded tensors once ---
        N = len(cleaned)
        self.shifts = torch.zeros(N, max_peaks, dtype=torch.float32)
        self.mask = torch.zeros(N, max_peaks, dtype=torch.bool)
        for i, arr in enumerate(cleaned):
            n = arr.shape[0]
            self.shifts[i, :n] = torch.from_numpy(arr)
            self.mask[i, :n] = True

    def __len__(self) -> int:
        return self.shifts.shape[0]

    def __getitem__(self, index: int) -> dict:
        return {
            self.central_modality: [col[index] for col in self.central_modality_data],
            "c_nmr": (self.shifts[index], self.mask[index]),
        }


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
        augment: bool = False,
        vec_size: int = 10_000,
        **kwargs,
    ) -> None:
        self.h_nmr = data
        # For multi-modal setups, if provided
        self.central_modality = kwargs.get("central_modality")
        self.other_modality = "h_nmr"
        self.central_modality_data = kwargs.get("central_modality_data")
        self.augment = bool(augment)
        self.vec_size = vec_size

    def __len__(self):
        return len(self.h_nmr)

    def __getitem__(self, index: int) -> dict:
        return {
            self.central_modality: [i[index] for i in self.central_modality_data],
            self.other_modality: self.hnmr_to_vec(self.h_nmr[index]),
        }

    def hnmr_to_vec(self, nmr_shifts: list[list[float]]) -> Tensor:
        nmr_array = np.array(nmr_shifts) / np.max(nmr_shifts)
        if self.augment:
            resolutions_available = [500, 1000, 2000, 3000, 5000, 10000]
            self.vec_size = np.random.choice(resolutions_available, p=[0.05, 0.15, 0.2, 0.2, 0.2, 0.2])
            augment_prob = np.random.rand()
            if augment_prob > 0.1:
                nmr_array = augment(nmr_array)
                # resolution to 2000 (but still in a vector of 10_000)
                nmr_array = reduce_resolution_by_averaging(nmr_array, window_size=int(10_000 / self.vec_size))
        else:
            # just add random noise
            noise = np.random.normal(0, 0.01, nmr_array.shape)
            # nmr_array = nmr_array + noise
            # nmr_array = reduce_resolution_by_averaging(nmr_array, window_size=int(10_000 / self.vec_size))
        # nmr_array = np.cumsum(nmr_array, axis=0)
        nmr_array = nmr_array / np.max(nmr_array)
        return torch.tensor(
            nmr_array,
            dtype=torch.float32,
        ).unsqueeze(0)


class StringDatasetEmbedding(Dataset):
    def __init__(
        self,
        data: list[list[int]],
    ) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx][0]),
            torch.tensor(self.data[idx][1]),
        )


class HSQCDataset(Dataset):
    def __init__(
        self,
        data: list[list[float]],
        **kwargs,
    ) -> None:
        self.hsqc = data
        self.central_modality = kwargs["central_modality"]
        self.other_modality = "hsqc"
        self.central_modality_data = kwargs["central_modality_data"]

    def __len__(self):
        return len(self.hsqc)

    def __getitem__(self, index: int) -> dict:
        # input shape (512, 512) with 1 channel
        image = generate_hsqc_matrix(self.hsqc[index])
        return {
            self.central_modality: [i[index] for i in self.central_modality_data],
            self.other_modality: torch.tensor(image, dtype=torch.float32).unsqueeze(0),
        }
