from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from molbind.data.components.cnmr import augment_13c
from molbind.data.components.hnmr import augment
from molbind.utils import generate_hsqc_matrix
from molbind.utils.spec2struct import reduce_resolution_by_averaging


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
            if self.central_modality_data_type is str
            else Tensor(self.central_modality_data[idx]),
            self.other_modality: tuple([i[idx] for i in self.other_modality_data])
            if self.other_modality_data_type is str
            else Tensor(self.other_modality_data)[idx],
        }


class ImageDataset(Dataset):
    def __init__(
        self,
        data: list[str],
        central_modality: str,
        central_modality_data: tuple[Tensor, ...],
        augment: bool = False,
        input_size: int = 384,
        image_render_size: int = 384,
        **kwargs,
    ) -> None:
        """Dataset for the molecule-image modality.

        Images are rendered on the fly from SMILES with RDKit and preprocessed
        with MolScribe's transform, yielding a ``[3, input_size, input_size]``
        tensor consumable by :class:`MolScribeImageEncoder`.

        Args:
            data: list of SMILES strings to depict (the "image" column).
            central_modality: name of the central modality.
            central_modality_data: tokenized central-modality tensors.
            augment: if ``True`` apply MolScribe's train-time depiction
                augmentation (random rotation, crops, blur, noise). Use only for
                training; keep ``False`` for reproducible val/eval embeddings.
            input_size: side length of the preprocessed tensor (MolScribe: 384).
            image_render_size: side length of the RDKit-rendered image.
        """

        from molscribe.dataset import get_transforms

        self.smiles = data
        self.central_modality = central_modality
        self.other_modality = "image"
        self.central_modality_data = central_modality_data
        self.image_render_size = image_render_size
        self.transform = get_transforms(input_size, augment=augment, rotate=augment)

    def __len__(self) -> int:
        return len(self.smiles)

    def _render(self, smiles: str) -> np.ndarray:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        size = self.image_render_size
        if mol is None:
            # Unparseable SMILES -> blank white image; CropWhite/Resize handle it.
            return np.full((size, size, 3), 255, dtype=np.uint8)
        drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        image = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")
        return np.array(image)

    def __getitem__(self, idx: int) -> dict:
        image = self._render(self.smiles[idx])
        image_tensor = self.transform(image=image, keypoints=[])["image"]
        return {
            self.central_modality: [i[idx] for i in self.central_modality_data],
            self.other_modality: image_tensor,
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
        vec_len: int = 2048,
        min_value: float = -5,
        max_value: float = 230,
        augment: bool = False,
        **kwargs,
    ) -> None:
        self.c_nmr = data
        self.vec_len = vec_len
        self.min_value = min_value
        self.max_value = max_value
        self.central_modality = kwargs["central_modality"]
        self.other_modality = "c_nmr"
        self.central_modality_data = kwargs["central_modality_data"]
        self.augment = augment

    def __len__(self):
        return len(self.c_nmr)

    def __getitem__(self, index: int) -> dict:
        return {
            self.central_modality: [i[index] for i in self.central_modality_data],
            self.other_modality: self.c_nmr_to_vec(self.c_nmr[index]),
        }

    def c_nmr_to_vec(self, nmr_shifts: list[float]) -> Tensor:
        nmr_array = np.array(nmr_shifts) / np.max(nmr_shifts)
        return torch.tensor(nmr_array, dtype=torch.float32).unsqueeze(0)


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


class GeneralDataset(Dataset):
    def __init__(
        self,
        data: list[list[float]],
        **kwargs,
    ) -> None:
        self.general = data
        # self.min_value = min_value
        # self.max_value = max_value
        self.central_modality = kwargs["central_modality"]
        self.other_modality = "general"
        self.central_modality_data = kwargs["central_modality_data"]
        self.pad_length = 10000

    def __len__(self):
        return len(self.general)

    # def __getitem__(self, index: int) -> dict:
    #     general = torch.tensor(self.general[index], dtype=torch.float32)
    #     # pad (or truncate) to self.pad_length
    #     if general.size(0) < self.pad_length:
    #         general = torch.nn.functional.pad(general, (0, self.pad_length - general.size(0)))
    #     else:
    #         general = general[:self.pad_length]
    #     general = (general - general.min()) / (general.max() - general.min())
    #     general = general.unsqueeze(0)

    #     return {
    #         self.central_modality: [g[index] for g in self.central_modality_data],
    #         self.other_modality: general,
    #     }
    def __getitem__(self, index: int) -> dict:
        general = torch.tensor(self.general[index], dtype=torch.float32)
        # interpolate to self.pad_length points
        general = general.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        general = torch.nn.functional.interpolate(
            general, size=self.pad_length, mode="linear", align_corners=False
        ).squeeze(0).squeeze(0)  # (pad_length,)
        general = (general - general.min()) / (general.max() - general.min())
        general = general.unsqueeze(0)

        return {
            self.central_modality: [g[index] for g in self.central_modality_data],
            self.other_modality: general,
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
