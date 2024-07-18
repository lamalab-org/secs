import random  # noqa: I002
from typing import List, Literal, Optional, Tuple, Union  # noqa: UP035

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torchvision import transforms

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
            central_modality (str): name of central modality as found in ModalityConstants
            central_modality_data (Union[Tensor, Tuple[Tensor, Tensor]]): central modality data
            that is either a tensor or a tuple of tensors depending on the data type
        Returns:
            None
        """

        super().__init__()
        from molbind.data.available import (
            ModalityConstants,
            NonStringModalities,
            StringModalities,
        )

        self.central_modality = central_modality
        self.modality = "graph"
        self.smiles_list = graph_data[self.modality].to_list()
        self.central_modality_data = central_modality_data
        self.central_modality_data_type = ModalityConstants[central_modality].data_type
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
            ModalityConstants,
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
            self.central_modality_data_type = ModalityConstants[
                self.central_modality
            ].data_type
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


class ImageDataset(Dataset):
    def __init__(self, image_files: List[str], **kwargs):  # noqa: UP006
        """Dataset for images.

        Args:
            image_files (List[str]): list of image file paths
            labels (List[int]): list of SMILES labels
        """
        from molbind.data.available import (
            NonStringModalities,
        )

        self.image_files = image_files
        self.central_modality = kwargs["central_modality"]
        self.other_modality = NonStringModalities.IMAGE
        self.central_modality_data = kwargs["central_modality_data"]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        image_as_tensor = self.read_image_to_tensor(self.image_files[idx], repeats=1)
        return {
            self.central_modality: [i[idx] for i in self.central_modality_data],
            self.other_modality: image_as_tensor.mean(dim=0),
        }

    @classmethod
    def read_imagefile(cls, filepath: str) -> Image.Image:
        img = Image.open(filepath, "r")

        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            # Paste image to background image
            bg.paste(img, (0, 0), img)
            return bg.convert("L")
        else:
            return img.convert("L")

    @classmethod
    def fit_image(cls, img: Image):
        old_size = img.size
        desired_size = 224
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, Image.BICUBIC)
        new_img = Image.new("L", (desired_size, desired_size), "white")
        new_img.paste(
            img, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
        )
        return ImageOps.expand(new_img, int(np.random.randint(5, 25, size=1)), "white")  # noqa: NPY002

    @classmethod
    def transform_image(cls, image: Image):
        image = cls.fit_image(image)
        img_PIL = transforms.RandomRotation(
            (-15, 15), interpolation=3, expand=True, center=None, fill=255
        )(image)
        img_PIL = transforms.ColorJitter(
            brightness=[0.75, 2.0], contrast=0, saturation=0, hue=0
        )(img_PIL)
        shear_value = np.random.uniform(0.1, 7.0)  # noqa: NPY002
        shear = random.choice(
            [
                [0, 0, -shear_value, shear_value],
                [-shear_value, shear_value, 0, 0],
                [-shear_value, shear_value, -shear_value, shear_value],
            ]
        )
        img_PIL = transforms.RandomAffine(
            0, translate=None, scale=None, shear=shear, interpolation=3, fill=255
        )(img_PIL)
        img_PIL = ImageEnhance.Contrast(ImageOps.autocontrast(img_PIL)).enhance(2.0)
        img_PIL = transforms.Resize((224, 224), interpolation=3)(img_PIL)
        img_PIL = ImageOps.autocontrast(img_PIL)
        return transforms.ToTensor()(img_PIL)

    def read_image_to_tensor(self, filepath: str, repeats: int = 50):
        extension = filepath.split(".")[-1] in ("jpg", "jpeg", "png")
        if not extension:
            return "Image must be jpg or png format!"
        image = self.read_imagefile(filepath)
        return torch.cat(
            [torch.unsqueeze(self.transform_image(image), 0) for _ in range(repeats)],
            dim=0,
        )


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
