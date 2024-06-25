from functools import partial  # noqa: I002
from typing import Dict, List, Tuple, Union  # noqa: UP035

import pandas as pd
import polars as pl
import selfies as sf
from torch import Tensor
from torch.utils.data import Dataset

from molbind.data.available import (
    ModalityConstants,
    NonStringModalities,
    StringModalities,
)
from molbind.data.components.datasets import (
    FingerprintMolBindDataset,
    GraphDataset,
    ImageDataset,
    StringDataset,
    StructureDataset,
)


class MolBindDataset:
    def __init__(
        self,
        data: pl.DataFrame,
        central_modality: Union[StringModalities, NonStringModalities],
        other_modalities: List[str],  # noqa: UP006
        **kwargs,
    ) -> None:
        """Dataset for multimodal data."""
        self.data = data
        self.central_modality = central_modality
        self.central_modality_data_type = ModalityConstants[central_modality].data_type

        # if self.central_modality_data_type == str:
        init_str_fn = partial(
            self._tokenize_strings,
            context_length=kwargs.get("context_length", 256),
            modality=central_modality,
        )
        self.central_modality_handlers = {
            StringModalities.SMILES: init_str_fn,
            StringModalities.SELFIES: init_str_fn,
            StringModalities.IUPAC_NAME: init_str_fn,
            StringModalities.DESCRIPTION: init_str_fn,
            NonStringModalities.STRUCTURE: lambda x: x,
            NonStringModalities.GRAPH: lambda x: x,
            NonStringModalities.FINGERPRINT: lambda x: x,
            NonStringModalities.IMAGE: lambda x: x,
        }

        self.dataset_builders = {
            StringModalities.SMILES: partial(
                self.build_string_dataset,
                modality=StringModalities.SMILES,
                context_length=kwargs.get("context_length", 256),
            ),
            StringModalities.SELFIES: partial(
                self.build_string_dataset,
                modality=StringModalities.SELFIES,
                context_length=kwargs.get("context_length", 256),
            ),
            StringModalities.IUPAC_NAME: partial(
                self.build_string_dataset,
                modality=StringModalities.IUPAC_NAME,
                context_length=kwargs.get("context_length", 256),
            ),
            StringModalities.DESCRIPTION: partial(
                self.build_string_dataset,
                modality=StringModalities.DESCRIPTION,
                context_length=kwargs.get("context_length", 256),
            ),
            NonStringModalities.GRAPH: self.build_graph_dataset,
            NonStringModalities.STRUCTURE: self.build_3D_coordinates_dataset,
            NonStringModalities.FINGERPRINT: self.build_fp_dataset,
            NonStringModalities.IMAGE: self.build_image_dataset,
        }
        # central modality data
        self.central_modality_data = self.central_modality_handlers[central_modality](
            self.data[central_modality].to_list()
        )
        self.data = data.to_pandas().reset_index()
        self.other_modalities = other_modalities

    def _handle_central_modality_data(
        self, data_pair: pd.DataFrame
    ) -> Tuple[Tensor, Tensor]:  # noqa: UP006
        if self.central_modality_data_type == str:
            central_modality_data = (
                self.central_modality_data[0][data_pair.index.to_list()],
                self.central_modality_data[1][data_pair.index.to_list()],
            )
        return central_modality_data

    def build_graph_dataset(self) -> GraphDataset:
        modality = "graph"
        graph_data = self.data[[self.central_modality, modality]].dropna()
        # perform graph operations
        # add graph dataset logic here
        return GraphDataset(
            graph_data=graph_data,
            central_modality=self.central_modality,
            central_modality_data=self._handle_central_modality_data(graph_data),
        )

    def build_3D_coordinates_dataset(self) -> StructureDataset:
        modality = "structure"
        struc_data = self.data[[self.central_modality, modality]].dropna()
        return StructureDataset(
            sdf_file_list=struc_data[modality].to_list(),
            dataset_mode="molbind",
            central_modality=self.central_modality,
            central_modality_data=self._handle_central_modality_data(struc_data),
        )

    def build_string_dataset(
        self, modality: str, context_length: int = 256
    ) -> StringDataset:
        string_data = self.data[[self.central_modality, modality]].dropna()
        other_modality_data = self._tokenize_strings(
            string_data[modality].to_list(),
            context_length=context_length,
            modality=modality,
        )
        return StringDataset(
            central_modality=self.central_modality,
            other_modality=modality,
            central_modality_data=self._handle_central_modality_data(string_data),
            other_modality_data=other_modality_data,
        )

    def build_fp_dataset(self) -> FingerprintMolBindDataset:
        modality = "fingerprint"
        fp_data = self.data[[self.central_modality, modality]].dropna()
        # perform fingerprint operations
        return FingerprintMolBindDataset(
            central_modality=self.central_modality,
            fingerprint_data=fp_data["fingerprint"].to_list(),
            central_modality_data=self._handle_central_modality_data(fp_data),
        )

    def build_image_dataset(self) -> ImageDataset:
        modality = "image"
        image_data = self.data[[self.central_modality, modality]].dropna()
        # perform image operations
        return ImageDataset(
            image_data=image_data["image"].to_list(),
            central_modality=self.central_modality,
            central_modality_data=self._handle_central_modality_data(image_data),
        )

    def build_datasets_for_modalities(
        self,
    ) -> Dict[str, Dataset]:  # noqa: UP006
        datasets = {}
        for modality in self.other_modalities:
            if modality in self.data.columns:
                dataset = self.dataset_builders[modality]()
                datasets[modality] = dataset
        # CombinedLoader does not work with DDPSampler directly
        # Thus this ^ is added to the dataloaders in the datamodule
        return datasets

    # private class methods
    @staticmethod
    def _tokenize_strings(
        dataset: List[str],  # noqa: UP006
        context_length: int,
        modality: str,
    ) -> Tuple[Tensor, Tensor]:  # noqa: UP006
        tokenized_data = ModalityConstants[modality].tokenizer(
            dataset,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=context_length,
        )
        return tokenized_data["input_ids"], tokenized_data["attention_mask"]

    @staticmethod
    def _build_selfies_from_smiles(smi_list: List[str]) -> List[str]:  # noqa: UP006
        return [sf.encoder(smi) for smi in smi_list]

    @staticmethod
    def _build_smiles_from_selfies(selfies_list: List[str]) -> List[str]:  # noqa: UP006
        return [sf.decoder(selfies) for selfies in selfies_list]
