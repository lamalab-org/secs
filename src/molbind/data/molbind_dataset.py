from functools import partial

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
    IrDataset,
    MassSpecNegativeDataset,
    MassSpecPositiveDataset,
    MultiSpecDataset,
    StringDataset,
    StructureDataset,
    cNmrDataset,
    hNmrDataset,
)


class MolBindDataset:
    def __init__(
        self,
        data: pl.DataFrame,
        central_modality: StringModalities | NonStringModalities,
        other_modalities: list[str],
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
            NonStringModalities.C_NMR: self.build_c_nmr_dataset,
            NonStringModalities.H_NMR: self.build_h_nmr_dataset,
            NonStringModalities.IR: self.build_ir_dataset,
            NonStringModalities.MASS_SPEC_POSITIVE: self.build_mass_spec_positive_dataset,
            NonStringModalities.MASS_SPEC_NEGATIVE: self.build_mass_spec_negative_dataset,
            NonStringModalities.MULTI_SPEC: self.build_multi_spec_dataset,
        }
        self.data = data.reset_index(drop=True)
        # central modality data
        self.central_modality_data = self.central_modality_handlers[central_modality](self.data[central_modality].to_list())

        self.other_modalities = other_modalities

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

    def build_string_dataset(self, modality: str, context_length: int = 256) -> StringDataset:
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
            fingerprint_data=fp_data[modality].to_list(),
            central_modality_data=self._handle_central_modality_data(fp_data),
        )

    def build_image_dataset(self) -> ImageDataset:
        modality = "image"
        image_data = self.data[[self.central_modality, modality]].dropna()
        # perform image operations
        return ImageDataset(
            image_files=image_data[modality].to_list(),
            central_modality=self.central_modality,
            central_modality_data=self._handle_central_modality_data(image_data),
        )

    def build_c_nmr_dataset(self) -> cNmrDataset:
        modality = "c_nmr"
        c_nmr_data = self.data[[self.central_modality, modality]].dropna()
        return cNmrDataset(
            data=c_nmr_data[modality].to_list(),
            central_modality=self.central_modality,
            central_modality_data=self._handle_central_modality_data(c_nmr_data),
        )

    def build_ir_dataset(self) -> IrDataset:
        modality = "ir"
        ir_data = self.data[[self.central_modality, modality]].dropna()
        return IrDataset(
            data=ir_data[modality].to_list(),
            central_modality=self.central_modality,
            central_modality_data=self._handle_central_modality_data(ir_data),
        )

    def build_mass_spec_positive_dataset(self) -> MassSpecPositiveDataset:
        modality = "mass_spec_positive"
        mass_spec_data = self.data[[self.central_modality, modality]].dropna()
        return MassSpecPositiveDataset(
            data=mass_spec_data[modality].to_list(),
            central_modality=self.central_modality,
            central_modality_data=self._handle_central_modality_data(mass_spec_data),
        )

    def build_mass_spec_negative_dataset(self) -> MassSpecNegativeDataset:
        modality = "mass_spec_negative"
        mass_spec_data = self.data[[self.central_modality, modality]].dropna()
        return MassSpecNegativeDataset(
            data=mass_spec_data[modality].to_list(),
            central_modality=self.central_modality,
            central_modality_data=self._handle_central_modality_data(mass_spec_data),
        )

    def build_h_nmr_dataset(self) -> hNmrDataset:
        modality = "h_nmr"
        h_nmr_data = self.data[[self.central_modality, modality]].dropna()
        return hNmrDataset(
            data=h_nmr_data[modality].to_list(),
            central_modality=self.central_modality,
            central_modality_data=self._handle_central_modality_data(h_nmr_data),
        )

    def build_multi_spec_dataset(self) -> MultiSpecDataset:
        modality = "multi_spec"
        multi_spec_data = self.data[[self.central_modality, modality]].dropna()
        return MultiSpecDataset(
            data=multi_spec_data[modality].to_list(),
            central_modality=self.central_modality,
            central_modality_data=self._handle_central_modality_data(multi_spec_data),
        )

    def build_datasets_for_modalities(
        self,
    ) -> dict[str, Dataset]:
        datasets = {}
        for modality in self.other_modalities:
            if modality in self.data.columns:
                dataset = self.dataset_builders[modality]()
                datasets[modality] = dataset
        # CombinedLoader does not work with DDPSampler directly
        # Thus this ^ is added to the dataloaders in the datamodule
        return datasets

    def _handle_central_modality_data(self, data_pair: pd.DataFrame) -> tuple[Tensor, Tensor]:
        if self.central_modality_data_type is str:
            central_modality_data = (
                self.central_modality_data[0][data_pair.index.to_list()],
                self.central_modality_data[1][data_pair.index.to_list()],
            )
        return central_modality_data

    @staticmethod
    def _tokenize_strings(
        dataset: list[str],
        context_length: int,
        modality: str,
    ) -> tuple[Tensor, Tensor]:
        tokenized_data = ModalityConstants[modality].tokenizer(
            dataset,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=context_length,
        )
        return tokenized_data["input_ids"], tokenized_data["attention_mask"]

    @staticmethod
    def _build_selfies_from_smiles(smi_list: list[str]) -> list[str]:
        return [sf.encoder(smi) for smi in smi_list]

    @staticmethod
    def _build_smiles_from_selfies(selfies_list: list[str]) -> list[str]:
        return [sf.decoder(selfies) for selfies in selfies_list]
