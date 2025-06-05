from functools import partial

import pandas as pd
import polars as pl
from torch import Tensor
from torch.utils.data import Dataset

from molbind.data.available import (
    ModalityConstants,
    NonStringModalities,
    StringModalities,
)
from molbind.data.components.datasets import (
    FingerprintMolBindDataset,
    HSQCDataset,
    IrDataset,
    MassSpecNegativeDataset,
    MassSpecPositiveDataset,
    StringDataset,
    cNmrDataset,
    hNmrDataset,
)


class MolBindDataset:
    def __init__(
        self,
        data: pd.DataFrame,
        central_modality: StringModalities | NonStringModalities,
        other_modalities: list[str],
        config: dict | None = None,
        **kwargs,
    ) -> None:
        """Dataset for multimodal data."""
        self.data = data
        self.central_modality = central_modality
        self.central_modality_data_type = ModalityConstants[central_modality].data_type
        self.config = config
        # if self.central_modality_data_type == str:
        init_str_fn = partial(
            self._tokenize_strings,
            context_length=kwargs.get("context_length", 256),
            modality=central_modality,
        )
        self.central_modality_handlers = {
            StringModalities.SMILES: init_str_fn,
        }

        self.dataset_builders = {
            StringModalities.SMILES: partial(
                self.build_string_dataset,
                modality=StringModalities.SMILES,
                context_length=kwargs.get("context_length", 256),
            ),
            NonStringModalities.C_NMR: self.build_c_nmr_dataset,
            NonStringModalities.IR: self.build_ir_dataset,
            # NonStringModalities.MASS_SPEC_POSITIVE: self.build_mass_spec_positive_dataset,
            # NonStringModalities.MASS_SPEC_NEGATIVE: self.build_mass_spec_negative_dataset,
            NonStringModalities.H_NMR: self.build_hnmr_cnn_dataset,
            NonStringModalities.HSQC: self.build_hsqc_dataset,
        }
        self.data = data.reset_index(drop=True)
        # central modality data
        self.central_modality_data = self.central_modality_handlers[central_modality](self.data[central_modality].to_list())

        self.other_modalities = other_modalities

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

    def build_hnmr_cnn_dataset(self) -> hNmrDataset:
        modality = "h_nmr"
        h_nmr_cnn_data = self.data[[self.central_modality, modality]].dropna()
        return hNmrDataset(
            data=h_nmr_cnn_data[modality].to_list(),
            augment=self.config.data.h_nmr.augment if self.config else False,
            vec_size=self.config.data.h_nmr.vec_size if hasattr(self.config.data.h_nmr, "vec_size") else 10_000,
            central_modality=self.central_modality,
            central_modality_data=self._handle_central_modality_data(h_nmr_cnn_data),
        )

    def build_hsqc_dataset(self) -> HSQCDataset:
        modality = "hsqc"
        hsqc_data = self.data[[self.central_modality, modality]].dropna()
        return HSQCDataset(
            data=hsqc_data[modality].to_list(),
            central_modality=self.central_modality,
            central_modality_data=self._handle_central_modality_data(hsqc_data),
        )

    def build_datasets_for_modalities(self) -> dict[str, Dataset]:
        datasets = {}
        for modality in self.other_modalities:
            if modality in self.data.columns:
                dataset = self.dataset_builders[modality]()
                datasets[modality] = dataset
        # CombinedLoader does not work with DDPSampler directly
        # Thus this ^ is added to the dataloaders in the datamodule
        return datasets

    def _handle_central_modality_data(self, data_pair: pd.DataFrame) -> tuple[Tensor, Tensor]:
        return (
            self.central_modality_data[0][data_pair.index.to_list()],
            self.central_modality_data[1][data_pair.index.to_list()],
        )

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
