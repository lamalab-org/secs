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
    GraphDataset,
    StringDataset,
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
        self.custom_negatives = kwargs.get("custom_negatives", False)
        if self.custom_negatives:
            self.custom_negatives_samples = data["custom_negatives"].to_list()
        else:
            self.custom_negatives_samples = None
        # if self.central_modality_data_type == str:
        init_str_fn = partial(
            self._tokenize_strings,
            context_length=kwargs.get("context_length", 256),
            modality=central_modality,
        )
        self.central_modality_handlers = {
            StringModalities.PSMILES: init_str_fn,
            StringModalities.BIGSMILES: init_str_fn,
            StringModalities.SMILES: init_str_fn,
            StringModalities.SELFIES: init_str_fn,
            StringModalities.IUPAC_NAME: init_str_fn,
            NonStringModalities.GRAPH: lambda x: x,
        }

        self.dataset_builders = {
            StringModalities.SMILES: partial(
                self.build_string_dataset,
                modality=StringModalities.SMILES,
                context_length=kwargs.get("context_length", 256),
            ),
            StringModalities.BIGSMILES: partial(
                self.build_string_dataset,
                modality=StringModalities.BIGSMILES,
                context_length=kwargs.get("context_length", 256),
            ),
            StringModalities.PSMILES: partial(
                self.build_string_dataset,
                modality=StringModalities.PSMILES,
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
            NonStringModalities.GRAPH: self.build_graph_dataset,
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
        if self.central_modality_data_type is str and self.custom_negatives:
            central_modality_data = (
                self.central_modality_data[0][data_pair.index.to_list()],
                self.central_modality_data[1][data_pair.index.to_list()],
                [
                    self._tokenize_strings(list(self.custom_negatives_samples[i]), 128, self.central_modality)
                    for i in data_pair.index.to_list()
                ],
            )
        else:
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
