from typing import Dict, List, Tuple  # noqa: UP035, I002

import polars as pl
import selfies as sf
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from networkx import Graph
from torch import Tensor
from torch.utils.data import DataLoader

from molbind.data.available import MODALITY_DATA_TYPES, STRING_TOKENIZERS
from molbind.data.components.datasets import (
    FingerprintMolBindDataset,
    GraphDataset,
    StringDataset,
)
from molbind.data.utils.graph_utils import pyg_from_smiles


class MolBindDataset:
    def __init__(
        self,
        data: pl.DataFrame,
        central_modality: str,
        other_modalities: List[str],  # noqa: UP006
        **kwargs,
    ) -> None:
        """Dataset for multimodal data."""
        self.data = data
        self.central_modality = central_modality
        self.central_modality_data_type = MODALITY_DATA_TYPES[central_modality]
        self.central_modality_data = self._tokenize_strings(
            self.data[central_modality].to_list(),
            context_length=kwargs["context_length"],
            modality=central_modality,
        )
        self.other_modalities = other_modalities

    def build_graph_dataset(self) -> GraphDataset:
        modality = "graph"
        graph_data = self.data[[self.central_modality, modality]].drop_nulls()
        # perform graph operations
        return GraphDataset(graph_data, modality, self.central_modality)

    def build_string_dataset(self, modality, context_length=256) -> StringDataset:
        string_data = self.data[[self.central_modality, modality]]
        other_modality_data = self._tokenize_strings(
            string_data[modality].to_list(),
            context_length=context_length,
            modality=modality,
        )
        return StringDataset(
            central_modality=self.central_modality,
            other_modality=modality,
            central_modality_data=self.central_modality_data,
            other_modality_data=other_modality_data,
        )

    def build_fp_dataset(self) -> FingerprintMolBindDataset:
        fp_data = self.data[[self.central_modality, "fingerprint"]]
        # perform fingerprint operations
        return FingerprintMolBindDataset(
            central_modality=self.central_modality,
            fingerprint_data=fp_data["fingerprint"],
            central_modality_data=self.central_modality_data,
        )

    def build_multimodal_dataloader(
        self, batch_size: int, drop_last: bool, shuffle: bool, num_workers: int
    ) -> CombinedLoader:
        datasets, dataloaders = {}, {}
        for modality in self.other_modalities:
            if modality == "graph":
                dataset = self.build_graph_dataset()
            elif modality in ["smiles", "selfies", "inchi", "ir", "nmr", "mass"]:
                dataset = self.build_string_dataset(modality)
            elif modality == "fingerprint":
                dataset = self.build_fp_dataset()
            datasets[modality] = dataset

        for modality in [*datasets]:
            dataloaders[modality] = DataLoader(
                datasets[modality],
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
            )
        return CombinedLoader(dataloaders, "sequential")

    @classmethod
    def general_assert(cls: "MolBindDataset") -> None:
        assert cls.central_modality_data_type == str
        assert cls.central_modality in cls.data.columns
        assert (
            cls.central_modality_data_type == MODALITY_DATA_TYPES[cls.central_modality]
        )

    @classmethod
    def specific_assert_dataset(cls: "MolBindDataset") -> None:
        assert len(cls.data.columns) == 2
        assert len(cls.central_modality_data) == len(cls.data[cls.other_modalities[0]])

    # private class methods
    @staticmethod
    def _tokenize_strings(
        dataset: List[str],
        context_length: int,
        modality: str,  # noqa: UP006
    ) -> Tuple[Tensor, Tensor]:  # noqa: UP006
        tokenized_data = STRING_TOKENIZERS[modality](
            dataset,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=context_length,
        )
        return tokenized_data["input_ids"], tokenized_data["attention_mask"]

    @staticmethod
    def _build_graph_from_smiles(smi_list: List[str]) -> List[Graph]:  # noqa: UP006
        return pyg_from_smiles(smi_list)

    @staticmethod
    def _build_selfies_from_smiles(smi_list: List[str]) -> List[str]:  # noqa: UP006
        return [sf.encoder(smi) for smi in smi_list]

    @staticmethod
    def _build_smiles_from_selfies(selfies_list: List[str]) -> List[str]:  # noqa: UP006
        return [sf.decoder(selfies) for selfies in selfies_list]
