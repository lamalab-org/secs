from functools import partial
import gc

import pandas as pd
import polars as pl
import selfies as sf
from torch import Tensor
from torch.utils.data import Dataset

from molbind.data.available import (
    ModalityConstants,
    StringModalities,
)
from molbind.data.components.datasets import (
    StringDataset,
)


class MolBindDataset:
    def __init__(
        self,
        data: pl.DataFrame,
        central_modality: StringModalities,
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
            StringModalities.POLYMER_NAME: init_str_fn,
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
            StringModalities.POLYMER_NAME: partial(
                self.build_string_dataset,
                modality=StringModalities.POLYMER_NAME,
                context_length=kwargs.get("context_length", 256),
            ),
        }

        if hasattr(self.data, "with_row_index"):
            # Polars DataFrame
            self.data = self.data.with_row_index()
        elif hasattr(self.data, "reset_index"):
            # Pandas DataFrame
            self.data = self.data.reset_index(drop=True)

        # Only tokenize central modality data once here
        if self.central_modality_data_type == str:
            central_data_list = self.data[central_modality].to_list()
            self.central_modality_data = self.central_modality_handlers[central_modality](central_data_list)
            # Clear the list to free memory
            del central_data_list
        else:
            self.central_modality_data = self.data[central_modality].to_list()

        self.other_modalities = other_modalities

        # Force garbage collection after initialization
        gc.collect()

    def build_string_dataset(self, modality: str, context_length: int = 256) -> StringDataset:
        # Use more memory-efficient data selection - handle both pandas and polars
        relevant_columns = [self.central_modality, modality]

        if hasattr(self.data, "select"):
            # Polars DataFrame
            string_data = self.data.select(relevant_columns).drop_nulls()
        else:
            # Pandas DataFrame
            string_data = self.data[relevant_columns].dropna()

        other_modality_data = self._tokenize_strings(
            string_data[modality].to_list(),
            context_length=context_length,
            modality=modality,
        )

        central_modality_data = self._handle_central_modality_data(string_data)

        dataset = StringDataset(
            central_modality=self.central_modality,
            other_modality=modality,
            central_modality_data=central_modality_data,
            other_modality_data=other_modality_data,
        )

        # Clean up temporary data
        del string_data, other_modality_data
        gc.collect()

        return dataset

    def build_datasets_for_modalities(self) -> dict[str, Dataset]:
        datasets = {}
        for modality in self.other_modalities:
            if modality in self.data.columns:
                dataset = self.dataset_builders[modality]()
                datasets[modality] = dataset
                gc.collect()
        return datasets

    def _handle_central_modality_data(self, data_pair) -> tuple[Tensor, Tensor]:
        # Use more efficient indexing for both pandas and polars DataFrames
        if hasattr(data_pair, "get_column") and hasattr(data_pair, "row_index"):
            # Polars DataFrame with row_index
            indices = data_pair.get_column("row_index").to_list()
        elif hasattr(data_pair, "index"):
            # Pandas DataFrame
            indices = data_pair.index.to_list()
        else:
            # Fallback to range indexing
            indices = list(range(len(data_pair)))

        if self.central_modality_data_type == str:
            if self.custom_negatives:
                # Use list comprehension with memory cleanup
                custom_neg_tokenized = []
                for i in indices:
                    tokenized = self._tokenize_strings(list(self.custom_negatives_samples[i]), 128, self.central_modality)
                    custom_neg_tokenized.append(tokenized)

                central_modality_data = (
                    self.central_modality_data[0][indices],
                    self.central_modality_data[1][indices],
                    custom_neg_tokenized,
                )
            else:
                central_modality_data = (
                    self.central_modality_data[0][indices],
                    self.central_modality_data[1][indices],
                )
        else:
            # For non-string modalities, use more efficient indexing
            central_modality_data = [self.central_modality_data[i] for i in indices]

        return central_modality_data

    @staticmethod
    def _tokenize_strings(
        dataset: list[str],
        context_length: int,
        modality: str,
    ) -> tuple[Tensor, Tensor]:
        # Add memory optimization for tokenization
        tokenizer = ModalityConstants[modality].tokenizer

        # Process in smaller batches to reduce memory usage
        batch_size = min(1000, len(dataset))
        all_input_ids = []
        all_attention_masks = []

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            tokenized_batch = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=context_length,
            )
            all_input_ids.append(tokenized_batch["input_ids"])
            all_attention_masks.append(tokenized_batch["attention_mask"])

            # Clear batch data
            del tokenized_batch, batch

        # Concatenate all batches
        import torch

        final_input_ids = torch.cat(all_input_ids, dim=0)
        final_attention_masks = torch.cat(all_attention_masks, dim=0)

        # Clean up temporary lists
        del all_input_ids, all_attention_masks
        gc.collect()

        return final_input_ids, final_attention_masks

    @staticmethod
    def _build_selfies_from_smiles(smi_list: list[str]) -> list[str]:
        return [sf.encoder(smi) for smi in smi_list]

    @staticmethod
    def _build_smiles_from_selfies(selfies_list: list[str]) -> list[str]:
        return [sf.decoder(selfies) for selfies in selfies_list]
