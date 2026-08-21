import pandas as pd
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset

from secs.data.modalities import (
    ModalityConstants,
    NonStringModalities,
    StringModalities,
)


class SECSDataset:
    def __init__(
        self,
        data: pd.DataFrame,
        central_modality: StringModalities | NonStringModalities,
        other_modalities: list[str],
        config: dict | None = None,
        context_length: int = 256,
        split: str = "train",
    ) -> None:
        self.data = data.reset_index(drop=True)
        self.central_modality = central_modality
        self.other_modalities = other_modalities
        self.config = config
        self.context_length = context_length
        # "train" enables train-only augmentation (e.g. image depictions).
        self.split = split
        self.central_modality_data = self._encode_central_modality()

    def build_datasets_for_modalities(self) -> dict[str, Dataset]:
        """One Dataset per requested modality that the dataframe actually carries.

        CombinedLoader does not work with DDPSampler directly, so the sampler is
        added to the dataloaders in the datamodule rather than here.
        """
        datasets = {}
        for modality in self.other_modalities:
            if modality not in self.data.columns:
                logger.warning(f"Modality {modality} requested but missing from the dataframe; skipping.")
                continue
            datasets[modality] = self._build_dataset(modality)
        return datasets

    def _build_dataset(self, modality: str) -> Dataset:
        spec = ModalityConstants[modality]
        paired = self.data[[self.central_modality, modality]].dropna()
        central_data = self._select_central_modality_rows(paired.index)

        if spec.data_type is str:
            return spec.dataset(
                central_modality=self.central_modality,
                central_modality_data=central_data,
                other_modality=modality,
                other_modality_data=self._tokenize_strings(paired[modality].to_list(), modality, self.context_length),
            )
        return spec.dataset(
            data=paired[modality].to_list(),
            central_modality=self.central_modality,
            central_modality_data=central_data,
            **self._dataset_kwargs(modality),
        )

    def _dataset_kwargs(self, modality: str) -> dict:
        """Per-modality knobs read off the experiment config, if it carries them."""
        if modality == NonStringModalities.H_NMR:
            h_nmr_cfg = getattr(getattr(self.config, "data", None), "h_nmr", None)
            return {
                "augment": getattr(h_nmr_cfg, "augment", False),
                "vec_size": getattr(h_nmr_cfg, "vec_size", 10_000),
            }
        return {}

    def _encode_central_modality(self) -> tuple[Tensor, Tensor]:
        values = self.data[self.central_modality].to_list()
        if ModalityConstants[self.central_modality].data_type is not str:
            raise ValueError(f"Central modality {self.central_modality} is not supported yet.")
        return self._tokenize_strings(values, self.central_modality, self.context_length)

    def _select_central_modality_rows(self, index: pd.Index) -> tuple[Tensor, Tensor]:
        rows = index.to_list()
        return self.central_modality_data[0][rows], self.central_modality_data[1][rows]

    @staticmethod
    def _tokenize_strings(dataset: list[str], modality: str, context_length: int) -> tuple[Tensor, Tensor]:
        tokenizer = ModalityConstants[modality].tokenizer
        tokenized_data = tokenizer(
            dataset,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=context_length,
        )
        return tokenized_data["input_ids"], tokenized_data["attention_mask"]
