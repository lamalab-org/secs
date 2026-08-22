import pandas as pd
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from secs.data.components.central import (
    CentralModalityData,
    GraphCentralModality,
    TokenizedCentralModality,
)
from secs.data.modalities import (
    ModalityConstants,
    NonStringModalities,
    StringModalities,
)

DERIVED_FROM: dict[str, str] = {NonStringModalities.GRAPH: StringModalities.SMILES}


def source_column(modality: str) -> str:
    """The frame column a modality is read from, itself unless it is derived."""
    return DERIVED_FROM.get(modality, modality)


def columns_to_read(modalities: list[str], central_modality: str) -> list[str]:
    """The frame columns a run actually has to load for these modalities."""
    wanted = [DERIVED_FROM.get(modality, modality) for modality in [*modalities, central_modality]]
    return [*dict.fromkeys(wanted)]


def derive_modality_columns(data: pd.DataFrame, modalities: list[str], central_modality: str) -> pd.DataFrame:
    """Fill in modalities read off another column instead of stored."""
    derived = {
        modality: DERIVED_FROM[modality]
        for modality in [*modalities, central_modality]
        if modality in DERIVED_FROM and modality not in data.columns
    }
    missing = {modality: source for modality, source in derived.items() if source not in data.columns}
    if missing:
        raise ValueError(f"Cannot build {sorted(missing)}: the frame has no {sorted(set(missing.values()))} column.")
    return data.assign(**{modality: data[source] for modality, source in derived.items()}) if derived else data


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
        self.data = derive_modality_columns(self.data, other_modalities, central_modality)
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

    def _encode_central_modality(self) -> CentralModalityData:
        """Prepare the central modality for row-by-row pairing."""
        values = self.data[self.central_modality].to_list()
        data_type = ModalityConstants[self.central_modality].data_type
        if data_type is str:
            return TokenizedCentralModality(*self._tokenize_strings(values, self.central_modality, self.context_length))
        if data_type is Data:
            return GraphCentralModality(values)
        raise ValueError(f"Central modality {self.central_modality} is not supported yet.")

    def _select_central_modality_rows(self, index: pd.Index) -> CentralModalityData:
        return self.central_modality_data.select(index.to_list())

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
