"""Encoding candidate SMILES with trained SECS models.

`secs-app` hardcoded ``device="cpu"`` and the research copy hardcoded
``"cuda"``; that single difference was the main reason the two copies could
not be shared. Here the device is a constructor argument that defaults to
whatever hardware is present.
"""

from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from loguru import logger
from torch import Tensor

from secs.data.modalities import ModalityConstants
from secs.models import MolBind
from secs.utils import rename_keys_with_prefix, select_device


def load_model(config, device: str) -> MolBind:
    """Build a MolBind from a composed Hydra config and load its checkpoint."""
    model = MolBind(config).to(device)
    model.load_state_dict(
        rename_keys_with_prefix(torch.load(config.ckpt_path, map_location=torch.device(device))["state_dict"]),
        strict=True,
    )
    model.eval()
    return model


def load_models(
    configs_path: str | Path,
    experiments: dict[str, str | None],
    device: str | None = None,
) -> dict[str, MolBind]:
    """Load one model per modality from Hydra experiment names.

    Modalities whose experiment is None, or which fail to load, are omitted
    from the result rather than mapped to None -- callers previously had to
    filter Nones out again at every use site.
    """
    device = device or select_device()
    config_dir = str(Path(configs_path).resolve())
    models: dict[str, MolBind] = {}
    for modality, experiment in experiments.items():
        if not experiment:
            continue
        try:
            with initialize_config_dir(version_base="1.3", config_dir=config_dir):
                config = compose(config_name="molbind_config", overrides=[f"experiment={experiment}"])
            models[modality] = load_model(config, device)
            logger.info(f"Loaded {modality} model from experiment: {experiment}")
        except Exception as error:  # noqa: BLE001  (one bad model must not sink the run)
            logger.warning(f"Failed to load {modality} model ({experiment}): {error}")
    return models


class SmilesEmbedder:
    """Encodes SMILES into each modality's shared embedding space.

    One MolBind per modality, all encoding through the *smiles* tower, so a
    candidate structure can be compared against a target spectrum embedding.
    """

    def __init__(
        self,
        models: dict[str, MolBind],
        device: str | None = None,
        context_length: int = 128,
        chunk_size: int = 8192,
    ) -> None:
        self.models = models
        self.device = device or select_device()
        self.context_length = context_length
        self.chunk_size = chunk_size
        self._tokenizer = ModalityConstants["smiles"].tokenizer

    @property
    def modalities(self) -> list[str]:
        return list(self.models)

    def tokenize(self, smiles: list[str]) -> tuple[Tensor, Tensor]:
        tokens = self._tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.context_length,
        )
        return tokens["input_ids"], tokens["attention_mask"]

    def _encode_batch(self, smiles: list[str]) -> dict[str, Tensor]:
        input_ids, attention_mask = self.tokenize(smiles)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        embeddings: dict[str, Tensor] = {}
        with torch.inference_mode():
            for modality, model in self.models.items():
                embeddings[modality] = model.encode_modality((input_ids, attention_mask), modality="smiles")
        return embeddings

    def encode(self, smiles: list[str]) -> dict[str, Tensor]:
        """Encode SMILES in chunks. Returns {modality: (N, D)} on the CPU.

        Results come back on the CPU so large candidate sets do not pin GPU
        memory between generations.
        """
        if not smiles or not self.models:
            return {modality: torch.empty(0) for modality in self.models}

        parts: dict[str, list[Tensor]] = {modality: [] for modality in self.models}
        for start in range(0, len(smiles), self.chunk_size):
            chunk = smiles[start : start + self.chunk_size]
            for modality, embedding in self._encode_batch(chunk).items():
                parts[modality].append(embedding.cpu())

        return {modality: torch.cat(tensors, dim=0) for modality, tensors in parts.items()}
