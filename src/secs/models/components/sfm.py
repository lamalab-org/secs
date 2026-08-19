from pathlib import Path
from typing import Any

import torch
import yaml
from analchem.model.transformer import SpecFormer
from analchem.utils import task_shapes
from torch import nn

from secs.utils import rename_keys_with_prefix


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")

    return config


def _build_backbone(
    model_name: str,
    model_config: dict[str, Any],
    tasks: dict[str, int | None] | None = None,
) -> nn.Module:
    name = model_name.lower()

    if name in {"transformer", "specformer"}:
        return SpecFormer(**model_config, tasks=tasks)

    if name in {"mamba", "specmamba"}:
        from analchem.model.mamba import SpecMamba

        backbone = SpecMamba(**model_config, tasks=tasks)
        if not hasattr(backbone, "tasks"):
            backbone.tasks = None
        return backbone

    raise ValueError(f"Unsupported analchem model '{model_name}'.")


class SfmEmbeddingModel(nn.Module):
    """Inference wrapper whose forward pass returns analchem embeddings."""

    def __init__(self, config_path: str, ckpt_path: str, freeze_encoder: bool) -> None:
        super().__init__()
        config = _load_yaml(config_path)
        self.backbone = _build_backbone(config["module"]["model"], config["module"]["model_config"])
        if ckpt_path:
            self.backbone.load_state_dict(rename_keys_with_prefix(torch.load(ckpt_path)["state_dict"]), strict=False)
        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.backbone(x, mask=mask, return_tasks=False)
