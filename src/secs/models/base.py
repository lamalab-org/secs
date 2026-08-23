"""Common API shared by every modality encoder.

A modality can have several encoders (different backbones for the same kind of
spectrum). They all look the same to `MolBind`:

    encoder = SomeEncoder(ckpt_path=..., freeze_encoder=..., **backbone_kwargs)
    embedding = encoder(batch)          # (B, output_dim)

Subclasses build their backbone into `self.encoder`, set `self.output_dim`, and
then call `self._finalize()` to get checkpoint loading and freezing for free.
"""

from abc import abstractmethod

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from transformers import AutoModelForCausalLM


def xavier_init(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        if len(param.shape) > 1:
            nn.init.xavier_uniform_(param)
    return model


def unwrap_state_dict(state: dict, prefixes: tuple[str, ...] = ("model.", "encoder.")) -> dict:
    """Pull the backbone tensors out of a checkpoint saved by any of our entry points.

    Checkpoints come either raw or wrapped under 'state_dict'/'model'/'encoder',
    with keys carrying the module path they were saved under ("encoder." from a
    standalone encoder run, "model.encoder." from a Lightning one).
    """
    for key in ("state_dict", "model", "encoder"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    cleaned = {}
    for key, value in state.items():
        stripped = True
        while stripped:
            stripped = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]  # noqa: PLW2901
                    stripped = True
                    break
        cleaned[key] = value
    return cleaned


class ModalityEncoder(nn.Module):
    """Base class for all modality encoders.

    Args:
        ckpt_path: optional checkpoint for the backbone.
        freeze_encoder: freeze the backbone parameters and keep it in eval mode.
    """

    #: dimension of the embedding returned by `forward`, before the projection head
    output_dim: int | None = None

    def __init__(self, ckpt_path: str | None = None, freeze_encoder: bool = False) -> None:
        super().__init__()
        self.ckpt_path = ckpt_path
        self.frozen = freeze_encoder

    @property
    def _backbone(self) -> nn.Module:
        """The module that checkpoints load into and that freezing applies to."""
        return self.encoder

    def _finalize(self) -> None:
        """Load the checkpoint and apply freezing. Call at the end of `__init__`."""
        if self.ckpt_path is not None:
            self.load_checkpoint(self.ckpt_path)
        if self.frozen:
            self.freeze()

    def load_checkpoint(self, ckpt_path: str, strict: bool = False) -> None:
        """Load a backbone checkpoint, raw `.pth` state dict or Lightning `.ckpt` alike.

        Loading is non-strict, so a checkpoint may carry heads this backbone does
        not build (a MolCLR `.pth` brings its contrastive `out_lin` along). What
        it may not do is match *nothing*: that leaves a randomly initialised
        backbone behind a config that says "pretrained", which is a whole
        training run wasted on a silent typo.
        """
        state = unwrap_state_dict(torch.load(ckpt_path, map_location="cpu"))
        # Counted by hand rather than from `missing`: BatchNorm keeps
        # `num_batches_tracked` out of the missing list, which would make a
        # checkpoint that matched nothing look like it matched one key per norm.
        backbone_state = self._backbone.state_dict()
        loaded = sum(1 for key, value in state.items() if getattr(backbone_state.get(key), "shape", None) == value.shape)
        missing, unexpected = self._backbone.load_state_dict(state, strict=strict)
        name = type(self).__name__
        if not loaded:
            raise ValueError(
                f"{name}: nothing in {ckpt_path} matched the backbone. "
                f"The checkpoint holds keys like {sorted(state)[:3]}, "
                f"while the backbone expects {sorted(missing)[:3]}."
            )
        logger.info(f"{name}: loaded {loaded} tensors from {ckpt_path}")
        if missing:
            logger.warning(f"{name}: {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            logger.warning(f"{name}: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")

    def freeze(self) -> None:
        self.frozen = True
        for param in self._backbone.parameters():
            param.requires_grad = False
        nn.Module.train(self._backbone, False)

    def train(self, mode: bool = True):
        # A frozen backbone stays in eval: BatchNorm and dropout must not keep
        # drifting while the weights they belong to are fixed. Called through
        # `nn.Module` directly so an encoder whose `_backbone` is itself does
        # not recurse.
        nn.Module.train(self, mode)
        if self.frozen:
            nn.Module.train(self._backbone, False)
        return self

    @abstractmethod
    def forward(self, x):
        """Encode one batch of this modality into (B, output_dim)."""


class HFCausalLMEncoder(ModalityEncoder):
    """Generic HuggingFace backbone over (token_ids, attention_mask), mean pooled."""

    def __init__(
        self,
        model_name: str,
        freeze_encoder: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__(freeze_encoder=freeze_encoder)
        self.model_name = model_name
        self.pretrained = pretrained
        self._initialize_encoder()

    def _initialize_encoder(self) -> None:
        self.encoder = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.pretrained:
            if self.frozen:
                self.freeze()
        else:
            self.encoder = xavier_init(self.encoder)

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        token_ids, attention_mask = x
        output = self.encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return self._non_pad_token_embed_averaging(output.hidden_states[-1], attention_mask)

    @staticmethod
    def _non_pad_token_embed_averaging(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        attention_mask = attention_mask.float().unsqueeze(-1)
        sum_ = (last_hidden_state * attention_mask).sum(dim=1)
        norm = attention_mask.squeeze(-1).sum(dim=1).unsqueeze(1)
        return sum_ / norm
