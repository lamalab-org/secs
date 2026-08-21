"""Shared stubs: a tiny registered encoder so tests never fetch pretrained weights."""

import pytest
from omegaconf import DictConfig, OmegaConf
from torch import nn

from secs.models import ModalityEncoder, register_encoder
from secs.models.registry import ENCODER_REGISTRY

STUB_BACKBONE = "stub"


class StubSmilesEncoder(ModalityEncoder):
    """Stands in for MoLFormer so tests never reach the network."""

    output_dim = 8

    def __init__(self, freeze_encoder: bool = False, **kwargs) -> None:
        super().__init__(freeze_encoder=freeze_encoder)
        self.encoder = nn.Embedding(64, self.output_dim)
        self._finalize()

    def forward(self, x):
        token_ids, attention_mask = x[0], x[1]
        embedded = self.encoder(token_ids) * attention_mask.unsqueeze(-1)
        return embedded.sum(dim=1)


@pytest.fixture
def registered_stub():
    """Register a second smiles backbone and take it back out again."""
    register_encoder("smiles", STUB_BACKBONE)(StubSmilesEncoder)
    yield STUB_BACKBONE
    ENCODER_REGISTRY["smiles"].pop(STUB_BACKBONE)


@pytest.fixture
def stub_encoder_cls():
    return StubSmilesEncoder


@pytest.fixture
def secs_config():
    """Factory for a minimal but complete config: enough for MolBind and SECSModule."""
    return _secs_config


def _secs_config(smiles_backbone: str, c_nmr_backbone: str | None = "transformer", **overrides) -> DictConfig:
    c_nmr_encoder = {"embed_dim": 16, "depth": 1, "num_heads": 2}
    if c_nmr_backbone is not None:
        c_nmr_encoder["name"] = c_nmr_backbone
    config = OmegaConf.create(
        {
            "data": {"central_modality": "smiles", "modalities": ["c_nmr"], "batch_size": 4},
            "trainer": {"gpus_per_node": 1, "num_nodes": 1},
            "model": {
                "encoders": {"smiles": {"name": smiles_backbone}, "c_nmr": c_nmr_encoder},
                "projection_heads": {
                    "smiles_is_on": True,
                    "c_nmr_is_on": True,
                    "smiles_freeze": False,
                    "c_nmr_freeze": False,
                    "smiles": {"dims": [StubSmilesEncoder.output_dim, 4], "activation": "LeakyReLU"},
                    "c_nmr": {"dims": [16, 4], "activation": "LeakyReLU"},
                },
                "loss": {"temperature": 0.07, "symmetric": True},
                "optimizer": {"lr": 1e-4, "weight_decay": 1e-4},
            },
        }
    )
    return OmegaConf.merge(config, OmegaConf.create(overrides))
