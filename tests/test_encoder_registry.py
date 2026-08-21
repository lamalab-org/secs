"""The registry is what lets one modality have several backbones."""

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from secs.models import ModalityEncoder, MolBind, available_encoders, register_encoder, resolve_encoder
from secs.models.encoders.c_nmr.transformer import CNmrTransformerEncoder
from secs.models.encoders.smiles.molformer import MolformerEncoder
from secs.models.registry import DEFAULT_BACKBONE, ENCODER_REGISTRY


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
    register_encoder("smiles", "stub")(StubSmilesEncoder)
    yield "stub"
    ENCODER_REGISTRY["smiles"].pop("stub")


def molbind_config(smiles_backbone: str, c_nmr_backbone: str | None) -> OmegaConf:
    c_nmr_encoder = {"embed_dim": 16, "depth": 1, "num_heads": 2}
    if c_nmr_backbone is not None:
        c_nmr_encoder["name"] = c_nmr_backbone
    return OmegaConf.create(
        {
            "data": {"central_modality": "smiles", "modalities": ["c_nmr"]},
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
            },
        }
    )


@pytest.mark.parametrize(
    ("modality", "backbone"),
    [("c_nmr", "transformer"), ("h_nmr", "cnn"), ("hsqc", "cnn"), ("ir", "cnn"), ("smiles", "molformer")],
)
def test_builtin_encoders_are_registered(modality, backbone):
    """Importing `secs.models` is enough to populate the registry."""
    assert backbone in available_encoders()[modality]


def test_resolve_falls_back_to_the_modality_default():
    assert resolve_encoder("c_nmr") is CNmrTransformerEncoder
    assert resolve_encoder("smiles") is MolformerEncoder
    assert DEFAULT_BACKBONE["c_nmr"] == "transformer"


def test_resolve_picks_the_named_backbone(registered_stub):
    assert resolve_encoder("smiles", registered_stub) is StubSmilesEncoder
    # registering a non-default backbone must not steal the default
    assert resolve_encoder("smiles") is MolformerEncoder


def test_unknown_modality_and_backbone_are_reported():
    with pytest.raises(ValueError, match="No encoder registered for modality 'raman'"):
        resolve_encoder("raman")
    with pytest.raises(ValueError, match="Unknown backbone 'lstm' for modality 'c_nmr'"):
        resolve_encoder("c_nmr", "lstm")


def test_registering_the_same_name_twice_is_refused(registered_stub):
    with pytest.raises(ValueError, match="already registered"):
        register_encoder("smiles", registered_stub)(CNmrTransformerEncoder)


def test_molbind_builds_the_backbone_named_in_the_config(registered_stub):
    model = MolBind(molbind_config(registered_stub, "transformer"))
    assert isinstance(model.dict_encoders["smiles"], StubSmilesEncoder)
    assert isinstance(model.dict_encoders["c_nmr"], CNmrTransformerEncoder)
    # `name` is consumed by the registry, the rest of the block is backbone kwargs
    assert model.dict_encoders["c_nmr"].output_dim == 16


def test_molbind_defaults_the_backbone_when_the_config_omits_a_name(registered_stub):
    model = MolBind(molbind_config(registered_stub, None))
    assert isinstance(model.dict_encoders["c_nmr"], CNmrTransformerEncoder)


def test_molbind_forward_returns_one_embedding_per_modality(registered_stub):
    model = MolBind(molbind_config(registered_stub, "transformer")).eval()
    batch = {
        "smiles": (torch.randint(0, 64, (3, 5)), torch.ones(3, 5, dtype=torch.long)),
        "c_nmr": (torch.rand(3, 7) * 200, torch.ones(3, 7, dtype=torch.bool)),
    }
    with torch.no_grad():
        embeddings = model(batch)
    assert set(embeddings) == {"smiles", "c_nmr"}
    assert embeddings["smiles"].shape == embeddings["c_nmr"].shape == (3, 4)


def test_molbind_rejects_an_unsupported_modality(registered_stub):
    cfg = molbind_config(registered_stub, "transformer")
    cfg.data.modalities = ["raman"]
    with pytest.raises(ValueError, match="Modality raman not supported yet"):
        MolBind(cfg)
