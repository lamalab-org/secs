"""Every modality encoder honours the same contract: build, forward, freeze, load."""

import pytest
import torch

from secs.models import CNmrTransformerEncoder, HNmrCNNEncoder, HsqcCNNEncoder, IrCNNEncoder
from secs.models.base import ModalityEncoder, unwrap_state_dict

# Keep the CNNs at their smallest scale: these tests are about the shared API,
# not about the capacity of any particular backbone.
ENCODERS = {
    "c_nmr": (
        lambda **kw: CNmrTransformerEncoder(embed_dim=16, depth=1, num_heads=2, **kw),
        lambda: (torch.rand(2, 6) * 200, torch.ones(2, 6, dtype=torch.bool)),
    ),
    "h_nmr": (
        lambda **kw: HNmrCNNEncoder(scale="tiny", **kw),
        lambda: torch.randn(2, 1, 2048),
    ),
    "hsqc": (
        lambda **kw: HsqcCNNEncoder(scale="tiny", input_height=64, input_width=64, **kw),
        lambda: torch.randn(2, 1, 64, 64),
    ),
    "ir": (IrCNNEncoder, lambda: torch.randn(2, 1, 1600)),
}


@pytest.fixture(params=sorted(ENCODERS))
def modality(request):
    return request.param


def build(modality: str, **kwargs) -> ModalityEncoder:
    return ENCODERS[modality][0](**kwargs)


def sample(modality: str):
    return ENCODERS[modality][1]()


def test_encoder_returns_one_embedding_per_sample(modality):
    encoder = build(modality).eval()
    with torch.no_grad():
        embedding = encoder(sample(modality))
    assert embedding.ndim == 2
    assert embedding.shape[0] == 2


def test_output_dim_matches_the_embedding_when_declared(modality):
    encoder = build(modality).eval()
    assert encoder.output_dim is not None
    with torch.no_grad():
        embedding = encoder(sample(modality))
    assert embedding.shape[1] == encoder.output_dim


def test_freezing_stops_gradients_and_pins_eval_mode(modality):
    encoder = build(modality, freeze_encoder=True)
    assert all(not p.requires_grad for p in encoder.parameters())
    # .train() must not wake a frozen backbone up: BatchNorm/dropout would
    # otherwise keep drifting while the weights stay fixed.
    encoder.train()
    assert not encoder._backbone.training
    assert all(not module.training for module in encoder._backbone.modules())


def test_unfrozen_encoder_trains_by_default(modality):
    encoder = build(modality)
    assert any(p.requires_grad for p in encoder.parameters())
    encoder.train()
    assert encoder.training


@pytest.mark.parametrize("prefix", ["", "encoder.", "model.encoder."])
def test_checkpoint_round_trip(modality, tmp_path, prefix):
    """A backbone reloads whether it was saved standalone or inside a Lightning run."""
    trained = build(modality)
    ckpt = tmp_path / "encoder.ckpt"
    backbone_state = trained.encoder.state_dict() if modality != "ir" else trained.state_dict()
    torch.save({"state_dict": {f"{prefix}{k}": v for k, v in backbone_state.items()}}, ckpt)

    restored = build(modality, ckpt_path=str(ckpt))
    trained.eval()
    restored.eval()
    batch = sample(modality)
    with torch.no_grad():
        assert torch.allclose(trained(batch), restored(batch), atol=1e-6)


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"weight": 1},
        {"state_dict": {"weight": 1}},
        {"model": {"weight": 1}},
        {"state_dict": {"encoder.weight": 1}},
        {"state_dict": {"model.encoder.weight": 1}},
    ],
)
def test_unwrap_state_dict_handles_nesting_and_prefixes(checkpoint):
    assert unwrap_state_dict(checkpoint) == {"weight": 1}


def test_c_nmr_encoder_ignores_padding_peaks():
    """Padded slots must not change the embedding, otherwise batching leaks."""
    encoder = build("c_nmr").eval()
    shifts = torch.tensor([[20.0, 60.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False, False]])
    with torch.no_grad():
        base = encoder((shifts, mask))
        noisy = encoder((torch.tensor([[20.0, 60.0, 111.0, 7.0]]), mask))
    assert torch.allclose(base, noisy, atol=1e-6)


def test_c_nmr_encoder_survives_a_fully_padded_row():
    """An all-padding row used to make attention softmax over -inf and return NaN."""
    encoder = build("c_nmr").eval()
    shifts = torch.zeros(2, 4)
    mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
    with torch.no_grad():
        embedding = encoder((shifts, mask))
    assert torch.isfinite(embedding).all()
    assert torch.count_nonzero(embedding[1]) == 0
