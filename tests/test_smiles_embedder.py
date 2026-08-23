"""SmilesEmbedder enters each model through its own central tower: SMILES tokens or RDKit graphs."""

import pytest
import torch

try:
    from secs.data.modalities import ModalityConstants
    from secs.elucidation.embedding import SmilesEmbedder

    ModalityConstants["smiles"].tokenizer  # noqa: B018 - fetched from the Hub; skip the module if unavailable
except Exception as exc:  # pragma: no cover
    pytest.skip(f"MoLFormer tokenizer unavailable: {exc}", allow_module_level=True)


class StubModel:
    """Looks like a MolBind to the embedder: a central modality and an encode_modality."""

    def __init__(self, central_modality: str, dim: int = 4) -> None:
        self.central_modality = central_modality
        self.dim = dim
        self.calls: list[str] = []

    def encode_modality(self, inputs, modality: str) -> torch.Tensor:
        self.calls.append(modality)
        if modality == "graph":
            # one row per graph; encode the atom count so rows are distinguishable
            counts = torch.bincount(inputs.batch, minlength=inputs.num_graphs).float()
            return counts.unsqueeze(1).repeat(1, self.dim)
        input_ids, _ = inputs
        return torch.ones(input_ids.shape[0], self.dim)


def test_graph_central_model_is_entered_through_the_graph_tower():
    model = StubModel("graph")
    out = SmilesEmbedder({"h_nmr": model}, device="cpu").encode(["CCO", "c1ccccc1"])["h_nmr"]
    assert model.calls == ["graph"]
    assert out.shape == (2, 4)
    assert out[0, 0] == 3.0  # ethanol: 3 heavy atoms
    assert out[1, 0] == 6.0  # benzene: 6


def test_smiles_central_model_is_entered_through_the_smiles_tower():
    model = StubModel("smiles")
    out = SmilesEmbedder({"c_nmr": model}, device="cpu").encode(["CCO", "c1ccccc1"])["c_nmr"]
    assert model.calls == ["smiles"]
    assert out.shape == (2, 4)


def test_mixed_models_each_use_their_own_tower():
    graph_model, smiles_model = StubModel("graph"), StubModel("smiles")
    out = SmilesEmbedder({"h_nmr": graph_model, "c_nmr": smiles_model}, device="cpu").encode(["CCO"])
    assert set(out) == {"h_nmr", "c_nmr"}
    assert graph_model.calls == ["graph"]
    assert smiles_model.calls == ["smiles"]


def test_unparseable_candidates_get_a_zero_embedding_and_keep_their_row():
    model = StubModel("graph")
    out = SmilesEmbedder({"h_nmr": model}, device="cpu").encode(["CCO", "not_a_smiles", "C"])["h_nmr"]
    assert out.shape == (3, 4)
    assert (out[1] == 0).all()
    assert out[0, 0] == 3.0
    assert out[2, 0] == 1.0


def test_all_unparseable_chunk_still_has_the_right_width():
    model = StubModel("graph")
    out = SmilesEmbedder({"h_nmr": model}, device="cpu").encode(["??", "not_a_smiles"])["h_nmr"]
    assert out.shape == (2, 4)
    assert (out == 0).all()


def test_chunking_does_not_change_results():
    model = StubModel("graph")
    smiles = ["CCO", "c1ccccc1", "CC(=O)OC", "C", "CCN"]
    whole = SmilesEmbedder({"h_nmr": model}, device="cpu", chunk_size=100).encode(smiles)["h_nmr"]
    chunked = SmilesEmbedder({"h_nmr": model}, device="cpu", chunk_size=2).encode(smiles)["h_nmr"]
    assert torch.equal(whole, chunked)


def test_unknown_central_modality_is_rejected_up_front():
    with pytest.raises(ValueError, match="central modality"):
        SmilesEmbedder({"h_nmr": StubModel("ir")}, device="cpu")
