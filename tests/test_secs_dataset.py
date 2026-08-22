"""SECSDataset pairs each modality with the central one; alignment is the thing to guard."""

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

try:
    from secs.data.modalities import ModalityConstants
    from secs.data.secs_dataset import SECSDataset
except Exception as exc:  # the SMILES tokenizer is fetched from the Hub
    pytest.skip(f"MoLFormer tokenizer unavailable: {exc}", allow_module_level=True)

SMILES = ["CCO", "c1ccccc1", "CCN", "CCC"]
RNG = np.random.default_rng(0)
CONTEXT_LENGTH = 24


def frame(**columns) -> pd.DataFrame:
    return pd.DataFrame({"smiles": SMILES, **columns})


def build(data: pd.DataFrame, modalities: list[str], config=None) -> dict:
    return SECSDataset(
        data=data,
        central_modality="smiles",
        other_modalities=modalities,
        config=config,
        context_length=CONTEXT_LENGTH,
    ).build_datasets_for_modalities()


def tokenize(smiles: list[str]):
    return SECSDataset._tokenize_strings(smiles, "smiles", CONTEXT_LENGTH)


@pytest.fixture
def ir_column():
    return [list(np.linspace(0, 1, 1800)) for _ in SMILES]


def test_builds_one_dataset_per_requested_modality(ir_column):
    datasets = build(frame(c_nmr=[[20.0, 60.0]] * 4, ir=ir_column), ["c_nmr", "ir"])
    assert set(datasets) == {"c_nmr", "ir"}
    assert isinstance(datasets["c_nmr"], ModalityConstants["c_nmr"].dataset)
    assert isinstance(datasets["ir"], ModalityConstants["ir"].dataset)
    assert len(datasets["c_nmr"]) == len(SMILES)


def test_modality_missing_from_the_frame_is_skipped():
    datasets = build(frame(c_nmr=[[20.0]] * 4), ["c_nmr", "hsqc"])
    assert set(datasets) == {"c_nmr"}


def test_rows_missing_a_modality_are_dropped_with_their_central_row():
    """The central tokens must follow the surviving rows, not the original ones."""
    data = frame(c_nmr=[[20.0, 60.0], None, [12.0, 44.0], None])
    dataset = build(data, ["c_nmr"])["c_nmr"]

    assert len(dataset) == 2
    expected_ids, expected_mask = tokenize(["CCO", "CCN"])
    for row, (ids, mask) in enumerate(zip(expected_ids, expected_mask, strict=True)):
        central_ids, central_mask = dataset[row]["smiles"]
        assert (central_ids == ids).all()
        assert (central_mask == mask).all()


def test_c_nmr_rows_without_usable_peaks_drop_their_central_row_too():
    """cNmrDataset filters out-of-range peak lists; the pairing has to survive that."""
    data = frame(c_nmr=[[20.0, 60.0], [9_999.0], [12.0], [-500.0]])
    dataset = build(data, ["c_nmr"])["c_nmr"]

    assert len(dataset) == 2
    expected_ids, _ = tokenize(["CCO", "CCN"])
    for row, ids in enumerate(expected_ids):
        assert (dataset[row]["smiles"][0] == ids).all()


def test_c_nmr_items_are_padded_shifts_and_a_mask():
    dataset = build(frame(c_nmr=[[20.0, 60.0, 90.0]] * 4), ["c_nmr"])["c_nmr"]
    shifts, mask = dataset[0]["c_nmr"]
    assert shifts.shape == mask.shape == (dataset.max_peaks,)
    assert mask.sum() == 3
    assert shifts[mask].tolist() == [20.0, 60.0, 90.0]
    assert shifts[~mask].sum() == 0


def test_h_nmr_reads_its_knobs_from_the_config():
    data = frame(h_nmr=[list(RNG.random(10_000)) for _ in SMILES])
    config = OmegaConf.create({"data": {"h_nmr": {"augment": True, "vec_size": 2_000}}})
    dataset = build(data, ["h_nmr"], config=config)["h_nmr"]
    assert dataset.augment is True
    assert dataset.vec_size == 2_000


@pytest.mark.parametrize(
    "config",
    [None, OmegaConf.create({"data": {}}), OmegaConf.create({"data": {"h_nmr": {"augment": False}}})],
)
def test_h_nmr_falls_back_when_the_config_says_nothing(config):
    """A config without an h_nmr block must not raise; it used to AttributeError."""
    data = frame(h_nmr=[list(RNG.random(10_000)) for _ in SMILES])
    dataset = build(data, ["h_nmr"], config=config)["h_nmr"]
    assert dataset.augment is False
    assert dataset.vec_size == 10_000


def test_central_modality_is_tokenized_once_for_the_whole_frame():
    dataset = SECSDataset(
        data=frame(c_nmr=[[20.0]] * 4),
        central_modality="smiles",
        other_modalities=["c_nmr"],
        context_length=CONTEXT_LENGTH,
    )
    central = dataset.central_modality_data
    assert central.input_ids.shape == central.attention_mask.shape == (len(SMILES), CONTEXT_LENGTH)
    # rows come back as the pair the encoder takes
    input_ids, attention_mask = central[0]
    assert input_ids.shape == attention_mask.shape == (CONTEXT_LENGTH,)


def test_non_string_central_modality_is_refused():
    with pytest.raises(ValueError, match="Central modality c_nmr is not supported yet"):
        SECSDataset(
            data=pd.DataFrame({"c_nmr": [[20.0]], "ir": [[0.0]]}),
            central_modality="c_nmr",
            other_modalities=["ir"],
        )
