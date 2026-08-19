"""13C shift prediction service wrapping CASCADE-2.0 (Predict_SMILES_FF_GPR).

Runs in its own container because CASCADE pins TensorFlow 2.11 / Python 3.10,
which cannot be installed alongside the SECS package. Implements the contract
that `secs.elucidation.verifiers.HttpShiftSimulator` expects:

    POST /            {"smiles": [...]}  ->  {"shifts": [[...]|null, ...],
                                              "uncertainty": [[...]|null, ...]}

Predictions come back in input order, with null for molecules that could not
be embedded or predicted.
"""

import os
import pickle
import sys
from pathlib import Path

import __main__

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # the notebook runs this on CPU

MODEL_DIR = Path(os.environ.get("CASCADE_MODEL_DIR", "/app/model"))
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR / "modules"))
# model.py loads inducing_index_points_250.npy by relative path, so it has to
# be imported with the model directory as the working directory.
os.chdir(MODEL_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from model import make_model  # noqa: E402
from nfp.preprocessing import GraphSequence  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

RDLogger.DisableLog("rdApp.*")
tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")

# Target normalisation used during training; predictions come back standardised.
SHIFT_SCALE = 50.484337
SHIFT_MEAN = 99.798111
CONFIDENCE_Z = 1.96

BATCH_SIZE = int(os.environ.get("CASCADE_BATCH_SIZE", "32"))
EMBED_SEED = int(os.environ.get("CASCADE_EMBED_SEED", "42"))


def _compute_stacked_offsets(sizes, repeats):
    return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)


def _ragged_const(arr):
    return tf.ragged.constant(np.expand_dims(arr, axis=0), ragged_rank=1)


class RBFSequence(GraphSequence):
    def process_data(self, batch_data):
        offset = _compute_stacked_offsets(batch_data["n_pro"], batch_data["n_atom"])
        offset = np.where(batch_data["atom_index"] >= 0, offset, 0)
        batch_data["atom_index"] += offset
        for feature in ["node_attributes", "node_coordinates", "edge_indices", "atom_index", "n_pro"]:
            batch_data[feature] = _ragged_const(batch_data[feature])
        for drop in ["n_atom", "n_bond", "distance", "bond", "node_graph_indices"]:
            del batch_data[drop]
        return batch_data


def _mol_iter(df):
    for _, row in df.iterrows():
        yield (row["Mol"], row["atom_index"])


def _embed(smiles: str):
    """SMILES -> MMFF-optimised 3D conformer. The model is geometry-based."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = EMBED_SEED
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except (ValueError, RuntimeError):
        return None  # no MMFF parameters for this element set
    return mol


def _carbon_indices(mol) -> np.ndarray:
    return np.array([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6], dtype=int)


def atomic_number_tokenizer(atom):
    return atom.GetAtomicNum()


# The preprocessor was pickled from a notebook, so it refers to functions by
# their __main__ qualified name. Republish them there before unpickling.
__main__.atomic_number_tokenizer = atomic_number_tokenizer

with (MODEL_DIR / "preprocessor_orig.p").open("rb") as handle:
    PREPROCESSOR = pickle.load(handle)["preprocessor"]

MODEL = make_model()
MODEL.load_weights(str(MODEL_DIR / "best_model_val_mae.h5"))

app = FastAPI(title="CASCADE-2.0 13C shift prediction")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "CASCADE-2.0 Predict_SMILES_FF_GPR"}


@app.post("/")
def predict(request: dict) -> dict:
    smiles_list = request.get("smiles", [])
    n = len(smiles_list)
    shifts: list[list[float] | None] = [None] * n
    uncertainty: list[list[float] | None] = [None] * n
    if n == 0:
        return {"shifts": shifts, "uncertainty": uncertainty}

    # Embed first; molecules that fail keep their None slot and are not sent
    # to the model, so one bad structure cannot shift the batch alignment.
    mols, slots, counts = [], [], []
    for i, smi in enumerate(smiles_list):
        mol = _embed(smi)
        if mol is None:
            continue
        indices = _carbon_indices(mol)
        if indices.size == 0:
            shifts[i], uncertainty[i] = [], []
            continue
        mols.append(mol)
        slots.append(i)
        counts.append(len(indices))

    if not mols:
        return {"shifts": shifts, "uncertainty": uncertainty}

    frame = pd.DataFrame({"Mol": mols, "atom_index": [_carbon_indices(m) for m in mols]})
    sequence = RBFSequence(PREPROCESSOR.predict(_mol_iter(frame)), batch_size=BATCH_SIZE)

    means, stddevs = [], []
    for batch in sequence:
        distribution = MODEL(batch)
        means.extend(distribution.mean().numpy().flatten())
        stddevs.extend(distribution.stddev().numpy().flatten())

    means = np.asarray(means) * SHIFT_SCALE + SHIFT_MEAN
    stddevs = np.asarray(stddevs) * SHIFT_SCALE * CONFIDENCE_Z

    cursor = 0
    for slot, count in zip(slots, counts, strict=True):
        shifts[slot] = [round(float(v), 3) for v in means[cursor : cursor + count]]
        uncertainty[slot] = [round(float(v), 3) for v in stddevs[cursor : cursor + count]]
        cursor += count

    return {"shifts": shifts, "uncertainty": uncertainty}
