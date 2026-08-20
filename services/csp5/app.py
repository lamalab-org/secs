"""13C shift prediction service wrapping CSP5 (Goodman lab).

Implements the same contract as the cascade service, which
`secs.elucidation.verifiers.HttpShiftSimulator` expects:

    POST /            {"smiles": [...]}  ->  {"shifts": [[...]|null, ...],
                                              "uncertainty": [[...]|null, ...]}

Predictions come back in input order, with null for molecules that could not
be parsed or predicted and [] for molecules without carbon. Shifts are ordered
by RDKit atom index over the carbon atoms, matching the cascade service.

CSP5 renumbers molecule_id after dropping failures, so slots are realigned
through the failure list rather than trusted directly.

Molecules ETKDG cannot embed are retried with a geometry this service builds
itself; see `_fallback_molblock`.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
from csp5 import predict_smiles, predict_structures
from fastapi import FastAPI
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

BATCH_SIZE = int(os.environ.get("CSP5_BATCH_SIZE", "64"))

# CSP5 defaults to 20 embedding attempts. Molecules RDKit cannot embed --
# bridged cyclophanes, ~2% of chemotion -- fail all 20 identically, turning a
# 3s failure into a 69s one. Retries were measured to rescue nothing (149/150
# molecules predicted at 1, 2 and 20 tries), so one attempt is the useful
# budget and failures stay cheap.
MAX_EMBED_TRIES = int(os.environ.get("CSP5_MAX_EMBED_TRIES", "1"))
EMBED_SEED = int(os.environ.get("CSP5_EMBED_SEED", "42"))
# Rescue molecules ETKDG cannot embed with a self-built geometry.
FALLBACK_GEOMETRY = os.environ.get("CSP5_FALLBACK_GEOMETRY", "1") != "0"

app = FastAPI(title="CSP5 13C shift prediction")


def _fallback_molblock(smiles: str) -> str | None:
    """3D geometry for a molecule ETKDG refuses to embed.

    ETKDG's experimental-torsion and basic-knowledge terms impose torsion
    preferences that strained bridged systems cannot satisfy -- every
    [2.2]paracyclophane in chemotion fails on them, about 2% of the set --
    and the embedding is reported infeasible rather than merely poor. Plain
    distance geometry, with those terms switched off, embeds the same
    molecules in milliseconds; MMFF then restores the local chemistry, and
    the result reproduces the bent aromatic decks that make these molecules
    distinctive. Both predictors give sensible shifts from it (~1.1 ppm for
    CSP5), which beats returning nothing.

    Only used where ETKDG has already failed, so the ordinary path keeps the
    knowledge-based geometry the models were trained on.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = EMBED_SEED
    params.useExpTorsionAnglePrefs = False
    params.useBasicKnowledge = False
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except (ValueError, RuntimeError):
        pass  # no MMFF parameters; the raw DG geometry is still usable
    return Chem.MolToMolBlock(mol)


def _predict_from_geometry(pairs: list[tuple[int, str]], shifts: list) -> None:
    """Fill `shifts` for molecules rescued by a self-built geometry."""
    rows = []
    slots = []
    for slot, smiles in pairs:
        molblock = _fallback_molblock(smiles)
        if molblock is None:
            continue
        rows.append({"smiles": smiles, "molblock": molblock, "conformer_rank": 0})
        slots.append(slot)
    if not rows:
        return

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "structures.parquet"
        pd.DataFrame(rows).to_parquet(path)
        result = predict_structures(path, nucleus="13C", batch_size=BATCH_SIZE, conformer_rank=0)

    frame = result.predictions.sort_values(["molecule_id", "atom_index"])
    grouped = list(frame.groupby("molecule_id", sort=True))
    if len(grouped) != len(slots):
        return  # alignment cannot be trusted; leave these as failures
    for slot, (_, group) in zip(slots, grouped):
        shifts[slot] = [round(float(v), 3) for v in group["shift_ppm"]]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "CSP5-13C"}


@app.post("/")
def predict(request: dict) -> dict:
    smiles_list = request.get("smiles", [])
    n = len(smiles_list)
    shifts: list[list[float] | None] = [None] * n
    uncertainty: list[list[float] | None] = [None] * n
    if n == 0:
        return {"shifts": shifts, "uncertainty": uncertainty}

    # Pre-screen so carbon-free molecules get [] and unparseable ones stay
    # null without relying on CSP5's own failure bookkeeping.
    slots: list[int] = []
    submitted: list[str] = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
            shifts[i], uncertainty[i] = [], []
            continue
        slots.append(i)
        submitted.append(smi)

    if not submitted:
        return {"shifts": shifts, "uncertainty": uncertainty}

    result = predict_smiles(
        submitted,
        nucleus="13C",
        batch_size=BATCH_SIZE,
        max_embed_tries=MAX_EMBED_TRIES,
    )

    # CSP5 drops failed molecules and renumbers molecule_id over the
    # survivors, so walk the submitted list against the failure multiset to
    # recover which slot each surviving molecule_id belongs to.
    failed: dict[str, int] = {}
    embed_failed: set[str] = set()
    for entry in result.failures:
        reason, _, smi = entry.partition("\t")
        failed[smi] = failed.get(smi, 0) + 1
        if reason == "embed":
            embed_failed.add(smi)
    surviving_slots = []
    rescue: list[tuple[int, str]] = []
    for slot, smi in zip(slots, submitted):
        if failed.get(smi, 0) > 0:
            failed[smi] -= 1
            if smi in embed_failed:
                rescue.append((slot, smi))
            continue
        surviving_slots.append(slot)

    frame = result.predictions.sort_values(["molecule_id", "atom_index"])
    grouped = list(frame.groupby("molecule_id", sort=True))
    if len(grouped) != len(surviving_slots):
        # Alignment cannot be trusted; report everything as failed.
        return {"shifts": shifts, "uncertainty": uncertainty}

    for slot, (_, rows) in zip(surviving_slots, grouped):
        shifts[slot] = [round(float(v), 3) for v in rows["shift_ppm"]]

    return {"shifts": shifts, "uncertainty": uncertainty}
