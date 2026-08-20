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
"""

import os

from csp5 import predict_smiles
from fastapi import FastAPI
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

BATCH_SIZE = int(os.environ.get("CSP5_BATCH_SIZE", "64"))

app = FastAPI(title="CSP5 13C shift prediction")


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

    result = predict_smiles(submitted, nucleus="13C", batch_size=BATCH_SIZE)

    # CSP5 drops failed molecules and renumbers molecule_id over the
    # survivors, so walk the submitted list against the failure multiset to
    # recover which slot each surviving molecule_id belongs to.
    failed: dict[str, int] = {}
    for entry in result.failures:
        smi = entry.split("\t", 1)[-1]
        failed[smi] = failed.get(smi, 0) + 1
    surviving_slots = []
    for slot, smi in zip(slots, submitted):
        if failed.get(smi, 0) > 0:
            failed[smi] -= 1
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
