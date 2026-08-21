"""13C shift prediction service wrapping the HOSE lookup table.

Implements the same contract as the cascade and csp5 services, so
`secs.elucidation.verifiers.HttpShiftSimulator` can point at any of them:

    POST /            {"smiles": [...]}  ->  {"shifts": [[...]|null, ...],
                                              "uncertainty": [[...]|null, ...]}

The lookup module is copied verbatim from
`src/secs/elucidation/verifiers/hose.py` at build time rather than
reimplemented here, so the service and the in-process simulator cannot drift
apart. Installing the `secs` package instead would pull torch and faiss in
for what is a hash lookup. The table itself is data: it is built by
`scripts/build_hose_table.py` and mounted at runtime.

Coverage is deliberately visible. A HOSE table returns nothing for an
environment it has never seen, so the response reports how many carbons were
actually predicted alongside the shifts -- a caller comparing MAEs against a
neural predictor needs to know that the easy carbons may be the only ones
answered.
"""

import os
from pathlib import Path

from fastapi import FastAPI
from hose import HoseShiftTable
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

TABLE_PATH = Path(os.environ.get("HOSE_TABLE_PATH", "/app/hose_table.json"))
MIN_COUNT = os.environ.get("HOSE_MIN_COUNT")

TABLE = HoseShiftTable.load(TABLE_PATH)
if MIN_COUNT is not None:
    TABLE.min_count = int(MIN_COUNT)

app = FastAPI(title="HOSE 13C shift prediction")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "HOSE lookup table",
        "entries": len(TABLE.table),
        "min_count": TABLE.min_count,
    }


@app.post("/")
def predict(request: dict) -> dict:
    smiles_list = request.get("smiles", [])
    n = len(smiles_list)
    shifts: list[list[float] | None] = [None] * n
    uncertainty: list[list[float] | None] = [None] * n
    n_carbons: list[int] = [0] * n

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
        n_carbons[i] = len(carbons)
        if not carbons:
            shifts[i], uncertainty[i] = [], []
            continue
        # predict_atom rather than predict(), so an unmatched carbon is a
        # reported gap instead of a silently shorter array.
        values = []
        for index in carbons:
            value, _ = TABLE.predict_atom(mol, index)
            if value is not None:
                values.append(round(float(value), 3))
        shifts[i] = values if values else None

    predicted = [0 if s is None else len(s) for s in shifts]
    return {
        "shifts": shifts,
        "uncertainty": uncertainty,
        "n_carbons": n_carbons,
        "n_predicted": predicted,
    }
