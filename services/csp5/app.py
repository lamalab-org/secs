"""13C shift prediction service wrapping CSP5 (Goodman lab).

Implements the same contract as the cascade and hose services, which
`secs.elucidation.verifiers.HttpShiftSimulator` expects:

    POST /            {"smiles": [...]}  ->  {"shifts": [[...]|null, ...],
                                              "uncertainty": [[...]|null, ...]}

Predictions come back in input order, with null for molecules that could not
be parsed or predicted and [] for molecules without carbon. Shifts are ordered
by RDKit atom index over the carbon atoms, matching the other services.

Work is sharded across a process pool: conformer embedding is CPU-bound and
serial per molecule, and CSP5's own `num_workers` is unusable (it raises
`'generator' object has no attribute 'next'` in 0.2.18), so one request would
otherwise pin a single core while a GA waits on thousands of candidates.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import torch
from fastapi import FastAPI
from rdkit import Chem, RDLogger

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
WORKERS = int(os.environ.get("CSP5_WORKERS", "8"))
# Below this, pool dispatch costs more than it saves.
MIN_SHARD = int(os.environ.get("CSP5_MIN_SHARD", "8"))
# Work unit handed to a worker; small enough to load-balance.
CHUNK = int(os.environ.get("CSP5_CHUNK", "32"))

_POOL: ProcessPoolExecutor | None = None


def _init_worker() -> None:
    """One model per worker, and one thread each.

    Torch would otherwise start a full thread pool per process and the
    workers would fight over the same cores, which is slower than serial.
    """

    torch.set_num_threads(1)
    from csp5 import predict_smiles

    predict_smiles(["CCO"], nucleus="13C", max_embed_tries=1)  # warm the weights


def _pool() -> ProcessPoolExecutor | None:
    global _POOL
    if WORKERS <= 1:
        return None
    if _POOL is None:
        # spawn: torch and fork in a threaded server deadlock.
        import multiprocessing

        _POOL = ProcessPoolExecutor(
            max_workers=WORKERS,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
        )
    return _POOL


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
    """
    from rdkit.Chem import AllChem

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


def _predict_chunk(smiles: list[str]) -> list[list[float] | None]:
    """Shifts for one shard, aligned to its own input order.

    Alignment happens here rather than in the parent so a shard is entirely
    self-contained: CSP5 drops failed molecules and renumbers molecule_id
    over the survivors, and that bookkeeping only makes sense next to the
    call that produced it.
    """
    import tempfile
    from pathlib import Path

    import pandas as pd
    from csp5 import predict_smiles, predict_structures

    out: list[list[float] | None] = [None] * len(smiles)
    result = predict_smiles(smiles, nucleus="13C", batch_size=BATCH_SIZE, max_embed_tries=MAX_EMBED_TRIES)

    failed: dict[str, int] = {}
    embed_failed: set[str] = set()
    for entry in result.failures:
        reason, _, smi = entry.partition("\t")
        failed[smi] = failed.get(smi, 0) + 1
        if reason == "embed":
            embed_failed.add(smi)

    surviving, rescue = [], []
    for slot, smi in enumerate(smiles):
        if failed.get(smi, 0) > 0:
            failed[smi] -= 1
            if smi in embed_failed:
                rescue.append((slot, smi))
            continue
        surviving.append(slot)

    frame = result.predictions.sort_values(["molecule_id", "atom_index"])
    grouped = list(frame.groupby("molecule_id", sort=True))
    if len(grouped) == len(surviving):
        for slot, (_, rows) in zip(surviving, grouped):
            out[slot] = [round(float(v), 3) for v in rows["shift_ppm"]]

    if rescue and FALLBACK_GEOMETRY:
        rows, slots = [], []
        for slot, smi in rescue:
            molblock = _fallback_molblock(smi)
            if molblock is None:
                continue
            rows.append({"smiles": smi, "molblock": molblock, "conformer_rank": 0})
            slots.append(slot)
        if rows:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "structures.parquet"
                pd.DataFrame(rows).to_parquet(path)
                rescued = predict_structures(path, nucleus="13C", batch_size=BATCH_SIZE, conformer_rank=0)
            frame = rescued.predictions.sort_values(["molecule_id", "atom_index"])
            grouped = list(frame.groupby("molecule_id", sort=True))
            if len(grouped) == len(slots):
                for slot, (_, group) in zip(slots, grouped):
                    out[slot] = [round(float(v), 3) for v in group["shift_ppm"]]
    return out


app = FastAPI(title="CSP5 13C shift prediction")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "CSP5-13C", "workers": WORKERS}


@app.post("/")
def predict(request: dict) -> dict:
    smiles_list = request.get("smiles", [])
    n = len(smiles_list)
    shifts: list[list[float] | None] = [None] * n
    uncertainty: list[list[float] | None] = [None] * n
    if n == 0:
        return {"shifts": shifts, "uncertainty": uncertainty}

    # Pre-screen so carbon-free molecules get [] and unparseable ones stay
    # null without a worker round trip.
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

    pool = _pool()
    if pool is None or len(submitted) < MIN_SHARD:
        predicted = _predict_chunk(submitted)
    else:
        # Many small chunks rather than one per worker: per-molecule cost
        # varies by an order of magnitude (embedding dominates, and hard
        # scaffolds take seconds), so equal shards leave workers idle while
        # one grinds. The pool hands out the next chunk as each frees up.
        size = max(1, min(CHUNK, len(submitted) // WORKERS or 1))
        chunks = [submitted[i : i + size] for i in range(0, len(submitted), size)]
        predicted = []
        for part in pool.map(_predict_chunk, [c for c in chunks if c]):
            predicted.extend(part)

    for slot, values in zip(slots, predicted):
        shifts[slot] = values

    return {"shifts": shifts, "uncertainty": uncertainty}
