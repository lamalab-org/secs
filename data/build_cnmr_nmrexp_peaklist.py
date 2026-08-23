"""Build a 13C peak-list dataset from the NMRexp literature corpus.

A spectrum is the molecule's published shifts plus whatever else was in the
tube: the deuterated solvent's residual and a few common impurities. Nothing
else -- no synthetic trace, no peak picking, no corruption model.

    python build_cnmr_nmrexp_peaklist.py --limit 20000 --out sample.parquet
    python build_cnmr_nmrexp_peaklist.py --push
"""

import argparse
import ast
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

from datasets import Dataset, DatasetDict

RDLogger.DisableLog("rdApp.*")

SEED = 42
N_PROC = max(1, cpu_count() - 1)
X_MIN, X_MAX = 0.0, 218.0
PARQUET_PATH = "NMRexp_10to24_1_1004.parquet"
VAL_FRAC = 0.01
HF_REPO = "jablonkagroup/nmrexp-cnmr-peaklist-1.5M"

# --- solvents ---------------------------------------------------------------
# Residual 13C of the deuterated solvent (Gottlieb 1997; Fulmer 2010), as the
# collapsed centre. Present in nearly every spectrum.
SOLVENT_PEAKS = {
    "CDCl3": [77.16], "DMSO-d6": [39.52], "CD3OD": [49.00], "C6D6": [128.06],
    "CD2Cl2": [53.84], "acetone-d6": [29.84, 206.26], "CD3CN": [1.32, 118.26],
    "THF-d8": [25.31, 67.21], "pyridine-d5": [123.87, 135.91, 150.35],
    "toluene-d8": [20.43, 125.13, 127.96, 128.87, 137.86],
    "DMF-d7": [29.76, 34.89, 163.15], "D2O": [],
}
SOLVENT_PRESENT_PROB = 0.90

# NMRexp writes solvents as formulae; the reference tables name them as Gottlieb
# does. Without this map the acetone rows silently get no residual at all.
_SOLVENT_KEY = {
    "cdcl3": "CDCl3", "chloroformd": "CDCl3",
    "dmsod6": "DMSO-d6", "d6dmso": "DMSO-d6", "cd3socd3": "DMSO-d6",
    "cd3od": "CD3OD", "meod": "CD3OD", "methanold4": "CD3OD",
    "c6d6": "C6D6", "benzened6": "C6D6", "cd2cl2": "CD2Cl2",
    "cd3cocd3": "acetone-d6", "acetoned6": "acetone-d6", "d6acetone": "acetone-d6",
    "cd3cn": "CD3CN", "acetonitriled3": "CD3CN",
    "thfd8": "THF-d8", "d8thf": "THF-d8",
    "pyridined5": "pyridine-d5", "c5d5n": "pyridine-d5",
    "phmed8": "toluene-d8", "toluened8": "toluene-d8", "cd3c6d5": "toluene-d8",
    "dmfd7": "DMF-d7", "d2o": "D2O",
}

# --- impurities -------------------------------------------------------------
# name -> (probability, {solvent: shifts}). A contaminant contributes all of its
# carbons or none of them, so each entry is drawn as a whole molecule.
IMPURITIES = {
    "tms": (0.50, {s: [0.00] for s in ["CDCl3", "DMSO-d6", "CD3OD", "C6D6", "CD2Cl2", "acetone-d6"]}),
    "silicone_grease": (0.25, {"CDCl3": [1.19], "CD2Cl2": [1.22], "acetone-d6": [1.40], "C6D6": [1.38]}),
    "ethyl_acetate": (0.15, {
        "CDCl3": [171.36, 60.49, 21.04, 14.19], "acetone-d6": [170.96, 60.56, 20.83, 14.50],
        "DMSO-d6": [170.31, 59.74, 20.68, 14.40], "C6D6": [170.44, 60.21, 20.56, 14.19],
        "CD3OD": [172.89, 61.50, 20.88, 14.49], "D2O": [175.26, 62.32, 21.15, 13.92],
        "CD2Cl2": [171.24, 60.63, 21.15, 14.37]}),
    "methanol": (0.12, {
        "CDCl3": [50.41], "acetone-d6": [49.77], "DMSO-d6": [48.59], "C6D6": [49.97],
        "CD3OD": [49.86], "D2O": [49.50], "CD2Cl2": [50.45]}),
    "acetone": (0.12, {
        "CDCl3": [207.07, 30.92], "acetone-d6": [205.87, 30.60], "DMSO-d6": [206.31, 30.56],
        "C6D6": [204.43, 30.14], "CD3OD": [209.67, 30.67], "D2O": [215.94, 30.89],
        "CD2Cl2": [206.78, 31.00]}),
    "ethanol": (0.10, {
        "CDCl3": [58.28, 18.41], "acetone-d6": [57.72, 18.89], "DMSO-d6": [56.07, 18.51],
        "C6D6": [57.86, 18.72], "CD3OD": [58.26, 18.40], "D2O": [58.05, 17.47],
        "CD2Cl2": [58.57, 18.69]}),
    "dcm": (0.10, {
        "CDCl3": [53.52], "acetone-d6": [54.95], "DMSO-d6": [54.84], "C6D6": [53.46],
        "CD3OD": [54.78], "CD2Cl2": [54.24]}),
    "hexane": (0.10, {
        "CDCl3": [31.64, 22.70, 14.14], "acetone-d6": [32.30, 23.28, 14.34],
        "DMSO-d6": [30.95, 22.05, 13.88], "C6D6": [31.96, 23.04, 14.32],
        "CD3OD": [32.73, 23.68, 14.45], "CD2Cl2": [32.01, 23.07, 14.28]}),
    "thf": (0.08, {
        "CDCl3": [67.97, 25.62], "acetone-d6": [68.07, 26.15], "DMSO-d6": [67.03, 25.14],
        "C6D6": [67.80, 25.72], "CD3OD": [68.83, 26.48], "D2O": [68.68, 25.67],
        "CD2Cl2": [68.16, 25.98]}),
    "diethyl_ether": (0.08, {
        "CDCl3": [65.91, 15.20], "acetone-d6": [66.12, 15.78], "DMSO-d6": [62.05, 15.12],
        "C6D6": [65.94, 15.46], "CD3OD": [66.88, 15.46], "D2O": [66.42, 14.77],
        "CD2Cl2": [66.11, 15.44]}),
    "dmf": (0.06, {
        "CDCl3": [162.62, 36.50, 31.45], "acetone-d6": [162.79, 36.15, 31.03],
        "DMSO-d6": [162.29, 35.73, 30.73], "C6D6": [162.13, 35.25, 30.72],
        "CD3OD": [164.73, 36.89, 31.61], "D2O": [165.53, 37.54, 32.03],
        "CD2Cl2": [162.57, 36.56, 31.39]}),
    "dmso": (0.06, {
        "CDCl3": [40.76], "acetone-d6": [41.23], "DMSO-d6": [40.45], "C6D6": [40.03],
        "CD3OD": [40.45], "D2O": [39.39]}),
    "toluene": (0.05, {
        "CDCl3": [137.89, 129.07, 128.26, 125.33, 21.46],
        "acetone-d6": [138.48, 129.76, 129.03, 126.12, 21.46],
        "DMSO-d6": [137.35, 128.88, 128.18, 125.29, 20.99],
        "C6D6": [137.91, 129.33, 128.56, 125.68, 21.10],
        "CD3OD": [138.85, 129.91, 129.20, 126.29, 21.50],
        "CD2Cl2": [138.36, 129.35, 128.54, 125.62, 21.53]}),
    "dioxane": (0.04, {
        "CDCl3": [67.14], "acetone-d6": [67.60], "DMSO-d6": [66.36], "C6D6": [67.16],
        "CD3OD": [68.11], "D2O": [67.19], "CD2Cl2": [67.47]}),
    "acetic_acid": (0.04, {
        "CDCl3": [175.99, 20.81], "acetone-d6": [172.31, 20.51], "DMSO-d6": [171.93, 20.95],
        "C6D6": [175.82, 20.37], "CD3OD": [175.11, 20.56], "D2O": [177.21, 21.03],
        "CD2Cl2": [175.85, 20.91]}),
    "isopropanol": (0.04, {
        "CDCl3": [64.50, 25.14], "acetone-d6": [63.85, 25.67], "DMSO-d6": [64.92, 25.43],
        "C6D6": [64.23, 25.18], "CD3OD": [64.71, 25.27], "D2O": [64.88, 24.38],
        "CD2Cl2": [64.67, 25.43]}),
}


def canonical_solvent(name):
    """NMRexp solvent string -> a key in the reference tables, or None."""
    key = "".join(ch for ch in str(name or "").lower() if ch.isalnum())
    return _SOLVENT_KEY.get(key)


def molecule_peaks(raw):
    """NMR_processed -> the published shifts, in ppm.

    A shift stored as a two-element list is a multiplet reported as an
    interval; its midpoint is the line.
    """
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    out = []
    for entry in raw or []:
        if entry is None or len(entry) != 3:
            continue
        s = entry[0]
        if isinstance(s, (list, tuple, np.ndarray)):
            vals = [float(v) for v in s if v is not None]
            s = float(np.mean(vals)) if vals else None
        try:
            ppm = float(s)
        except (TypeError, ValueError):
            continue
        if X_MIN <= ppm < X_MAX:
            out.append(ppm)
    return out


def contaminant_peaks(solvent, rng):
    """Residual solvent plus whichever impurities this tube happened to have."""
    peaks = []
    if solvent and rng.random() < SOLVENT_PRESENT_PROB:
        peaks += SOLVENT_PEAKS.get(solvent, [])
    for prob, shifts in IMPURITIES.values():
        if rng.random() < prob:
            peaks += shifts.get(solvent, [])
    return peaks


def process_row(args):
    """One source record -> one row, or None if it cannot be one."""
    peaks, solvent, smiles, seed = args
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or not any(a.GetSymbol() == "C" for a in mol.GetAtoms()):
        return None
    shifts = molecule_peaks(peaks)
    if not shifts:  # a row with no molecular signal is not a training example
        return None
    solvent = canonical_solvent(solvent)
    shifts += contaminant_peaks(solvent, np.random.default_rng(seed))
    shifts = sorted({round(s, 2) for s in shifts if X_MIN <= s <= X_MAX})
    return {"smiles": Chem.MolToSmiles(mol), "c_nmr": shifts}


def build(limit=None, out_path=None, push=False, n_proc=None):
    data = pd.read_parquet(PARQUET_PATH)
    cnmr = data[(data.NMR_type == "13C NMR") & ~data.NMR_solvent.isin(["not_known", "mixed"])]
    cnmr = cnmr.reset_index(drop=True)
    if limit is not None and len(cnmr) > limit:
        cnmr = cnmr.sample(n=limit, random_state=SEED).reset_index(drop=True)
    print(f"filtered 13C rows: {len(cnmr)}")

    cols = zip(cnmr.NMR_processed, cnmr.NMR_solvent, cnmr.SMILES, strict=True)
    tasks = [(p, s, sm, SEED + i) for i, (p, s, sm) in enumerate(cols)]
    # imap, not imap_unordered: the dedup below keeps the first row per SMILES,
    # so the order the workers finish in would otherwise decide the output.
    with Pool(processes=n_proc or N_PROC) as pool:
        rows = list(tqdm(pool.imap(process_row, tasks, chunksize=256), total=len(tasks)))
    out = pd.DataFrame([r for r in rows if r is not None])
    out = out.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)

    counts = out.c_nmr.map(len)
    print(f"rows {len(out)} | peaks per spectrum: mean {counts.mean():.1f} median {counts.median():.0f} max {counts.max()}")

    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_val = int(len(out) * VAL_FRAC)
    val_df, train_df = out.iloc[:n_val], out.iloc[n_val:]
    print(f"train {len(train_df)} | val {len(val_df)}")

    if out_path:
        out.to_parquet(out_path, index=False)
        print(f"saved -> {out_path}")

    DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True), preserve_index=False),
        "val": Dataset.from_pandas(val_df.reset_index(drop=True), preserve_index=False),
    }).push_to_hub(HF_REPO, private=False)
    print(f"pushed to {HF_REPO}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="cap the number of source spectra")
    ap.add_argument("--out", default=None, help="write the result to this parquet")
    ap.add_argument("--push", action="store_true", help=f"push to {HF_REPO}")
    ap.add_argument("--n-proc", type=int, default=None, help=f"worker processes (default {N_PROC})")
    args = ap.parse_args()
    build(limit=args.limit, out_path=args.out, push=args.push, n_proc=args.n_proc)
