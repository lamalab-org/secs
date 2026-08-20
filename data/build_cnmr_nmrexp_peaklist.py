import ast
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from rdkit import Chem
from tqdm import tqdm

# =============================================================================
# Config
# =============================================================================
X_MIN, X_MAX = 0.0, 218.0
SEED = 42

SOLVENT_JITTER_PPM = 0.2
IMPURITY_JITTER_PPM = 0.1

PARQUET_PATH = "NMRexp_10to24_1_1004.parquet"
N_SAMPLES = 1_500_000
VAL_FRAC = 0.01
HF_REPO = "jablonkagroup/nmrexp-cnmr-peaklist-1.5M"
N_PROC = max(1, cpu_count() - 1)

# --- per-spectrum dirtiness regimes -----------------------------------------
# name: (probability, impurity-prob multiplier, spurious Poisson-mean range,
#        drop-prob range, jitter-sigma range)
REGIMES = {
    "clean":   (0.20, 0.0, (0.0, 0.3),  (0.00, 0.05), (0.01, 0.04)),
    "typical": (0.55, 1.0, (0.0, 1.5),  (0.02, 0.12), (0.02, 0.08)),
    "dirty":   (0.25, 2.5, (4.0, 15.0), (0.05, 0.20), (0.04, 0.12)),
}

AUG_MERGE_TOL_RANGE = (0.05, 0.25)
AUG_SPLIT_PROB = 0.05
AUG_SPLIT_OFFSET_PPM = 0.4
AUG_SOLVENT_DROP_PROB = 0.10
AUG_QUAT_EXTRA_DROP = 0.15
AUG_CROWD_WINDOW = 2.0
AUG_CROWD_EXTRA_DROP = 0.05

# CDCl3 etc. couple to D, so an uncollapsed residual shows as a multiplet.
# Spacing is J/field; field is unknown here, so it is drawn from a range that
# covers 100-200 MHz carbon frequencies.
SOLVENT_MULTIPLET_PROB = 0.35
SOLVENT_MULTIPLET = {  # solvent -> (n_lines, spacing_ppm_range)
    "CDCl3": (3, (0.16, 0.32)),
    "DMSO-d6": (7, (0.10, 0.21)),
    "CD3OD": (7, (0.11, 0.22)),
    "acetone-d6": (7, (0.10, 0.20)),
    "C6D6": (3, (0.12, 0.24)),
    "CD2Cl2": (5, (0.14, 0.27)),
}

# =============================================================================
# Reference shifts (Gottlieb 1997; Fulmer 2010)
# =============================================================================
SOLVENT_RESIDUALS = {
    "CDCl3": [77.16],
    "DMSO-d6": [39.52],
    "CD3OD": [49.00],
    "C6D6": [128.06],
    "CD2Cl2": [53.84],
    "acetone-d6": [29.84, 206.26],
    "D2O": [],
}

IMPURITIES = {
    "tms": {
        "prob": 0.5,
        "shifts": {s: [0.00] for s in ["CDCl3", "DMSO-d6", "CD3OD", "C6D6", "CD2Cl2", "acetone-d6"]},
    },
    "silicone_grease": {
        "prob": 0.25,
        "shifts": {"CDCl3": [1.19], "CD2Cl2": [1.22], "acetone-d6": [1.40], "C6D6": [1.38]},
    },
    "ethyl_acetate": {
        "prob": 0.15,
        "shifts": {
            "CDCl3": [171.36, 60.49, 21.04, 14.19],
            "acetone-d6": [170.96, 60.56, 20.83, 14.50],
            "DMSO-d6": [170.31, 59.74, 20.68, 14.40],
            "C6D6": [170.44, 60.21, 20.56, 14.19],
            "CD3OD": [172.89, 61.50, 20.88, 14.49],
            "D2O": [175.26, 62.32, 21.15, 13.92],
            "CD2Cl2": [171.24, 60.63, 21.15, 14.37],
        },
    },
    "methanol": {
        "prob": 0.12,
        "shifts": {
            "CDCl3": [50.41], "acetone-d6": [49.77], "DMSO-d6": [48.59], "C6D6": [49.97],
            "CD3OD": [49.86], "D2O": [49.50], "CD2Cl2": [50.45],
        },
    },
    "ethanol": {
        "prob": 0.1,
        "shifts": {
            "CDCl3": [58.28, 18.41], "acetone-d6": [57.72, 18.89], "DMSO-d6": [56.07, 18.51],
            "C6D6": [57.86, 18.72], "CD3OD": [58.26, 18.40], "D2O": [58.05, 17.47],
            "CD2Cl2": [58.57, 18.69],
        },
    },
    "acetone": {
        "prob": 0.12,
        "shifts": {
            "CDCl3": [207.07, 30.92], "acetone-d6": [205.87, 30.60], "DMSO-d6": [206.31, 30.56],
            "C6D6": [204.43, 30.14], "CD3OD": [209.67, 30.67], "D2O": [215.94, 30.89],
            "CD2Cl2": [206.78, 31.00],
        },
    },
    "dcm": {
        "prob": 0.1,
        "shifts": {
            "CDCl3": [53.52], "acetone-d6": [54.95], "DMSO-d6": [54.84], "C6D6": [53.46],
            "CD3OD": [54.78], "CD2Cl2": [54.24],
        },
    },
    "hexane": {
        "prob": 0.1,
        "shifts": {
            "CDCl3": [31.64, 22.70, 14.14], "acetone-d6": [32.30, 23.28, 14.34],
            "DMSO-d6": [30.95, 22.05, 13.88], "C6D6": [31.96, 23.04, 14.32],
            "CD3OD": [32.73, 23.68, 14.45], "CD2Cl2": [32.01, 23.07, 14.28],
        },
    },
    "thf": {
        "prob": 0.08,
        "shifts": {
            "CDCl3": [67.97, 25.62], "acetone-d6": [68.07, 26.15], "DMSO-d6": [67.03, 25.14],
            "C6D6": [67.80, 25.72], "CD3OD": [68.83, 26.48], "D2O": [68.68, 25.67],
            "CD2Cl2": [68.16, 25.98],
        },
    },
    "diethyl_ether": {
        "prob": 0.08,
        "shifts": {
            "CDCl3": [65.91, 15.20], "acetone-d6": [66.12, 15.78], "DMSO-d6": [62.05, 15.12],
            "C6D6": [65.94, 15.46], "CD3OD": [66.88, 15.46], "D2O": [66.42, 14.77],
            "CD2Cl2": [66.11, 15.44],
        },
    },
    "dmf": {
        "prob": 0.06,
        "shifts": {
            "CDCl3": [162.62, 36.50, 31.45], "acetone-d6": [162.79, 36.15, 31.03],
            "DMSO-d6": [162.29, 35.73, 30.73], "C6D6": [162.13, 35.25, 30.72],
            "CD3OD": [164.73, 36.89, 31.61], "D2O": [165.53, 37.54, 32.03],
            "CD2Cl2": [162.57, 36.56, 31.39],
        },
    },
    "dmso": {
        "prob": 0.06,
        "shifts": {
            "CDCl3": [40.76], "acetone-d6": [41.23], "DMSO-d6": [40.45], "C6D6": [40.03],
            "CD3OD": [40.45], "D2O": [39.39],
        },
    },
    "toluene": {
        "prob": 0.05,
        "shifts": {
            "CDCl3": [137.89, 129.07, 128.26, 125.33, 21.46],
            "acetone-d6": [138.48, 129.76, 129.03, 126.12, 21.46],
            "DMSO-d6": [137.35, 128.88, 128.18, 125.29, 20.99],
            "C6D6": [137.91, 129.33, 128.56, 125.68, 21.10],
            "CD3OD": [138.85, 129.91, 129.20, 126.29, 21.50],
            "CD2Cl2": [138.36, 129.35, 128.54, 125.62, 21.53],
        },
    },
    "dioxane": {
        "prob": 0.04,
        "shifts": {
            "CDCl3": [67.14], "acetone-d6": [67.60], "DMSO-d6": [66.36], "C6D6": [67.16],
            "CD3OD": [68.11], "D2O": [67.19], "CD2Cl2": [67.47],
        },
    },
    "acetic_acid": {
        "prob": 0.04,
        "shifts": {
            "CDCl3": [175.99, 20.81], "acetone-d6": [172.31, 20.51], "DMSO-d6": [171.93, 20.95],
            "C6D6": [175.82, 20.37], "CD3OD": [175.11, 20.56], "D2O": [177.21, 21.03],
            "CD2Cl2": [175.85, 20.91],
        },
    },
    "isopropanol": {
        "prob": 0.04,
        "shifts": {
            "CDCl3": [64.50, 25.14], "acetone-d6": [63.85, 25.67], "DMSO-d6": [64.92, 25.43],
            "C6D6": [64.23, 25.18], "CD3OD": [64.71, 25.27], "D2O": [64.88, 24.38],
            "CD2Cl2": [64.67, 25.43],
        },
    },
}

# Long-chain grease: terminal CH3, CH3-adjacent CH2, omega-2 CH2, CH2 envelope.
GREASE_ANCHORS = {
    "CDCl3": [14.1, 22.7, 31.9, 29.7],
    "acetone-d6": [14.3, 23.3, 32.3, 30.7],
    "DMSO-d6": [13.9, 22.1, 31.2, 29.2],
    "C6D6": [14.3, 23.1, 32.2, 30.2],
    "CD3OD": [14.5, 23.7, 33.1, 31.3],
    "CD2Cl2": [14.3, 23.1, 32.3, 30.1],
}
GREASE_PROB = 0.4


# =============================================================================
# Helpers
# =============================================================================
def canonicalize_smiles(smi):
    if smi is None:
        return None
    mol = Chem.MolFromSmiles(smi)
    return None if mol is None else Chem.MolToSmiles(mol)


def in_range(s):
    return X_MIN <= s < X_MAX


def merge_close(shifts, tol):
    if not shifts:
        return shifts
    shifts = sorted(shifts)
    merged, group = [], [shifts[0]]
    for s in shifts[1:]:
        if s - group[-1] < tol:
            group.append(s)
        else:
            merged.append(float(np.mean(group)))
            group = [s]
    merged.append(float(np.mean(group)))
    return merged


def pick_regime(rng):
    names = list(REGIMES)
    probs = [REGIMES[n][0] for n in names]
    return REGIMES[rng.choice(names, p=probs)]


# =============================================================================
# Augmentation of analyte peaks
# =============================================================================
def augment_analyte(shifts, mults, rng, drop_range, sigma_range):
    drop_p = rng.uniform(*drop_range)
    sigma = rng.uniform(*sigma_range)

    order = np.argsort(shifts)
    shifts = [shifts[i] for i in order]
    mults = [mults[i] for i in order]

    out = []
    for i, s in enumerate(shifts):
        p = drop_p
        n_nb = sum(1 for j, t in enumerate(shifts) if j != i and abs(t - s) < AUG_CROWD_WINDOW)
        p += min(n_nb * AUG_CROWD_EXTRA_DROP, 0.25)
        if mults[i] == "s":
            p += AUG_QUAT_EXTRA_DROP
        if rng.random() < min(p, 0.9):
            continue

        s2 = s + rng.normal(0.0, sigma)
        if in_range(s2):
            out.append(float(s2))
        if rng.random() < AUG_SPLIT_PROB:
            twin = s2 + rng.choice([-1.0, 1.0]) * AUG_SPLIT_OFFSET_PPM
            if in_range(twin):
                out.append(float(twin))
    return out


# =============================================================================
# Contaminant generators
# =============================================================================
def solvent_peaks(solvent, rng, augment):
    out = []
    for res in SOLVENT_RESIDUALS.get(solvent, []):
        if augment and rng.random() < AUG_SOLVENT_DROP_PROB:
            continue
        centre = res + float(rng.uniform(-SOLVENT_JITTER_PPM, SOLVENT_JITTER_PPM))
        mult = SOLVENT_MULTIPLET.get(solvent)
        if augment and mult is not None and rng.random() < SOLVENT_MULTIPLET_PROB:
            n, sp_range = mult
            sp = rng.uniform(*sp_range)
            lines = [centre + (k - (n - 1) / 2) * sp for k in range(n)]
        else:
            lines = [centre]
        out.extend(s for s in lines if in_range(s))
    return out


def impurity_peaks(solvent, rng, prob_mult):
    out = []
    for entry in IMPURITIES.values():
        table = entry["shifts"]
        if solvent not in table:
            continue
        p = entry["prob"] * prob_mult if prob_mult != 1.0 else entry["prob"]
        if rng.random() >= min(p, 0.95):
            continue
        for s in table[solvent]:
            s2 = s + float(rng.uniform(-IMPURITY_JITTER_PPM, IMPURITY_JITTER_PPM))
            if in_range(s2):
                out.append(float(s2))
    return out


def grease_peaks(solvent, rng, prob_mult, dirty):
    anchors = GREASE_ANCHORS.get(solvent)
    if anchors is None or rng.random() >= min(GREASE_PROB * max(prob_mult, 0.0), 0.95):
        return []
    out = []
    ch3, ch2_a, ch2_b, envelope = anchors
    base = [ch3, envelope] if not dirty else [ch3, ch2_a, ch2_b, envelope]
    for s in base:
        out.append(s + float(rng.uniform(-IMPURITY_JITTER_PPM, IMPURITY_JITTER_PPM)))
    if dirty:
        for _ in range(int(rng.integers(1, 5))):
            out.append(envelope + float(rng.normal(0.0, 0.5)))
    return [s for s in out if in_range(s)]


def spurious_peaks(existing, rng, rate_range):
    rate = rng.uniform(*rate_range)
    out = []
    for _ in range(int(rng.poisson(rate))):
        u = rng.random()
        if u < 0.15 or not existing:
            cand = rng.uniform(0.0, 5.0)
        elif u < 0.40:
            cand = rng.uniform(X_MIN, X_MAX)
        else:
            cand = existing[rng.integers(len(existing))] + rng.normal(0.0, 3.0)
        if in_range(cand):
            out.append(float(cand))
    return out


# =============================================================================
# Assembly
# =============================================================================
def peaks_to_shifts(peaks, solvent, rng):
    if isinstance(peaks, str):
        peaks = ast.literal_eval(peaks)

    shifts, mults = [], []
    for s, mult, _J in peaks:
        if s is not None and in_range(float(s)):
            shifts.append(float(s))
            mults.append(mult)

    _, prob_mult, spur_range, drop_range, sigma_range = pick_regime(rng)
    dirty = prob_mult > 1.0

    out = augment_analyte(shifts, mults, rng, drop_range, sigma_range)
    out += solvent_peaks(solvent, rng, augment=True)
    out += impurity_peaks(solvent, rng, prob_mult)
    out += grease_peaks(solvent, rng, prob_mult, dirty)

    out = merge_close(out, rng.uniform(*AUG_MERGE_TOL_RANGE))
    out += spurious_peaks(out, rng, spur_range)
    return sorted(out)


def process_row(args):
    peaks, solvent, smiles, seed = args
    try:
        canon = canonicalize_smiles(smiles)
        if canon is None:
            return None
        rng = np.random.default_rng(seed)
        shifts = peaks_to_shifts(peaks, solvent, rng)
        if not shifts:
            return None
        return {"smiles": canon, "c_nmr": shifts, "x_min": float(X_MIN), "x_max": float(X_MAX)}
    except Exception:
        return None


# =============================================================================
# Build
# =============================================================================
def load_cnmr():
    data = pd.read_parquet(PARQUET_PATH)
    cnmr = data[data.NMR_type == "13C NMR"]
    cnmr = cnmr[~cnmr.NMR_solvent.isin(["not_known", "mixed"])]
    return cnmr.reset_index(drop=True)


def build():
    cnmr = load_cnmr()
    print(f"filtered 13C rows available: {len(cnmr)}")
    if len(cnmr) > N_SAMPLES:
        cnmr = cnmr.sample(n=N_SAMPLES, random_state=SEED).reset_index(drop=True)

    tasks = [
        (p, s, sm, SEED + i)
        for i, (p, s, sm) in enumerate(zip(cnmr.NMR_processed, cnmr.NMR_solvent, cnmr.SMILES))
    ]

    rows = []
    with Pool(processes=N_PROC) as pool:
        for res in tqdm(pool.imap_unordered(process_row, tasks, chunksize=256), total=len(tasks), desc="processing"):
            if res is not None:
                rows.append(res)

    out = pd.DataFrame(rows)
    n_before = len(out)
    out = out.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)
    print(f"built {n_before} | after dedup {len(out)}")

    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_val = int(len(out) * VAL_FRAC)
    val_df, train_df = out.iloc[:n_val], out.iloc[n_val:]
    print(f"train {len(train_df)} | val {len(val_df)}")

    ds = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df.reset_index(drop=True), preserve_index=False),
            "val": Dataset.from_pandas(val_df.reset_index(drop=True), preserve_index=False),
        }
    )
    ds.push_to_hub(HF_REPO, private=False)
    print(f"pushed to {HF_REPO}")


if __name__ == "__main__":
    build()