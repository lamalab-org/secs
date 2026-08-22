"""Build a 13C peak-list dataset from the NMRexp literature corpus.

The point of this script is that it does NOT invent a corruption model.

Downstream the model is handed peak lists extracted from *raw* spectra by
`build_cnmr_chemotion_peaklist.pick_peak_shifts` -- rolling-min baseline, a
global 8-sigma threshold, prominence-gated `find_peaks`, intensity-weighted
merging at `merge_hz`/field. Those steps have failure modes (weak peaks lost at
the threshold, shoulders swallowed by the prominence gate, merge centroids
pulled toward the taller line, tolerance scaling with field) that no
hand-written list of drop/jitter/spurious probabilities reproduces.

So instead of perturbing the published peak list, we go the long way round:

    published shifts + J + field + solvent
        -> resolved lines (J multiplets expanded at the recorded field)
        -> line intensities (quaternary vs protonated, symmetry degeneracy)
        -> a synthetic 1D trace (Lorentzians + baseline drift + noise at a
           sampled SNR)
        -> pick_peak_shifts   <-- THE SAME FUNCTION USED ON REAL SPECTRA
        -> peak list

Training and test peak lists are then draws from one operator by construction.
The dirtiness knobs that remain are physical (linewidth, SNR, impurity
concentration, quaternary relaxation) rather than a per-peak drop probability,
and they are calibrated against the measured peak-count distribution of the
real extracted set -- see `--calibrate`.

Usage
    python build_cnmr_nmrexp_peaklist.py --calibrate      # check the match, no push
    python build_cnmr_nmrexp_peaklist.py --limit 20000 --no-push
    python build_cnmr_nmrexp_peaklist.py                  # full build + push
"""

import argparse
import ast
import importlib.util
import math
import os
import re
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

# =============================================================================
# The extraction operator, imported from the chemotion builder.
#
# Imported rather than copied on purpose: if the peak picker changes, the
# training data has to change with it, and a copy would silently drift.
# =============================================================================
_CHEM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_cnmr_chemotion_peaklist.py")
_spec = importlib.util.spec_from_file_location("build_cnmr_chemotion_peaklist", _CHEM_PATH)
chem = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chem)

pick_peak_shifts = chem.pick_peak_shifts
collapse_solvent_multiplets = chem.collapse_solvent_multiplets
carbon_frequency = chem.carbon_frequency

# =============================================================================
# Config
# =============================================================================
X_MIN, X_MAX = float(chem.KEEP_LO), float(chem.KEEP_HI)  # 0.0, 218.0
SEED = 42

PARQUET_PATH = "NMRexp_10to24_1_1004.parquet"
N_SAMPLES = 1_500_000
VAL_FRAC = 0.01
HF_REPO = "jablonkagroup/nmrexp-cnmr-peaklist-1.5M"
N_PROC = max(1, cpu_count() - 1)

CHEMOTION_REF = "chemotion_cnmr_peaklist.parquet"  # calibration target

# --- acquisition ------------------------------------------------------------
# 32768 points over 218 ppm is 0.0067 ppm/pt, which at 100 MHz is 0.67 Hz/pt --
# the digitisation of a real 13C acquisition, so the picker's digital-resolution
# floor behaves as it does on real data.
N_GRID = 32768
DEFAULT_FIELD_MHZ = 100.0  # used only when the record has no usable frequency

# Linewidth in Hz, not ppm: a 13C linewidth is set by T2 and shimming, both
# field-independent, whereas a ppm width would shrink with field.
LW_HZ_LOG_MEAN, LW_HZ_LOG_SIGMA = math.log(1.8), 0.45
LW_HZ_CLIP = (0.7, 8.0)
LW_PEAK_JITTER = (0.85, 1.7)  # per-line multiplier on the spectrum linewidth
LW_BROAD_MULT = (2.5, 6.0)  # for 'm' sites and reported ranges

# --- relative intensities ---------------------------------------------------
# A proton-decoupled 13C spectrum is not quantitative: protonated carbons get
# the NOE and relax fast, quaternaries get neither. That single asymmetry is
# what makes quaternary carbons the peaks a threshold eats first, and it is the
# dominant systematic difference between a published peak list and a picked one.
QUAT_INTENSITY = (0.18, 0.55)
PROTONATED_INTENSITY = (0.70, 1.30)
INTENSITY_LOGNORM_SIGMA = 0.30

# Prior on P(carbon is quaternary | ppm), used to decide *which* peaks carry the
# quaternary count that RDKit gives us for the molecule. Carbonyls are
# essentially always quaternary; the aliphatic CH region essentially never.
QUAT_PPM_PRIOR = [
    (165.0, X_MAX, 0.95),  # carbonyl / carboxyl / amide
    (140.0, 165.0, 0.75),  # substituted aromatic, ipso
    (110.0, 140.0, 0.30),  # aromatic CH dominates but ipso lives here too
    (95.0, 110.0, 0.25),
    (60.0, 95.0, 0.12),
    (X_MIN, 60.0, 0.10),
]

# --- signal-to-noise --------------------------------------------------------
# SNR = tallest analyte line / noise sigma. The picker's threshold is 8 sigma,
# so SNR alone decides how far down the intensity ladder the peak list reaches:
# at SNR 100 a line must clear 8% of the tallest peak to be reported. This is
# the knob that produces the long left tail of under-picked spectra.
# Spectra are not acquired at a uniformly random SNR. People acquire until the
# spectrum looks acceptable and stop, so the population is mostly decent with a
# minority of bad ones -- a scarce sample, a short night, a weak solubility.
#
# This matters more than it looks. A single log-uniform range spreads moderate
# peak loss across *every* spectrum, which reproduces chemotion's heavy left
# tail only by degrading all of them: 19% of true peaks deleted on average. A
# mixture puts the loss where it belongs -- concentrated in the bad minority --
# and reproduces the same tail while leaving the bulk of the corpus nearly
# intact. Same marginal peak count, far more learnable signal.
SNR_BAD_FRAC = 0.20
SNR_GOOD_LOG10 = (2.30, 3.30)  # ~200 to ~2000
SNR_BAD_LOG10 = (1.00, 2.00)  # ~10 to ~100

# --- baseline ---------------------------------------------------------------
# The rolling-min baseline the picker applies is imperfect, so the drift has to
# be present for its residual to be present.
BASELINE_AMPL_SIGMA = (0.0, 6.0)  # in units of noise sigma
BASELINE_N_MODES = (2, 6)
BASELINE_HUMP_PROB = 0.25
BASELINE_HUMP_AMPL_SIGMA = (2.0, 25.0)
BASELINE_HUMP_WIDTH_PPM = (8.0, 45.0)

# --- solvent / impurities ---------------------------------------------------
# 75.2% of the real extracted spectra record a solvent the collapser recognises;
# the rest keep their uncollapsed residual multiplet. Matching that rate is the
# difference between a training set where 77.0/77.2/77.4 never appears and a
# test set where it appears a quarter of the time.
SOLVENT_COLLAPSE_RATE = 0.752
SOLVENT_INTENSITY = (0.3, 12.0)  # relative to the tallest analyte line
SOLVENT_PRESENT_PROB = 0.90

IMPURITY_INTENSITY_LOG10 = (-2.6, -1.6)  # 0.25% to 2.5% of the tallest line
GREASE_PROB = 0.35
GREASE_INTENSITY_LOG10 = (-2.3, -0.5)

# Unidentified minor components: leftover reagent, a rotamer, a diastereomer, a
# side product. The named-impurity table cannot cover these, but they are why a
# picked spectrum carries more lines than the paper reports -- and the paper is
# our label. Drawn as molecules rather than as loose peaks: a contaminant
# contributes a handful of carbons at one concentration, not one peak.
MINOR_COMPONENT_LAMBDA = 0.10
MINOR_COMPONENT_LINES = (1, 9)
MINOR_COMPONENT_LOG10 = (-2.2, -0.4)

# A rotamer, atropisomer or minor diastereomer is not a random molecule: it is
# the analyte again, every carbon nudged by a fraction of a ppm. That is what
# fills the tight-pair population -- 22% of the gaps in the real extracted set
# are under 0.4 ppm, and scattering unrelated contaminants uniformly over the
# range cannot produce it, because they almost never land next to a real peak.
ROTAMER_PROB = 0.10
ROTAMER_AMPL = (0.08, 0.55)
ROTAMER_SHIFT_SIGMA_PPM = (0.08, 0.60)

# Two carbons the paper reports as one number are not always symmetry
# equivalent -- often they are merely close, and the text rounds them together
# while the spectrum resolves them. Perfect coincidence would render them as a
# single line that no picker could ever split.
ACCIDENTAL_DEGENERACY_PROB = 0.45
ACCIDENTAL_SPREAD_PPM = (0.02, 0.35)

# =============================================================================
# Reference shifts (Gottlieb 1997; Fulmer 2010)
#
# Solvent residuals are (centre, n_lines, J_CD in Hz): 13C couples to deuterium
# (spin 1), so the residual is a 1:1:1 triplet in CDCl3 and a 1:3:6:7:6:3:1
# septet in DMSO-d6. Rendering it with the true J at the true field is what puts
# the lines the right distance apart -- and whether they resolve then depends on
# the field, exactly as on a real instrument.
# =============================================================================
SOLVENT_RESIDUALS = {
    "CDCl3": [(77.16, 3, 32.0)],
    "DMSO-d6": [(39.52, 7, 21.0)],
    "CD3OD": [(49.00, 7, 21.4)],
    "C6D6": [(128.06, 3, 24.3)],
    "CD2Cl2": [(53.84, 5, 27.2)],
    "acetone-d6": [(29.84, 7, 19.4), (206.26, 1, 0.0)],
    "CD3CN": [(1.32, 7, 21.0), (118.26, 1, 0.0)],
    "THF-d8": [(25.31, 5, 20.2), (67.21, 5, 22.2)],
    "pyridine-d5": [(123.87, 3, 25.0), (135.91, 3, 24.5), (150.35, 3, 27.5)],
    "toluene-d8": [(20.43, 7, 19.5), (125.13, 3, 24.0), (127.96, 3, 24.0), (128.87, 3, 24.0), (137.86, 1, 0.0)],
    "DMF-d7": [(29.76, 7, 21.0), (34.89, 7, 21.1), (163.15, 3, 29.4)],
    "D2O": [],
}

# NMRexp writes solvents as formulae ('CD3COCD3'), the reference tables above
# name them the way Gottlieb does ('acetone-d6'). Without this map the acetone
# rows -- 13k of them -- silently get no residual peak at all, and the model
# never learns that a spectrum can carry one.
_SOLVENT_KEY = {
    "cdcl3": "CDCl3", "chloroformd": "CDCl3",
    "dmsod6": "DMSO-d6", "d6dmso": "DMSO-d6", "cd3socd3": "DMSO-d6",
    "cd3od": "CD3OD", "meod": "CD3OD", "methanold4": "CD3OD",
    "c6d6": "C6D6", "benzened6": "C6D6",
    "cd2cl2": "CD2Cl2",
    "cd3cocd3": "acetone-d6", "acetoned6": "acetone-d6", "d6acetone": "acetone-d6",
    "cd3cn": "CD3CN", "acetonitriled3": "CD3CN",
    "thfd8": "THF-d8", "d8thf": "THF-d8",
    "pyridined5": "pyridine-d5", "c5d5n": "pyridine-d5",
    "phmed8": "toluene-d8", "toluened8": "toluene-d8", "cd3c6d5": "toluene-d8",
    "dmfd7": "DMF-d7",
    "d2o": "D2O",
}


def canonical_solvent(name):
    """NMRexp solvent string -> a key in the reference tables, or None."""
    if not name:
        return None
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    return _SOLVENT_KEY.get(key)

# n_lines -> relative line intensities for coupling to equivalent spin-1 nuclei
DEUTERIUM_PATTERN = {
    1: [1.0],
    3: [1.0, 1.0, 1.0],
    5: [1.0, 2.0, 3.0, 2.0, 1.0],
    7: [1.0, 3.0, 6.0, 7.0, 6.0, 3.0, 1.0],
}

IMPURITIES = {
    "tms": {"prob": 0.50, "shifts": {s: [0.00] for s in ["CDCl3", "DMSO-d6", "CD3OD", "C6D6", "CD2Cl2", "acetone-d6"]}},
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
        "prob": 0.10,
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
        "prob": 0.10,
        "shifts": {
            "CDCl3": [53.52], "acetone-d6": [54.95], "DMSO-d6": [54.84], "C6D6": [53.46],
            "CD3OD": [54.78], "CD2Cl2": [54.24],
        },
    },
    "hexane": {
        "prob": 0.10,
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

# Long-chain grease: terminal CH3, its neighbour, omega-2, and the CH2 envelope
# that dominates. The envelope is one tall broad line, not four sharp ones.
GREASE_ANCHORS = {
    "CDCl3": [14.1, 22.7, 31.9, 29.7],
    "acetone-d6": [14.3, 23.3, 32.3, 30.7],
    "DMSO-d6": [13.9, 22.1, 31.2, 29.2],
    "C6D6": [14.3, 23.1, 32.2, 30.2],
    "CD3OD": [14.5, 23.7, 33.1, 31.3],
    "CD2Cl2": [14.3, 23.1, 32.3, 30.1],
}

# =============================================================================
# Parsing the literature record
# =============================================================================
_FREQ_RE = re.compile(r"(\d+(?:\.\d+)?)")

# Multiplet labels as they appear in NMRexp, longest first so 'quint' is not
# read as 'q' + 'uint'. Value is the number of lines, i.e. n+1 for coupling to
# n equivalent spin-1/2 nuclei (F, P, H in a coupled experiment).
_MULT_TOKENS = [
    ("quart", 4), ("quint", 5), ("hept", 7), ("sept", 7), ("quin", 5), ("sex", 6),
    ("br", 0), ("s", 1), ("d", 2), ("t", 3), ("q", 4), ("p", 5), ("h", 7), ("m", 0),
]


def parse_field_mhz(freq):
    """'101 MHz' -> 101.0. Returns None for 'not_known' and friends."""
    if freq is None:
        return None
    m = _FREQ_RE.search(str(freq))
    if not m:
        return None
    return carbon_frequency(m.group(1))


def parse_shift(s):
    """A reported shift -> (centre_ppm, span_ppm).

    NMRexp stores an overlapping range as a two-element list ([120.8, 120.5]),
    which is a real multiplet reported as an interval, not a single line. Its
    span is kept so it can be rendered across that interval rather than
    collapsed to a point -- collapsing it is what makes a training peak list
    narrower than the picked one.
    """
    if isinstance(s, (list, tuple, np.ndarray)):
        vals = [float(v) for v in s if v is not None]
        if not vals:
            return None
        return float(np.mean(vals)), float(max(vals) - min(vals))
    try:
        return float(s), 0.0
    except (TypeError, ValueError):
        return None


def parse_js(J):
    """Reported J -> list of couplings in Hz (may be a scalar or a list)."""
    if J is None:
        return []
    vals = J if isinstance(J, (list, tuple, np.ndarray)) else [J]
    out = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            out.append(f)
    return out


def parse_mult(m):
    """'dd' -> [2, 2]; 'qd' -> [4, 2]; 'brd' -> [2]; 'm'/None -> [].

    A 0 from the token table means 'broad' or 'unresolved multiplet': it carries
    no line count, only the instruction to widen the line.
    """
    if not m:
        return [], False
    s = str(m).lower().replace(".", "").replace("-", "")
    counts, broad = [], False
    i = 0
    while i < len(s):
        for tok, n in _MULT_TOKENS:
            if s.startswith(tok, i):
                if n == 0:
                    broad = True
                elif n > 1:
                    counts.append(n)
                i += len(tok)
                break
        else:
            i += 1
    return counts, broad


def parse_peaks(raw):
    """NMR_processed -> [(centre, span, [n_lines...], [J...], broad)]."""
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    sites = []
    for entry in raw:
        if entry is None or len(entry) != 3:
            continue
        s, m, J = entry
        parsed = parse_shift(s)
        if parsed is None:
            continue
        centre, span = parsed
        if not (X_MIN <= centre < X_MAX):
            continue
        counts, broad = parse_mult(m)
        sites.append((centre, span, counts, parse_js(J), broad or span > 0))
    return sites


# =============================================================================
# Structure -> how many carbons, and how many of them are quaternary
# =============================================================================
def carbon_counts(mol):
    """(n_carbons, n_quaternary) where quaternary means no attached hydrogen."""
    n_c = n_q = 0
    for a in mol.GetAtoms():
        if a.GetSymbol() != "C":
            continue
        n_c += 1
        if a.GetTotalNumHs() == 0:
            n_q += 1
    return n_c, n_q


def _quat_prior(ppm):
    for lo, hi, p in QUAT_PPM_PRIOR:
        if lo <= ppm < hi:
            return p
    return 0.2


def assign_quaternary(centres, n_quat, rng):
    """Mark exactly `n_quat` of the reported sites as quaternary.

    The published peak list is unassigned, so which line belongs to which carbon
    is unknown -- but the *count* is known exactly from the structure, and the
    ppm region is strongly informative. Sampling without replacement under the
    ppm prior, constrained to the true count, uses both without pretending to an
    assignment we do not have.
    """
    n = len(centres)
    n_quat = int(min(max(n_quat, 0), n))
    if n_quat == 0:
        return np.zeros(n, dtype=bool)
    if n_quat >= n:
        return np.ones(n, dtype=bool)

    w = np.array([_quat_prior(c) for c in centres], dtype=np.float64)
    w = np.clip(w, 1e-3, None)
    # Gumbel top-k: weighted sampling without replacement in one shot.
    keys = np.log(w) + rng.gumbel(size=n)
    flags = np.zeros(n, dtype=bool)
    flags[np.argsort(-keys)[:n_quat]] = True
    return flags


def assign_degeneracy(centres, n_carbons, rng):
    """Spread the carbons the peak list does not account for over its sites.

    Fewer reported peaks than carbons means symmetry-equivalent carbons share a
    line, and a shared line is proportionally taller -- so it survives the
    threshold where a singleton would not. Aromatic CH pairs are the common
    case, so the surplus is placed with a mild preference for that region.
    """
    n = len(centres)
    deg = np.ones(n, dtype=np.float64)
    surplus = int(n_carbons) - n
    if surplus <= 0 or n == 0:
        return deg
    surplus = min(surplus, 4 * n)
    w = np.array([2.5 if 105.0 <= c < 150.0 else 1.0 for c in centres])
    w /= w.sum()
    deg += rng.multinomial(surplus, w)
    return deg


# =============================================================================
# Sites -> resolved lines
# =============================================================================
def _binomial(n_lines):
    return np.array([math.comb(n_lines - 1, k) for k in range(n_lines)], dtype=np.float64)


def expand_multiplet(centre, counts, js, field_mhz):
    """One site -> the lines it actually shows at this field.

    A published '180.0 (d, J = 300.8 Hz)' is one number, but at 101 MHz it is
    two lines 2.98 ppm apart -- fully resolved, and a peak picker returns both.
    16.5% of the 13C records here carry a J, and 21% of those J values exceed
    100 Hz, so this is the single largest systematic difference between the
    published list and the picked one. It is computable exactly, because both
    the coupling and the field are in the record.
    """
    lines = [(centre, 1.0)]
    for n_lines, j_hz in zip(counts, js):
        sep = j_hz / field_mhz
        weights = _binomial(n_lines)
        weights /= weights.sum()
        offsets = (np.arange(n_lines) - (n_lines - 1) / 2.0) * sep
        lines = [(c + off, w * wt) for c, w in lines for off, wt in zip(offsets, weights)]
    return lines


def analyte_lines(sites, n_carbons, n_quat, field_mhz, rng):
    """Reported sites -> [(ppm, intensity, linewidth_multiplier)]."""
    if not sites:
        return []
    centres = [s[0] for s in sites]
    quat = assign_quaternary(centres, n_quat, rng)
    deg = assign_degeneracy(centres, n_carbons, rng)

    out = []
    for i, (centre, span, counts, js, broad) in enumerate(sites):
        resp = rng.uniform(*(QUAT_INTENSITY if quat[i] else PROTONATED_INTENSITY))
        amp = deg[i] * resp * float(rng.lognormal(0.0, INTENSITY_LOGNORM_SIGMA))
        lw_mult = rng.uniform(*(LW_BROAD_MULT if broad else LW_PEAK_JITTER))

        if span > 0:
            # A reported interval is several unresolved carbons; render them
            # across the interval rather than stacking them at its midpoint.
            k = max(2, int(round(deg[i])))
            for pos in np.linspace(centre - span / 2, centre + span / 2, k):
                out.append((float(pos), amp / k, lw_mult))
            continue

        k = int(round(deg[i]))
        if k >= 2 and rng.random() < ACCIDENTAL_DEGENERACY_PROB:
            spread = rng.uniform(*ACCIDENTAL_SPREAD_PPM)
            sub = [(centre + off, 1.0 / k) for off in np.linspace(-spread / 2, spread / 2, k)]
        else:
            sub = [(centre, 1.0)]

        for c, frac in sub:
            for ppm, w in expand_multiplet(c, counts, js, field_mhz):
                out.append((float(ppm), amp * frac * w, lw_mult))
    return out


def rotamer_lines(lines, rng):
    """A shifted, weaker copy of the analyte -- a minor conformer or isomer."""
    ampl = rng.uniform(*ROTAMER_AMPL)
    sigma = rng.uniform(*ROTAMER_SHIFT_SIGMA_PPM)
    out = []
    for ppm, amp, lw in lines:
        p = ppm + float(rng.normal(0.0, sigma))
        if X_MIN <= p < X_MAX:
            out.append((p, amp * ampl, lw))
    return out


def contaminant_lines(solvent, field_mhz, max_amp, rng):
    """Solvent residual, common impurities and grease, as real lines.

    Everything here is added as signal with a concentration, not as a 'spurious
    peak'. Whether any of it clears the picker's threshold is then decided by
    the picker, at the SNR this spectrum happens to have -- which is how it
    works on a real spectrum.
    """
    out = []

    if rng.random() < SOLVENT_PRESENT_PROB:
        for centre, n_lines, j_hz in SOLVENT_RESIDUALS.get(solvent, []):
            amp = max_amp * rng.uniform(*SOLVENT_INTENSITY)
            pattern = np.array(DEUTERIUM_PATTERN.get(n_lines, [1.0]), dtype=np.float64)
            pattern /= pattern.sum()
            sep = j_hz / field_mhz
            offsets = (np.arange(len(pattern)) - (len(pattern) - 1) / 2.0) * sep
            lw = rng.uniform(*LW_PEAK_JITTER)
            for off, w in zip(offsets, pattern):
                out.append((centre + float(off), amp * w, lw))

    for entry in IMPURITIES.values():
        table = entry["shifts"]
        if solvent not in table or rng.random() >= entry["prob"]:
            continue
        amp = max_amp * float(10 ** rng.uniform(*IMPURITY_INTENSITY_LOG10))
        for s in table[solvent]:
            out.append((float(s), amp, rng.uniform(*LW_PEAK_JITTER)))

    for _ in range(int(rng.poisson(MINOR_COMPONENT_LAMBDA))):
        amp = max_amp * float(10 ** rng.uniform(*MINOR_COMPONENT_LOG10))
        for _ in range(int(rng.integers(*MINOR_COMPONENT_LINES))):
            ppm = float(rng.uniform(X_MIN, X_MAX))
            out.append((ppm, amp * float(rng.lognormal(0.0, 0.4)), rng.uniform(*LW_PEAK_JITTER)))

    anchors = GREASE_ANCHORS.get(solvent)
    if anchors is not None and rng.random() < GREASE_PROB:
        amp = max_amp * float(10 ** rng.uniform(*GREASE_INTENSITY_LOG10))
        ch3, ch2_a, ch2_b, envelope = anchors
        out.append((ch3, amp * 0.5, rng.uniform(*LW_PEAK_JITTER)))
        out.append((ch2_a, amp * 0.4, rng.uniform(*LW_PEAK_JITTER)))
        out.append((ch2_b, amp * 0.4, rng.uniform(*LW_PEAK_JITTER)))
        # the CH2 envelope: many carbons in one broad line, hence tall and wide
        out.append((envelope, amp * 4.0, rng.uniform(*LW_BROAD_MULT)))

    return out


# =============================================================================
# Lines -> a trace
# =============================================================================
def render_trace(lines, x, lw_ppm):
    """Sum of Lorentzians on the ppm grid.

    Lorentzian, not Gaussian: an NMR line is the Fourier transform of an
    exponentially decaying FID, and its wide wings are what let a strong
    neighbour swallow a weak line in the picker's prominence test.
    """
    y = np.zeros_like(x)
    for ppm, amp, lw_mult in lines:
        if amp <= 0:
            continue
        hwhm = 0.5 * lw_ppm * lw_mult
        # evaluate only where the line is non-negligible (40 half-widths)
        lo = int(np.searchsorted(x, ppm - 40 * hwhm))
        hi = int(np.searchsorted(x, ppm + 40 * hwhm))
        lo, hi = max(lo, 0), min(max(hi, lo + 1), len(x))
        if hi <= lo:
            continue
        # never let a line fall between two samples and vanish
        if (hi - lo) < 3:
            lo = max(0, lo - 1)
            hi = min(len(x), lo + 3)
        seg = x[lo:hi] - ppm
        y[lo:hi] += amp * hwhm * hwhm / (seg * seg + hwhm * hwhm)
    return y


def add_baseline(y, x, sigma, rng):
    """Smooth drift plus an occasional broad hump."""
    span = x[-1] - x[0]
    n_modes = int(rng.integers(*BASELINE_N_MODES))
    ampl = sigma * rng.uniform(*BASELINE_AMPL_SIGMA)
    u = (x - x[0]) / span
    drift = np.zeros_like(x)
    for k in range(1, n_modes + 1):
        drift += rng.normal(0.0, 1.0 / k) * np.sin(math.pi * k * u + rng.uniform(0, 2 * math.pi))
    y = y + ampl * drift

    if rng.random() < BASELINE_HUMP_PROB:
        c = rng.uniform(x[0], x[-1])
        w = rng.uniform(*BASELINE_HUMP_WIDTH_PPM)
        y = y + sigma * rng.uniform(*BASELINE_HUMP_AMPL_SIGMA) * np.exp(-0.5 * ((x - c) / w) ** 2)
    return y


def synthesize(sites, n_carbons, n_quat, solvent, field_mhz, rng):
    """Everything above, assembled into (x, y, field)."""
    x = np.linspace(X_MIN, X_MAX, N_GRID)

    lw_hz = float(np.clip(rng.lognormal(LW_HZ_LOG_MEAN, LW_HZ_LOG_SIGMA), *LW_HZ_CLIP))
    lw_ppm = lw_hz / field_mhz

    lines = analyte_lines(sites, n_carbons, n_quat, field_mhz, rng)
    if not lines:
        return None
    max_amp = max(a for _, a, _ in lines)
    if rng.random() < ROTAMER_PROB:
        lines += rotamer_lines(lines, rng)
    lines += contaminant_lines(solvent, field_mhz, max_amp, rng)

    y = render_trace(lines, x, lw_ppm)

    snr_range = SNR_BAD_LOG10 if rng.random() < SNR_BAD_FRAC else SNR_GOOD_LOG10
    snr = float(10 ** rng.uniform(*snr_range))
    sigma = float(np.max(y)) / snr
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    y = add_baseline(y, x, sigma, rng)
    y = y + rng.normal(0.0, sigma, size=y.shape)
    return x, y


# =============================================================================
# One row, end to end
# =============================================================================
def process_row(args):
    peaks, solvent, freq, smiles, seed = args
    try:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return None
        canon = Chem.MolToSmiles(mol)
        n_carbons, n_quat = carbon_counts(mol)
        if n_carbons == 0:
            return None

        sites = parse_peaks(peaks)
        if not sites:
            return None

        rng = np.random.default_rng(seed)
        field = parse_field_mhz(freq) or DEFAULT_FIELD_MHZ
        solvent_key = canonical_solvent(solvent)

        trace = synthesize(sites, n_carbons, n_quat, solvent_key, field, rng)
        if trace is None:
            return None
        x, y = trace

        shifts = pick_peak_shifts(x, y, chem.DEFAULTS, field)
        if not shifts:
            return None

        # The real pipeline collapses the solvent multiplet only when the record
        # names a solvent it recognises, which it does 75% of the time. Applying
        # it always would train the model on a residual pattern the test set
        # keeps a quarter of the time.
        if rng.random() < SOLVENT_COLLAPSE_RATE:
            shifts = collapse_solvent_multiplets(shifts, solvent_key, field)

        shifts = sorted(float(s) for s in shifts if X_MIN <= s <= X_MAX)
        if not shifts:
            return None

        return {
            "smiles": canon,
            "c_nmr": shifts,
            "x_min": X_MIN,
            "x_max": X_MAX,
            "solvent": solvent_key,
            "base_frequency_MHz": float(field),
        }
    except Exception:
        return None


# =============================================================================
# Build
# =============================================================================
def load_cnmr(limit=None):
    data = pd.read_parquet(PARQUET_PATH)
    cnmr = data[data.NMR_type == "13C NMR"]
    cnmr = cnmr[~cnmr.NMR_solvent.isin(["not_known", "mixed"])]
    cnmr = cnmr.reset_index(drop=True)
    if limit is not None and len(cnmr) > limit:
        cnmr = cnmr.sample(n=limit, random_state=SEED).reset_index(drop=True)
    return cnmr


def make_tasks(cnmr):
    cols = zip(cnmr.NMR_processed, cnmr.NMR_solvent, cnmr.NMR_frequency, cnmr.SMILES)
    return [(p, s, f, sm, SEED + i) for i, (p, s, f, sm) in enumerate(cols)]


def run(tasks, desc="processing", n_proc=None):
    rows = []
    n_proc = n_proc or N_PROC
    with Pool(processes=n_proc) as pool:
        for res in tqdm(pool.imap_unordered(process_row, tasks, chunksize=64), total=len(tasks), desc=desc):
            if res is not None:
                rows.append(res)
    return pd.DataFrame(rows)


def _peaks_per_carbon(smis, peaks):
    out = []
    for smi, p in zip(smis, peaks):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n_c, _ = carbon_counts(mol)
        if n_c:
            out.append(len(p) / n_c)
    return np.asarray(out)


def _describe(name, r):
    if not len(r):
        print(f"{name:34s} (empty)")
        return
    print(
        f"{name:34s} n={len(r):6d}  p10={np.percentile(r, 10):.2f}  "
        f"med={np.median(r):.2f}  p90={np.percentile(r, 90):.2f}  mean={r.mean():.2f}"
    )


def _describe_gaps(name, peaks):
    """Nearest-neighbour peak spacing -- deliberately NOT one of the calibrated
    statistics, so it is an independent check that the generator is producing
    the right *arrangement* of peaks and not merely the right number."""
    arrs = [np.sort(np.asarray(p, dtype=np.float64)) for p in peaks if len(p) > 1]
    if not arrs:
        print(f"{name:34s} (empty)")
        return
    g = np.concatenate([np.diff(a) for a in arrs])
    print(
        f"{name:34s} med={np.median(g):5.2f}  "
        f"frac<0.4ppm={np.mean(g < 0.4):.3f}  frac<1ppm={np.mean(g < 1.0):.3f}"
    )


def _recall_precision(published, picked, tol=0.5):
    """How much of the published peak list survives, and how much of the
    generated list is explainable from it.

    Peaks-per-carbon alone does NOT pin this down, which is the trap: "delete
    19% of the true peaks and add 24% noise" and "delete 5% and add 13%" give
    the *same* peak count. Only the second is learnable. Any retune has to watch
    these two numbers or it can match every marginal and still quietly destroy
    the signal.
    """
    rec, prec = [], []
    for truth, got in zip(published, picked):
        if not len(truth) or not len(got):
            continue
        t = np.asarray(truth, dtype=np.float64)
        p = np.asarray(got, dtype=np.float64)
        d = np.abs(t[:, None] - p[None, :])
        rec.append(np.mean(d.min(axis=1) < tol))
        prec.append(np.mean(d.min(axis=0) < tol))
    return float(np.mean(rec)), float(np.mean(prec))


def calibrate(n, n_proc=None):
    """Compare peaks-per-carbon against the real extracted set.

    Peaks per carbon is the cheapest statistic that is sensitive to every knob
    at once: SNR and the quaternary response move the median and the left tail,
    the J expansion and the solvent-collapse rate move the right tail. Matching
    it does not prove the generator is right, but failing to match it proves it
    is wrong -- and the previous hand-written augmentation failed it badly
    (mean 1.31 against 0.95, p90 2.20 against 1.50).
    """
    cnmr = load_cnmr(limit=n)
    out = run(make_tasks(cnmr), desc="calibrating", n_proc=n_proc)
    print(f"\nyield: {len(out)}/{len(cnmr)} rows survived ({len(out) / len(cnmr):.1%})\n")

    print("peaks per carbon")
    print("-" * 68)
    ref = pd.read_parquet(CHEMOTION_REF) if os.path.exists(CHEMOTION_REF) else None
    if ref is not None:
        _describe("chemotion (real, extracted)", _peaks_per_carbon(ref.smiles, ref.c_nmr))
    else:
        print(f"  [{CHEMOTION_REF} not found -- no reference to compare against]")

    raw = [[s for s, _, _, _, _ in parse_peaks(p)] for p in cnmr.NMR_processed]
    _describe("nmrexp (published peak list)", _peaks_per_carbon(cnmr.SMILES, raw))
    _describe("nmrexp -> synth -> picked (new)", _peaks_per_carbon(out.smiles, out.c_nmr))

    print("\npeak spacing (held out of the calibration)")
    print("-" * 68)
    if ref is not None:
        _describe_gaps("chemotion (real, extracted)", ref.c_nmr)
    _describe_gaps("nmrexp (published peak list)", raw)
    _describe_gaps("nmrexp -> synth -> picked (new)", out.c_nmr)

    published = {}
    for ps, smi in zip(cnmr.NMR_processed, cnmr.SMILES):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        sites = parse_peaks(ps)
        if sites:
            published.setdefault(Chem.MolToSmiles(mol), [x[0] for x in sites])
    pairs = [(published[s], p) for s, p in zip(out.smiles, out.c_nmr) if s in published]
    rec, prec = _recall_precision([a for a, _ in pairs], [b for _, b in pairs])

    print("\nsignal vs noise (NOT constrained by the statistics above)")
    print("-" * 68)
    print(f"  published peak survives          {rec:.3f}")
    print(f"  generated peak is explainable    {prec:.3f}   ({1 - prec:.1%} not derivable from the structure)")

    print("-" * 68)
    counts = out.c_nmr.map(len)
    print(f"peaks per spectrum: mean {counts.mean():.1f} | median {counts.median():.0f} | max {counts.max()}")
    return out


def build(limit=None, out_path=None, push=True, n_proc=None):
    cnmr = load_cnmr(limit=limit if limit is not None else N_SAMPLES)
    print(f"filtered 13C rows: {len(cnmr)}")

    out = run(make_tasks(cnmr), n_proc=n_proc)
    n_before = len(out)
    out = out.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)
    print(f"built {n_before} | after dedup {len(out)} | dropped {len(cnmr) - n_before} in processing")

    counts = out.c_nmr.map(len)
    print(f"peaks per spectrum: mean {counts.mean():.1f} | median {counts.median():.0f} | max {counts.max()}")

    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_val = int(len(out) * VAL_FRAC)
    val_df, train_df = out.iloc[:n_val], out.iloc[n_val:]
    print(f"train {len(train_df)} | val {len(val_df)}")

    if out_path:
        out.to_parquet(out_path, index=False)
        print(f"saved -> {out_path}")

    if push:
        from datasets import Dataset, DatasetDict

        ds = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df.reset_index(drop=True), preserve_index=False),
                "val": Dataset.from_pandas(val_df.reset_index(drop=True), preserve_index=False),
            }
        )
        ds.push_to_hub(HF_REPO, private=False)
        print(f"pushed to {HF_REPO}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", type=int, nargs="?", const=20000, default=None,
                    help="build N rows and print the peaks-per-carbon match against chemotion; no push")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of source spectra")
    ap.add_argument("--out", default=None, help="also write the result to this parquet")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--n-proc", type=int, default=None)
    args = ap.parse_args()

    if args.calibrate is not None:
        out = calibrate(args.calibrate, n_proc=args.n_proc)
        if args.out:
            out.to_parquet(args.out, index=False)
            print(f"saved -> {args.out}")
    else:
        build(limit=args.limit, out_path=args.out, push=not args.no_push, n_proc=args.n_proc)
