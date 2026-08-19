import re
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.interpolate import interp1d


def get_atom_counts_from_formula(formula_string: str) -> dict[str, int]:
    """Parses a simple molecular formula string into a dictionary of atom counts.
    Example: "C6H12O6" -> {'C': 6, 'H': 12, 'O': 6}

    Args:
        formula_string (str): Molecular formula

    Returns:
        dict[str, int]: Atom types and counts dictionary
    """

    counts = defaultdict(int)
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula_string):
        counts[element] += int(count) if count else 1
    return dict(counts)


def build_formula_string(atom_counts: dict) -> str:
    """
    Hill-system formula string: C, H, then others alphabetically; omits counts <= 0 and '1'.

    Args:
        atom_counts (dict): The atom count dictionary {"C": 10, "H": 12, "O": 6}

    Returns:
        str: The molecular formula
    """
    others = sorted(e for e in atom_counts if e not in ("C", "H"))
    element_order = ["C", "H"] + others

    parts = []
    for element in element_order:
        count = atom_counts.get(element, 0)
        if count > 0:
            parts.append(element if count == 1 else f"{element}{count}")

    return "".join(parts)


def _apply_deltas(base: Dict[str, int], deltas: Dict[str, int]) -> Optional[Dict[str, int]]:
    """Apply atom-count deltas; return None if any count would go negative."""
    counts = base.copy()
    for elem, d in deltas.items():
        counts[elem] = counts.get(elem, 0) + d
        if counts[elem] < 0:
            return None
    # Drop zero-count elements
    return {e: c for e, c in counts.items() if c > 0}


def _zero_out(base: Dict[str, int], elements: List[str], h_delta: int) -> Optional[Dict[str, int]]:
    """Remove given elements entirely, adjusting H. Return None if nothing changed."""
    if not any(base.get(e, 0) > 0 for e in elements):
        return None  # transformation is a no-op
    counts = {e: c for e, c in base.items() if e not in elements and c > 0}
    counts["H"] = counts.get("H", 0) + h_delta
    if counts["H"] < 0:
        return None
    return counts


def gen_close_molformulas_from_seed(seed_formula: str) -> List[str]:
    """
    Generate chemically plausible 'neighbor' molecular formulas of a seed.
    """
    if not seed_formula or not isinstance(seed_formula, str):
        raise ValueError(f"Invalid seed formula: {seed_formula!r}")

    initial = get_atom_counts_from_formula(seed_formula)
    if not initial or all(v <= 0 for v in initial.values()):
        raise ValueError(f"Could not parse formula: {seed_formula!r}")

    # Simple delta-based transformations
    delta_sets: List[Dict[str, int]] = [
        {"C": -3, "H": -6},
        {"C": +1, "H": +2},
        {"C": -1, "H": -2},
        {"C": +2, "H": +4},
        {"C": -2, "H": -4},
        {"N": +1, "H": +1},
        {"N": -1, "H": -1},
        {"Cl": +1, "H": +1},
        {"Cl": -1, "H": -1},
        {"Br": +1, "H": +1},
        {"Br": -1, "H": -1},
        {"F": +1, "H": +1},
        {"S": +1},
        {"S": -1},
        {"P": +1},
        {"P": -1},
    ]

    candidates: List[Optional[Dict[str, int]]] = [_apply_deltas(initial, d) for d in delta_sets]

    # Structural removals
    total_halogens = sum(initial.get(x, 0) for x in ("Cl", "Br", "F"))
    candidates.append(_zero_out(initial, ["Cl", "Br", "F"], h_delta=total_halogens))
    candidates.append(_zero_out(initial, ["P"], h_delta=5))
    candidates.append(_zero_out(initial, ["S"], h_delta=4))

    seed_canonical = build_formula_string({e: c for e, c in initial.items() if c > 0})
    seen = set()
    results: List[str] = []
    for counts in candidates:
        if counts is None:
            continue
        formula = build_formula_string(counts)
        if formula and formula != seed_canonical and formula not in seen:
            seen.add(formula)
            results.append(formula)
    return results


def smiles_to_molecular_formula(smiles: str) -> str:
    """Convert a SMILES string to a molecular formula."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return rdMolDescriptors.CalcMolFormula(mol)


def is_neutral_no_isotopes(smiles: str) -> bool:
    """Check if molecule is neutral and contains no isotopes"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # Check for formal charges
        total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        if total_charge != 0:
            return False

        # Check for isotopes
        has_isotopes = any(atom.GetIsotope() != 0 for atom in mol.GetAtoms())
        return not has_isotopes
    except Exception:
        return False


def reduce_resolution_by_averaging(vector: np.ndarray, window_size: int) -> np.ndarray:
    """
    Reduces the resolution of a vector by window averaging and interpolation.

    Args:
        vector (np.ndarray): The input 1D numpy array of data.
        window_size (int): The size of the averaging window. A larger
                           window results in lower resolution.

    Returns:
        np.ndarray: A new vector with reduced resolution but the same
                    length as the input vector.
    """
    if isinstance(vector, list):
        vector = np.array(vector)

    if window_size <= 1:
        return vector

    averaged_vector = np.convolve(vector, np.ones(window_size) / window_size, mode="valid")
    original_x = np.linspace(0, 1, len(vector))
    averaged_x = np.linspace(0, 1, len(averaged_vector))
    interp_func = interp1d(averaged_x, averaged_vector, kind="linear", fill_value="extrapolate")

    # Apply the interpolation function to the original x-coordinates
    new_vector = interp_func(original_x)

    return new_vector
