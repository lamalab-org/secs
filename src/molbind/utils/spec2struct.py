import re
from collections import defaultdict

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.interpolate import interp1d


def get_atom_counts_from_formula(formula_string: str) -> dict:
    """
    Parses a simple molecular formula string into a dictionary of atom counts.
    Example: "C6H12O6" -> {'C': 6, 'H': 12, 'O': 6}
    Handles elements with one or two letters and optional counts.
    Assumes "flat" formulas (e.g., no parentheses or complex structures).
    """
    if not isinstance(formula_string, str):
        # Depending on desired strictness, could raise TypeError or return empty.
        # Matching original behavior of not erroring out for non-strings if it implicitly fails.
        return {}
    if not formula_string:
        return {}

    counts = defaultdict(int)
    # Regex: ([A-Z][a-z]?) matches an element symbol (e.g., C, Cl)
    #        (\d*) matches an optional count number
    pattern = re.compile(r"([A-Z][a-z]?)(\d*)")

    parsed_length = 0
    for match in pattern.finditer(formula_string):
        if match.start() != parsed_length:
            # This indicates unparsed characters, suggesting a complex or malformed formula.
            # For this scope, we assume simple formulas and might ignore such parts or errors.
            # Example: "CH(CH3)2" - (CH3)2 part would be problematic.
            # Consider raising ValueError for stricter parsing if needed.
            pass  # Silently continue with what was parsed.

        element = match.group(1)
        count_str = match.group(2)

        count = int(count_str) if count_str else 1
        counts[element] += count
        parsed_length = match.end()

    if parsed_length != len(formula_string):
        # This indicates trailing unparsed characters.
        # Similar to above, could indicate issues with formula string.
        pass  # Silently ignore for now.

    return dict(counts)


def build_formula_string(atom_counts: dict) -> str:
    """
    Constructs a molecular formula string from a dictionary of atom counts.
    Follows Hill system convention (C, then H, then other elements alphabetically).
    Omits elements with counts <= 0. Omits count if it's 1.
    Example: {'C': 6, 'H': 12, 'O': 6} -> "C6H12O6"
             {'C': 1, 'H': 4} -> "CH4"
    """
    if not isinstance(atom_counts, dict):
        return ""  # Or raise TypeError

    formula_parts = []

    # Carbon is typically first
    c_count = atom_counts.get("C", 0)
    if c_count > 0:
        formula_parts.append("C")
        if c_count > 1:
            formula_parts.append(str(c_count))

    # Hydrogen is typically second (if Carbon is present or by convention)
    h_count = atom_counts.get("H", 0)
    if h_count > 0:
        formula_parts.append("H")
        if h_count > 1:
            formula_parts.append(str(h_count))

    # Other elements, sorted alphabetically
    other_elements = sorted([el for el in atom_counts if el not in ("C", "H") and atom_counts.get(el, 0) > 0])

    for el_symbol in other_elements:
        count = atom_counts[el_symbol]  # Already filtered for count > 0
        formula_parts.append(el_symbol)
        if count > 1:
            formula_parts.append(str(count))

    return "".join(formula_parts)


def gen_close_molformulas_from_seed(seed_formula: str) -> list:
    initial_counts = get_atom_counts_from_formula(seed_formula)

    carbons = initial_counts.get("C", 0)
    hydrogens = initial_counts.get("H", 0)
    nitrogens = initial_counts.get("N", 0)
    chlorine = initial_counts.get("Cl", 0)
    bromine = initial_counts.get("Br", 0)
    fluorine = initial_counts.get("F", 0)
    phosphorus_orig = initial_counts.get("P", 0)  # Renamed to avoid conflict
    sulphur_orig = initial_counts.get("S", 0)  # Renamed to avoid conflict

    generated_formulas = []

    # Transformations based on original logic, applied to atom counts

    # 0: C-3, H-6
    counts = initial_counts.copy()
    counts["C"] = carbons - 3
    counts["H"] = hydrogens - 6
    generated_formulas.append(build_formula_string(counts))

    # 1: C+1, H+2
    counts = initial_counts.copy()
    counts["C"] = carbons + 1
    counts["H"] = hydrogens + 2
    generated_formulas.append(build_formula_string(counts))

    # 2: C-1, H-2
    counts = initial_counts.copy()
    counts["C"] = carbons - 1
    counts["H"] = hydrogens - 2
    generated_formulas.append(build_formula_string(counts))

    # 3: C+2, H+4
    counts = initial_counts.copy()
    counts["C"] = carbons + 2
    counts["H"] = hydrogens + 4
    generated_formulas.append(build_formula_string(counts))

    # 4: C-2, H-4
    counts = initial_counts.copy()
    counts["C"] = carbons - 2
    counts["H"] = hydrogens - 4
    generated_formulas.append(build_formula_string(counts))

    # 5: N+1, H+1
    counts = initial_counts.copy()
    counts["N"] = nitrogens + 1
    counts["H"] = hydrogens + 1
    generated_formulas.append(build_formula_string(counts))

    # 6: N-1, H-1
    counts = initial_counts.copy()
    counts["N"] = nitrogens - 1
    counts["H"] = hydrogens - 1
    generated_formulas.append(build_formula_string(counts))

    # 7: Cl+1, H+1
    counts = initial_counts.copy()
    counts["Cl"] = chlorine + 1
    counts["H"] = hydrogens + 1
    generated_formulas.append(build_formula_string(counts))

    # 8: Cl-1, H-1
    counts = initial_counts.copy()
    counts["Cl"] = chlorine - 1
    counts["H"] = hydrogens - 1
    generated_formulas.append(build_formula_string(counts))

    # 9: Br+1, H+1
    counts = initial_counts.copy()
    counts["Br"] = bromine + 1
    counts["H"] = hydrogens + 1
    generated_formulas.append(build_formula_string(counts))

    # 10: Br-1, H-1
    counts = initial_counts.copy()
    counts["Br"] = bromine - 1
    counts["H"] = hydrogens - 1
    generated_formulas.append(build_formula_string(counts))

    # 11: F+1, H+1
    counts = initial_counts.copy()
    counts["F"] = fluorine + 1
    counts["H"] = hydrogens + 1
    generated_formulas.append(build_formula_string(counts))

    # 12: Remove Cl, Br, F. H_new = H_original + total_original_halogens.
    total_halogens_original = initial_counts.get("Cl", 0) + initial_counts.get("Br", 0) + initial_counts.get("F", 0)
    counts = initial_counts.copy()
    counts["H"] = hydrogens + total_halogens_original
    counts["Cl"] = 0
    counts["Br"] = 0
    counts["F"] = 0
    generated_formulas.append(build_formula_string(counts))

    # 13: P becomes 0. H_new = H_original + 5.
    counts = initial_counts.copy()
    counts["H"] = hydrogens + 5
    counts["P"] = 0
    generated_formulas.append(build_formula_string(counts))

    # 14: S becomes 0. H_new = H_original + 4.
    counts = initial_counts.copy()
    counts["H"] = hydrogens + 4
    counts["S"] = 0
    generated_formulas.append(build_formula_string(counts))

    # 15: S+1
    counts = initial_counts.copy()
    counts["S"] = sulphur_orig + 1
    generated_formulas.append(build_formula_string(counts))

    # 16: S-1
    counts = initial_counts.copy()
    counts["S"] = sulphur_orig - 1
    generated_formulas.append(build_formula_string(counts))

    # 17: P+1
    counts = initial_counts.copy()
    counts["P"] = phosphorus_orig + 1
    generated_formulas.append(build_formula_string(counts))

    # 18: P-1
    counts = initial_counts.copy()
    counts["P"] = phosphorus_orig - 1
    generated_formulas.append(build_formula_string(counts))

    return generated_formulas


def smiles_to_molecular_formula(smiles: str) -> str:
    """Convert a SMILES string to a molecular formula."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return rdMolDescriptors.CalcMolFormula(mol)


def convert_to_molecular_formulas(datafile: str) -> list[str]:
    """Converts a datafile to a list of molecular formulas using Polars."""
    # Read the pickle file
    data = pl.read_pickle(datafile)
    data = data.with_columns(pl.col("smiles").map_elements(smiles_to_molecular_formula).alias("molecular_formula"))

    # Convert to list and return
    return data["molecular_formula"].to_list()


def is_neutral_no_isotopes(smiles):
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


def reduce_resolution_by_averaging(vector, window_size):
    """
    Reduces the resolution of a vector by window averaging and interpolation.

    This function first computes the moving average of the input vector
    using a specified window size, which reduces its length. It then
    interpolates the averaged data back to the original vector length.

    Args:
        vector (np.ndarray): The input 1D numpy array of data.
        window_size (int): The size of the averaging window. A larger
                           window results in lower resolution.

    Returns:
        np.ndarray: A new vector with reduced resolution but the same
                    length as the input vector.
    """
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)

    if window_size <= 1:
        return vector  # No resolution change needed

    # 1. Window Averaging (Downsampling)
    # We use convolution for a simple moving average.
    # The 'valid' mode means we only get points where the window fully overlaps.
    averaged_vector = np.convolve(vector, np.ones(window_size) / window_size, mode="valid")

    # 2. Interpolation (Upsampling)
    # Create the x-coordinates for the original and downsampled vectors
    original_x = np.linspace(0, 1, len(vector))
    averaged_x = np.linspace(0, 1, len(averaged_vector))

    # Create an interpolation function based on the averaged data
    interp_func = interp1d(averaged_x, averaged_vector, kind="linear", fill_value="extrapolate")

    # Apply the interpolation function to the original x-coordinates
    new_vector = interp_func(original_x)

    return new_vector
