from functools import cache

import numpy as np
from rdkit import Chem

from secs.elucidation.verifiers.base import Verifier


@cache
def n_distinct_environments(smiles: str, atomic_number: int = 6) -> int | None:
    """Count symmetry-distinct atoms of one element.

    This is the maximum number of peaks the corresponding nucleus can produce:
    symmetry-equivalent atoms are indistinguishable and give one signal. Uses
    RDKit's canonical ranking with breakTies=False so equivalent atoms share a
    rank.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if atomic_number == 1:
        mol = Chem.AddHs(mol)
    ranks = np.array(list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False)))
    indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == atomic_number]
    if not indices:
        return 0
    return len(set(ranks[indices]))


class PeakCountVerifier(Verifier):
    """Checks a candidate has enough distinct nuclei to account for the peaks seen.

    The constraint is one-sided. Symmetry-equivalent nuclei give one signal and
    peaks can coincide, so a molecule may have *more* environments than
    resolved peaks -- that is ordinary overlap and carries no penalty. It
    cannot have *fewer*: a structure with 6 distinct carbons cannot produce a
    12-peak 13C spectrum.

    Observed peak lists also contain solvent and impurity signals, which
    inflate the count through no fault of the candidate. `solvent_tolerance`
    forgives that many surplus peaks before any penalty applies.

    Scores 0 when the candidate can account for the spectrum, falling to -1 as
    the shortfall approaches the number of observed peaks.
    """

    def __init__(
        self,
        n_observed_peaks: int,
        atomic_number: int = 6,
        solvent_tolerance: int = 0,
        name: str = "peak_count",
    ) -> None:
        if n_observed_peaks <= 0:
            raise ValueError("n_observed_peaks must be positive.")
        self.n_observed_peaks = n_observed_peaks
        self.atomic_number = atomic_number
        self.solvent_tolerance = solvent_tolerance
        self.name = name

    def verify(self, smiles: str) -> float:
        n = n_distinct_environments(smiles, self.atomic_number)
        if n is None:
            return -1.0
        required = self.n_observed_peaks - self.solvent_tolerance
        shortfall = max(required - n, 0)
        return -min(shortfall / self.n_observed_peaks, 1.0)


class UnsaturationVerifier(Verifier):
    """Checks the degree of unsaturation implied by the target formula.

    Rings plus pi bonds are strongly constrained by a formula, so this
    catches candidates that satisfy the atom counts with the wrong skeleton.
    """

    name = "unsaturation"

    def __init__(self, target_counts: dict[str, int]) -> None:
        self.target_dou = self.degree_of_unsaturation(target_counts)

    @staticmethod
    def degree_of_unsaturation(counts: dict[str, int]) -> float:
        carbon = counts.get("C", 0)
        hydrogen = counts.get("H", 0)
        nitrogen = counts.get("N", 0)
        halogens = sum(counts.get(x, 0) for x in ("F", "Cl", "Br", "I"))
        return (2 * carbon + 2 + nitrogen - hydrogen - halogens) / 2

    def verify(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -1.0
        mol = Chem.AddHs(mol)
        counts: dict[str, int] = {}
        for atom in mol.GetAtoms():
            counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
        gap = abs(self.degree_of_unsaturation(counts) - self.target_dou)
        return -min(gap / max(self.target_dou, 1.0), 1.0)
