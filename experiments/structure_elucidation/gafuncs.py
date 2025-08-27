from collections.abc import Callable
from functools import cache

import numpy as np
from loguru import logger
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Contrib.SA_Score import sascorer


class CachedFunction:
    """Function which caches previously computed values to avoid repeat computation."""

    def __init__(
        self,
        f: Callable,
        original_smiles: str | None = None,
    ):
        """Init function

        :type f: callable
        :type original_smiles: str
        """
        self._f = f
        self.cache = {}
        self.best_smiles: tuple[float, str] = None
        self.original_smiles = original_smiles

    def eval_batch(self, inputs):
        # Eval function at non-cached inputs
        inputs_not_cached = [x for x in inputs if x not in self.cache]
        outputs_not_cached = self._batch_f_eval(inputs_not_cached)

        # Add new values to cache
        for x, y in zip(inputs_not_cached, outputs_not_cached):
            self.cache[x] = y
        return [self.cache[x] for x in inputs]

    def __call__(self, inputs, batch=True):
        # Ensure it is in batch form
        return self.eval_batch(inputs) if batch else self.eval_non_batch(inputs)


class CachedBatchFunction(CachedFunction):
    def _batch_f_eval(self, input_list):
        # this gen results
        # current best
        scores = self._f(input_list)
        # log the best 3 smiles
        score_list = list(zip(scores, input_list))
        # one best
        best_score, best_smiles = max(score_list, key=lambda x: x[0])
        if self.best_smiles is None or best_score > self.best_smiles[0]:
            self.best_smiles = (best_score, best_smiles)
            if self.best_smiles == self.original_smiles:
                logger.info("THE CORRECT SMILES WAS FOUND!!!🎉")
        logger.info(f"Best smiles: {self.best_smiles}")
        return scores


@cache
def get_number_of_topologically_distinct_atoms(smiles: str, atomic_number: int = 1):
    """Return the number of unique `element` environments based on environmental topology.
    This corresponds to the number of peaks one could maximally observe in an NMR spectrum.
    Args:
        smiles (str): SMILES string
        atomic_number (int, optional): Atomic number. Defaults to 1.

    Returns:
        int: Number of unique environments.

    Raises:
        ValueError: If not a valid SMILES string

    Example:
        >>> get_number_of_topologically_distinct_atoms("CCO", 1)
        3

        >>> get_number_of_topologically_distinct_atoms("CCO", 6)
        2
    """

    try:
        molecule = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(molecule) if atomic_number == 1 else molecule
        # Get unique canonical atom rankings
        atom_ranks = list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))

        # Select the unique element environments
        atom_ranks = np.array(atom_ranks)

        # Atom indices
        atom_indices = np.array([atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == atomic_number])
        return len(set(atom_ranks[atom_indices]))
    except (TypeError, ValueError, AttributeError, IndexError):
        return len(smiles)


@cache
def sascore(smiles):
    try:
        m = MolFromSmiles(smiles)
        return sascorer.calculateScore(m)
    except:
        return 10


def smiles_is_radical_or_is_charged_or_has_wrong_valence(smiles: str) -> bool:
    """
    Determines if a SMILES string represents a radical, charged molecule, or has wrong valence.

    Args:
        smiles (str): SMILES string representation of a molecule

    Returns:
        bool: True if the molecule is a radical OR charged OR has wrong valence, False otherwise
    """
    try:
        # Parse the SMILES string into a molecule object - without sanitization first
        mol = Chem.MolFromSmiles(smiles, sanitize=False)

        # Return False if SMILES is invalid
        if mol is None:
            return False

        # Check 1: Overall charge neutrality (before adding hydrogens)
        total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        if total_charge != 0:
            return True  # Molecule is charged

        # Check 2: Valence validity - try to sanitize
        try:
            # This will raise an exception if valence is invalid
            Chem.SanitizeMol(mol)
        except Exception as e:
            return True  # Molecule has wrong valence

        # Add hydrogens after sanitization succeeds
        mol = Chem.AddHs(mol)

        # Check 3: Unpaired electrons (radicals)
        for atom in mol.GetAtoms():
            # Get the number of radical electrons (unpaired electrons)
            num_radical_electrons = atom.GetNumRadicalElectrons()

            # If any atom has unpaired electrons, it's a radical
            if num_radical_electrons > 0:
                return True  # Molecule is a radical

        return False  # Molecule is neutral, has valid valence, and no radicals

    except Exception:
        # Return True for any parsing errors (likely invalid structures)
        return True
