from collections.abc import Callable
from functools import cache

import numpy as np
import xgboost
from loguru import logger
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MolFromSmiles
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
        for x, y in zip(inputs_not_cached, outputs_not_cached, strict=False):
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
        score_list = list(zip(scores, input_list, strict=False))
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
        except Exception:
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


def smiles_to_morgan_fingerprint(smiles, n_bits=4096, radius=6):
    """
    Converts a SMILES string into a Morgan fingerprint.

    Args:
        smiles (str): The SMILES representation of the molecule.
        n_bits (int): The size of the fingerprint bit vector.
        radius (int): The radius for the Morgan fingerprint algorithm.

    Returns:
        np.array or None: The fingerprint as a NumPy array, or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Could not parse SMILES: {smiles}")
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def predict_match_probability(model, smiles, spectrum, fp_bits=4096, fp_radius=6):
    """
    Predicts the probability that a SMILES string and a spectrum are a correct match.

    Args:
        model (xgb.XGBClassifier): The trained XGBoost model.
        smiles (str): The SMILES representation of the molecule.
        spectrum (list or np.array): The NMR spectrum data.
        fp_bits (int): The size of the fingerprint bit vector (must match training).
        fp_radius (int): The radius for the Morgan fingerprint (must match training).

    Returns:
        float: The predicted probability of a match (between 0.0 and 1.0).
               Returns None if the SMILES string is invalid.
    """
    # logger.info(f"Generating fingerprint for SMILES: {smiles}")

    # 1. Generate the Morgan fingerprint for the input SMILES
    fingerprint = smiles_to_morgan_fingerprint(smiles, n_bits=fp_bits, radius=fp_radius)

    if fingerprint is None:
        logger.error("Failed to generate fingerprint. Cannot make a prediction.")
        return None

    # 2. Ensure the spectrum is a NumPy array
    spectrum_arr = np.array(spectrum)

    # 3. Combine the fingerprint and spectrum to create the feature vector
    # The shape must be (1, num_features) for a single prediction
    feature_vector = np.concatenate([fingerprint, spectrum_arr]).reshape(1, -1)

    # logger.info("Predicting match probability...")

    # 4. Use the model to predict the probability
    # model.predict_proba returns probabilities for each class [class_0, class_1]
    # We want the probability of class 1 (a match)
    probability = model.predict_proba(feature_vector)[:, 1]

    return probability[0]


def load_model_from_path(model_path):
    """
    Loads a saved XGBoost model from a file.

    Args:
        model_path (str): The path to the saved model file (e.g., 'checkpoints_xgb/xgboost_model.json').

    Returns:
        xgb.XGBClassifier: The loaded XGBoost model.
    """
    logger.info(f"Loading model from {model_path}...")
    loaded_model = xgboost.XGBClassifier()
    loaded_model.load_model(model_path)
    logger.info("Model loaded successfully.")
    return loaded_model
