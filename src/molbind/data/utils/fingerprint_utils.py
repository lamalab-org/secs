import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem


def get_morgan_fingerprint_from_smiles(
    smiles: str, radius: int = 2, nbits: int = 2048
) -> np.array:
    """
    Get Morgan fingerprint of a molecule:

    Args:
        mol (rdkit.Chem.rdchem.Mol): molecule
        radius (int, optional): radius of the fingerprint. Defaults to 2.
        nbits (int, optional): number of bits in the fingerprint. Defaults to 2048.
    Returns:
        np.array: fingerprint array
    """
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint_array = np.zeros(nbits, dtype=np.int4)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
