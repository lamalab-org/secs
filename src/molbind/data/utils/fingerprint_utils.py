from typing import List  # noqa: UP035, I002

from rdkit.Chem import AllChem, MolFromSmiles


def get_morgan_fingerprint_from_smiles(
    smiles: str, radius: int = 4, nbits: int = 2048
) -> List[int]:  # noqa: UP006
    """
    Get Morgan fingerprint of a molecule:

    Args:
        smiles (str): SMILES of molecule
        radius (int, optional): radius of the fingerprint. Defaults to 4.
        nbits (int, optional): number of bits in the fingerprint. Defaults to 2048.

    Returns:
        list: morgan fingerprint as a list of bits
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None

    # Generate Morgan fingerprint
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
    return list(fingerprint)
