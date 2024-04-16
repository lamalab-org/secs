from rdkit.Chem import AllChem, MolFromSmiles


def get_morgan_fingerprint_from_smiles(
    smiles: str, radius: int = 4, nbits: int = 2048
) -> list:
    """
    Get Morgan fingerprint of a molecule:

    Args:
        mol (rdkit.Chem.rdchem.Mol): molecule
        radius (int, optional): radius of the fingerprint. Defaults to 2.
        nbits (int, optional): number of bits in the fingerprint. Defaults to 2048.

    Returns:
        list: fingerprint array
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None

    # Generate Morgan fingerprint
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
    return list(fingerprint)
