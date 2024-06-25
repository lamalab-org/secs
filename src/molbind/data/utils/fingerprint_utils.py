from collections.abc import Iterable  # noqa: I002
from typing import List  # noqa: UP035

import numpy as np
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles, MolToSmiles


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


def compute_morgan_fingerprints(
    smiles_list: Iterable[str],  # list of SMILEs
    n_bits: int = 2048,  # number of bits in the fingerprint
) -> np.ndarray:
    rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
    rdkit_smiles = [MolToSmiles(mol, isomericSmiles=False) for mol in rdkit_mols]
    rdkit_mols = [MolFromSmiles(smiles) for smiles in rdkit_smiles]
    X = [
        AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=n_bits)
        for mol in rdkit_mols
    ]
    return np.asarray(X)


def compute_fragprint(smiles: str) -> List[float]:  # noqa: UP006
    X = compute_morgan_fingerprints([smiles])
    fragments = {d[0]: d[1] for d in Descriptors.descList[115:]}
    X1 = np.zeros((1, len(fragments)))
    mol = MolFromSmiles(smiles)
    features = [fragments[d](mol) for d in fragments]
    X1[0, :] = features
    fingerprint = np.concatenate((X, X1), axis=1)
    return fingerprint.tolist()[0]


def fps(ls):
    if ls[1] is None:
        print("None")
        return compute_fragprint(ls[0])
    else:
        print(ls[1])
        return ls[1]
