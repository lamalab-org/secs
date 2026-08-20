from functools import cache
from pathlib import Path

from loguru import logger
from rdkit import Chem


@cache
def atom_counts(smiles: str) -> dict[str, int] | None:
    """Count atoms per element, including explicit hydrogens.

    Returns None for SMILES RDKit cannot parse.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Invalid SMILES for atom count: {smiles}")
        return None
    mol = Chem.AddHs(mol)
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
    return counts


@cache
def is_radical_charged_or_wrong_valence(smiles: str) -> bool:
    """True if the molecule is a radical, carries a net charge, or fails sanitisation.

    Unparseable SMILES count as True (i.e. penalised), matching the original
    behaviour: something we cannot even read should not score well.
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False

        if sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != 0:
            return True

        try:
            Chem.SanitizeMol(mol)
        except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
            return True

        mol = Chem.AddHs(mol)
        return any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms())
    except (ValueError, RuntimeError):
        return True


@cache
def synthetic_accessibility(smiles: str) -> float:
    """SA score (1 = easy, 10 = hard). Returns 10.0 when it cannot be computed.

    RDKit ships `sascorer` as a contrib module that is not on the import path
    by default, so the path append and import are deferred to first use rather
    than run at module import.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 10.0
    try:
        return _sascorer().calculateScore(mol)
    except (ValueError, RuntimeError, ZeroDivisionError):
        return 10.0


@cache
def _sascorer():
    """Import RDKit's contrib SA_Score module, adding it to sys.path first."""
    import sys  # noqa: PLC0415

    from rdkit.Chem import RDConfig  # noqa: PLC0415

    sa_path = str(Path(RDConfig.RDContribDir) / "SA_Score")
    if sa_path not in sys.path:
        sys.path.append(sa_path)
    import sascorer  # noqa: PLC0415

    return sascorer
