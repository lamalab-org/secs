import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem

MAX_SPHERE = 6


def sphere_keys(mol, atom_index: int, max_sphere: int = MAX_SPHERE) -> list[str]:
    """Canonical descriptions of an atom's environment, sphere by sphere.

    A HOSE code in the sense that matters here: a key for the atom's
    surroundings out to a given radius, so shifts can be looked up by
    environment and fall back to a smaller radius when nothing matches.
    Uses RDKit's canonical rooted SMILES of the environment subgraph rather
    than Bremser's original string grammar -- same lookup behaviour, and it
    stays consistent between table and query by construction.
    """
    atom = mol.GetAtomWithIdx(atom_index)
    base = f"{atom.GetSymbol()}{atom.GetTotalNumHs()}{'a' if atom.GetIsAromatic() else ''}{atom.GetFormalCharge():+d}"
    keys = [f"0:{base}"]
    for radius in range(1, max_sphere + 1):
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_index)
        if not env:
            break
        amap: dict[int, int] = {}
        try:
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            if atom_index not in amap:
                break
            smiles = Chem.MolToSmiles(submol, rootedAtAtom=amap[atom_index], canonical=True)
        except Exception:
            # A fragment RDKit cannot canonicalise (atropisomer bookkeeping,
            # broken ring stereo). Keep the smaller spheres already collected
            # rather than losing the atom entirely.
            break
        keys.append(f"{radius}:{base}:{smiles}")
    return keys


class HoseShiftTable:
    """Maps atom environments to observed 13C shifts.

    Prediction is a hash lookup at the largest sphere that has enough
    examples, falling back inward. No geometry, no neural network -- the cost
    is graph traversal, which is why it can run inside a search loop.
    """

    def __init__(self, table: dict[str, tuple[float, int]], min_count: int = 1) -> None:
        self.table = table
        self.min_count = min_count

    @classmethod
    def build(cls, records, max_sphere: int = MAX_SPHERE, min_count: int = 1):
        """Accumulate shifts per environment key from (mol, {atom_index: shift})."""
        buckets: dict[str, list[float]] = defaultdict(list)
        for mol, shifts in records:
            for atom_index, shift in shifts.items():
                for key in sphere_keys(mol, atom_index, max_sphere):
                    buckets[key].append(shift)
        table = {key: (float(np.median(values)), len(values)) for key, values in buckets.items()}
        return cls(table, min_count=min_count)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"min_count": self.min_count, "table": self.table}))

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text())
        return cls(payload["table"], min_count=payload["min_count"])

    def predict_atom(self, mol, atom_index: int) -> tuple[float | None, int]:
        """Shift for one atom, plus the sphere it was found at (-1 if unknown)."""
        for key in reversed(sphere_keys(mol, atom_index)):
            hit = self.table.get(key)
            if hit is not None and hit[1] >= self.min_count:
                return hit[0], int(key.split(":", 1)[0])
        return None, -1

    def predict(self, smiles: str) -> np.ndarray | None:
        """Predicted 13C shifts for every carbon, or None if unparseable."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        shifts = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            value, _ = self.predict_atom(mol, atom.GetIdx())
            if value is not None:
                shifts.append(value)
        return np.array(shifts, dtype=float) if shifts else None


class HoseShiftSimulator:
    """SpectrumSimulator backed by a HOSE table.

    Interchangeable with HttpShiftSimulator, so a search can swap a fast
    lookup for the slow simulator without touching the objective.
    """

    def __init__(self, table: HoseShiftTable, modality: str = "c_nmr") -> None:
        self.table = table
        self.modality = modality

    def simulate(self, smiles: list[str]) -> list[np.ndarray | None]:
        return [self.table.predict(s) for s in smiles]
