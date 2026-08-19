from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Objective(Protocol):
    """Scores a batch of SMILES. Higher is better."""

    def __call__(self, smiles: list[str]) -> np.ndarray: ...


class ScoringComponent:
    """One named term of an objective.

    Subclasses implement :meth:`score`; the name is used for reporting a
    breakdown of how each term contributed to a candidate's total.
    """

    name: str = "component"

    def score(self, smiles: list[str]) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, smiles: list[str]) -> np.ndarray:
        return self.score(smiles)


class WeightedObjective:
    """Weighted sum of scoring components.

    With every weight at 1.0 this reproduces the original reward:
    ``mean_cosine_similarity + formula_penalty - validity_penalty``.
    """

    def __init__(self, components: list[tuple[float, ScoringComponent]]) -> None:
        if not components:
            raise ValueError("WeightedObjective needs at least one component.")
        self.components = components

    def __call__(self, smiles: list[str]) -> np.ndarray:
        if not smiles:
            return np.array([])
        total = np.zeros(len(smiles), dtype=float)
        for weight, component in self.components:
            total += weight * np.asarray(component.score(smiles), dtype=float)
        return total

    def breakdown(self, smiles: list[str]) -> dict[str, np.ndarray]:
        """Per-component contributions, for inspecting why a candidate scored as it did."""
        if not smiles:
            return {component.name: np.array([]) for _, component in self.components}
        return {
            component.name: weight * np.asarray(component.score(smiles), dtype=float) for weight, component in self.components
        }
