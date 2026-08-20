from typing import Protocol, runtime_checkable

import numpy as np

from secs.elucidation.objective import ScoringComponent


@runtime_checkable
class SpectrumSimulator(Protocol):
    """Predicts a spectrum from a structure -- the forward direction.

    Retrieval asks "which known molecule has an embedding near this
    spectrum". A simulator asks the falsifiable question instead: "if the
    candidate were the answer, what spectrum would it produce, and does that
    match what was measured".
    """

    modality: str

    def simulate(self, smiles: list[str]) -> list[np.ndarray | None]:
        """Predicted spectrum per molecule; None where prediction failed."""
        ...


class Verifier(ScoringComponent):
    """A scoring component that checks a candidate against the observed data.

    Scores are <= 0: zero means fully consistent with the observation, more
    negative means less consistent. That keeps verifiers composable with the
    existing penalties in a WeightedObjective.
    """

    name = "verifier"

    def score(self, smiles: list[str]) -> np.ndarray:
        return np.array([self.verify(s) for s in smiles], dtype=float)

    def verify(self, smiles: str) -> float:
        raise NotImplementedError


class CallableSimulator:
    """Adapts a plain function into a SpectrumSimulator.

    Lets an existing model be plugged in without writing a class:

        simulator = CallableSimulator("c_nmr", my_model.predict_shifts)
    """

    def __init__(self, modality: str, function) -> None:
        self.modality = modality
        self.function = function

    def simulate(self, smiles: list[str]) -> list[np.ndarray | None]:
        out = []
        for prediction in self.function(smiles):
            out.append(None if prediction is None else np.asarray(prediction, dtype=float))
        return out
