import numpy as np

from secs.elucidation.verifiers.base import SpectrumSimulator, Verifier
from secs.elucidation.verifiers.metrics import hungarian_peak_distance


class SimulatedShiftVerifier(Verifier):
    """Compares an observed peak list against one predicted from the structure.

    Scores 0 when the simulated peaks land on the observed ones, falling to
    -1 once the mean mismatch reaches `tolerance_ppm`. Candidates the
    simulator cannot handle score `failure_penalty` rather than silently
    passing.

    With `symmetric`, a perfect match scores +1 instead of 0, putting the
    verifier on the same [-1, 1] scale as a cosine similarity. Note this is an
    affine rescaling -- `1 - 2d` is `2 * (-d) + 1` -- so against a fixed set of
    candidates it reorders nothing on its own; what it changes is the weight
    this term carries inside a WeightedObjective, doubling it.
    """

    def __init__(
        self,
        simulator: SpectrumSimulator,
        observed: np.ndarray,
        tolerance_ppm: float = 5.0,
        failure_penalty: float = -1.0,
        symmetric: bool = False,
        metric=hungarian_peak_distance,
        name: str | None = None,
    ) -> None:
        if tolerance_ppm <= 0:
            raise ValueError("tolerance_ppm must be positive.")
        self.simulator = simulator
        self.observed = np.asarray(observed, dtype=float).ravel()
        self.tolerance_ppm = tolerance_ppm
        self.failure_penalty = failure_penalty
        self.symmetric = symmetric
        self.metric = metric
        self.name = name or f"{simulator.modality}_shift_match"

    def score(self, smiles: list[str]) -> np.ndarray:
        # Simulators are usually batched models, so predict for the whole
        # batch rather than one molecule at a time.
        predictions = self.simulator.simulate(smiles)
        out = np.empty(len(smiles), dtype=float)
        for i, predicted in enumerate(predictions):
            if predicted is None or np.asarray(predicted).size == 0:
                out[i] = self.failure_penalty
                continue
            distance = self.metric(self.observed, predicted)
            mismatch = min(distance / self.tolerance_ppm, 1.0)
            out[i] = 1.0 - 2.0 * mismatch if self.symmetric else -mismatch
        return out
