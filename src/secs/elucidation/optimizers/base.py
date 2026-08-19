import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from secs.elucidation.caching import CachedObjective, ProgressCallback
from secs.elucidation.objective import Objective


@dataclass
class OptimizerResult:
    """Molecules found, best first.

    Stored as ``(smiles, score)``. The underlying mol_ga library yields
    ``(score, smiles)``; normalising here keeps that ordering quirk from
    leaking into callers.
    """

    population: list[tuple[str, float]] = field(default_factory=list)
    n_evaluated: int = 0
    generations: int = 0
    # Every molecule scored during the search, best first. `population` is only
    # what survived selection, so it hides the molecules the objective saw and
    # rejected -- which is most of what an experiment wants to look at.
    all_scored: list[tuple[str, float]] = field(default_factory=list)

    @property
    def best(self) -> tuple[str, float] | None:
        return self.population[0] if self.population else None

    def to_records(self, retrieved: set[str] | None = None) -> list[dict]:
        """Flatten to JSON-friendly records, marking which molecules were retrieved."""
        retrieved = retrieved or set()
        return [{"smiles": smiles, "score": score, "retrieved": smiles in retrieved} for smiles, score in self.population]


class MoleculeOptimizer(ABC):
    """Base class for molecule search strategies."""

    def __init__(self, seed: int = 42, callbacks: list[ProgressCallback] | None = None) -> None:
        self.seed = seed
        self.callbacks = callbacks or []

    def _wrap(self, objective: Objective, initial_population: list[str]) -> CachedObjective:
        return CachedObjective(objective, initial_population=initial_population, callbacks=self.callbacks)

    @abstractmethod
    def run(self, initial_population: list[str], objective: Objective) -> OptimizerResult:
        """Search for molecules maximising *objective*, starting from *initial_population*."""

    @property
    def rng(self) -> random.Random:
        return random.Random(self.seed)


class ScoreOnlyOptimizer(MoleculeOptimizer):
    """Scores the starting population and returns it ranked, proposing nothing new.

    Useful as a baseline -- it isolates how much of the final answer came from
    retrieval rather than from the search -- and it exercises the whole
    objective pipeline without any GA dependency.
    """

    def run(self, initial_population: list[str], objective: Objective) -> OptimizerResult:
        cached = self._wrap(objective, initial_population)
        cached.eval_batch(initial_population)
        state = cached.state
        ranked = state.ranked()
        return OptimizerResult(population=ranked, n_evaluated=state.n_evaluated, generations=0, all_scored=ranked)
