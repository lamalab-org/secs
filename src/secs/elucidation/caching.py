from typing import Protocol

from loguru import logger


class CacheState:
    """Snapshot of an in-flight optimisation, handed to progress callbacks."""

    def __init__(self, cache: dict[str, float], n_batches: int, initial_population: list[str]) -> None:
        self.cache = cache
        self.n_batches = n_batches
        self.initial_population = initial_population

    @property
    def generation(self) -> int:
        """Batches scored, minus the initial population.

        The first batch is the starting population; each later batch is one
        generation of offspring. This is a proxy -- an optimiser that scores
        more than once per generation would inflate it.
        """
        return max(self.n_batches - 1, 0)

    @property
    def n_evaluated(self) -> int:
        return len(self.cache)

    def ranked(self) -> list[tuple[str, float]]:
        """All molecules scored so far, best first."""
        return sorted(self.cache.items(), key=lambda item: item[1], reverse=True)

    def best(self) -> tuple[str, float] | None:
        ranked = self.ranked()
        return ranked[0] if ranked else None


class ProgressCallback(Protocol):
    """Notified after each batch is scored."""

    def __call__(self, state: CacheState) -> None: ...


class CachedObjective:
    """Memoises an objective by SMILES and reports progress.

    Optimisers regenerate the same molecules constantly, so caching is what
    makes the search affordable. Callers pass callbacks rather than
    subclassing.
    """

    def __init__(
        self,
        objective,
        initial_population: list[str] | None = None,
        callbacks: list[ProgressCallback] | None = None,
    ) -> None:
        self._objective = objective
        self.cache: dict[str, float] = {}
        self.initial_population = initial_population or []
        self.callbacks = callbacks or []
        self.n_batches = 0

    @property
    def state(self) -> CacheState:
        return CacheState(self.cache, self.n_batches, self.initial_population)

    def eval_batch(self, smiles: list[str]) -> list[float]:
        uncached = [s for s in smiles if s not in self.cache]
        if uncached:
            scores = self._objective(uncached)
            for candidate, score in zip(uncached, scores, strict=True):
                self.cache[candidate] = float(score)

        self.n_batches += 1
        state = self.state
        for callback in self.callbacks:
            callback(state)

        return [self.cache[s] for s in smiles]

    def eval_single(self, smiles: str) -> float:
        return self.eval_batch([smiles])[0]

    def __call__(self, smiles: list[str] | str) -> list[float] | float:
        if isinstance(smiles, str):
            return self.eval_single(smiles)
        return self.eval_batch(smiles)


class LogBestCallback:
    """Logs the best molecule whenever it improves."""

    def __init__(self, target_smiles: str | None = None) -> None:
        self.target_smiles = target_smiles
        self._best: tuple[str, float] | None = None

    def __call__(self, state: CacheState) -> None:
        best = state.best()
        if best is None:
            return
        if self._best is None or best[1] > self._best[1]:
            self._best = best
            logger.info(f"Generation {state.generation}: best {best[0]} (score {best[1]:.4f})")
            if self.target_smiles is not None and best[0] == self.target_smiles:
                logger.info("Target SMILES recovered.")


class TrajectoryCallback:
    """Records the search trajectory, one entry per scored batch.

    Lets a run be studied after the fact: when the best score improved, how
    many molecules it took, and whether a known answer was ever reached.
    `annotate` adds caller-supplied fields (similarity to a reference, say)
    to each entry.
    """

    def __init__(self, annotate=None, target_smiles: str | None = None) -> None:
        self.annotate = annotate
        self.target_smiles = target_smiles
        self.history: list[dict] = []

    def __call__(self, state: CacheState) -> None:
        best = state.best()
        entry = {
            "generation": state.generation,
            "n_evaluated": state.n_evaluated,
            "best_smiles": best[0] if best else None,
            "best_score": best[1] if best else None,
        }
        if self.target_smiles is not None:
            entry["target_seen"] = self.target_smiles in state.cache
            entry["target_score"] = state.cache.get(self.target_smiles)
        if self.annotate is not None and best is not None:
            entry.update(self.annotate(best[0]))
        self.history.append(entry)

    @property
    def generations(self) -> int:
        return self.history[-1]["generation"] if self.history else 0


class SnapshotCallback:
    """Atomically writes a JSON progress snapshot for an external status reader.

    Written via a temporary file plus replace, because a reader polling this
    path must never observe a half-written file. The snapshot deliberately uses
    a distinct filename from the final result, so its presence is not mistaken
    for completion.
    """

    def __init__(self, path, describe=None) -> None:
        from pathlib import Path  # noqa: PLC0415  (kept local to avoid a hard import for non-file users)

        self.path = Path(path)
        self.describe = describe

    def __call__(self, state: CacheState) -> None:
        import json  # noqa: PLC0415

        results = [
            {"smiles": smiles, "score": score, "retrieved": smiles in self.initial_set(state)}
            | (self.describe(smiles) if self.describe else {})
            for smiles, score in state.ranked()
        ]
        stage = "initial population" if state.n_batches == 1 else f"generation {state.generation}"
        snapshot = {
            "results": results,
            "metadata": {
                "stage": stage,
                "generation": state.generation,
                "n_evaluated": state.n_evaluated,
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        with temp.open("w") as handle:
            json.dump(snapshot, handle)
        temp.replace(self.path)

    @staticmethod
    def initial_set(state: CacheState) -> set[str]:
        return set(state.initial_population)
