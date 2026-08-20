from functools import partial

from loguru import logger

from secs.elucidation.objective import Objective
from secs.elucidation.optimizers.base import MoleculeOptimizer, OptimizerResult


class GraphGAOptimizer(MoleculeOptimizer):
    """Graph-crossover/mutation GA over SMILES.

    `mol_ga` is an optional dependency: it is imported inside :meth:`run` so
    that installing SECS without the ``elucidation`` extra still lets the rest
    of the package import.
    """

    def __init__(
        self,
        population_size: int = 512,
        offspring_size: int = 1024,
        max_generations: int = 10,
        frac_graph_ga_mutate: float = 0.3,
        seed: int = 42,
        callbacks=None,
    ) -> None:
        super().__init__(seed=seed, callbacks=callbacks)
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.max_generations = max_generations
        self.frac_graph_ga_mutate = frac_graph_ga_mutate

    def run(self, initial_population: list[str], objective: Objective) -> OptimizerResult:
        try:
            from mol_ga.graph_ga.gen_candidates import graph_ga_blended_generation  # noqa: PLC0415
            from mol_ga.preconfigured_gas import default_ga  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "GraphGAOptimizer needs the 'mol_ga' package. Install it with: pip install 'secs[elucidation]'"
            ) from error

        if not initial_population:
            raise ValueError("GraphGAOptimizer needs a non-empty starting population.")

        cached = self._wrap(objective, initial_population)

        result = default_ga(
            starting_population_smiles=initial_population,
            scoring_function=cached,
            max_generations=self.max_generations,
            offspring_size=self.offspring_size,
            population_size=self.population_size,
            logger=logger,
            rng=self.rng,
            offspring_gen_func=partial(graph_ga_blended_generation, frac_graph_ga_mutate=self.frac_graph_ga_mutate),
        )

        # mol_ga yields (score, smiles); OptimizerResult stores (smiles, score).
        population = sorted(((smiles, score) for score, smiles in result.population), key=lambda p: p[1], reverse=True)
        state = cached.state
        return OptimizerResult(
            population=population,
            n_evaluated=state.n_evaluated,
            generations=state.generation,
            all_scored=state.ranked(),
        )
