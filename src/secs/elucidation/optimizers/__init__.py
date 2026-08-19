from class_resolver import ClassResolver

from secs.elucidation.optimizers.base import (
    MoleculeOptimizer,
    OptimizerResult,
    ScoreOnlyOptimizer,
)
from secs.elucidation.optimizers.graph_ga import GraphGAOptimizer

OPTIMIZER_RESOLVER: ClassResolver[MoleculeOptimizer] = ClassResolver(
    [GraphGAOptimizer, ScoreOnlyOptimizer],
    base=MoleculeOptimizer,
    default=GraphGAOptimizer,
    suffix="Optimizer",
)

__all__ = [
    "OPTIMIZER_RESOLVER",
    "GraphGAOptimizer",
    "MoleculeOptimizer",
    "OptimizerResult",
    "ScoreOnlyOptimizer",
]
