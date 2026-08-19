"""Molecule search strategies, addressable by name.

Register a new optimiser here and it becomes selectable from a config string,
the same way activations are resolved in `secs.models.components.head`.
"""

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
)

__all__ = [
    "OPTIMIZER_RESOLVER",
    "GraphGAOptimizer",
    "MoleculeOptimizer",
    "OptimizerResult",
    "ScoreOnlyOptimizer",
]
