"""Structure elucidation: search for molecules whose embeddings match a spectrum.

The pipeline separates three concerns that were previously entangled in one
script:

- ``embedding``   turn SMILES into per-modality embeddings with trained models
- ``objective``   score candidates (composable, weighted components)
- ``optimizers``  propose candidates (pluggable search strategies)

An optimiser only ever sees an objective, so adding a search algorithm never
requires touching the scoring code, and vice versa.
"""

from secs.elucidation.caching import (
    CachedObjective,
    CacheState,
    LogBestCallback,
    ProgressCallback,
    SnapshotCallback,
)
from secs.elucidation.components import (
    EmbeddingSimilarity,
    FormulaPenalty,
    SyntheticAccessibility,
    ValidityPenalty,
    spectral_objective,
)
from secs.elucidation.embedding import SmilesEmbedder, load_model, load_models
from secs.elucidation.objective import Objective, ScoringComponent, WeightedObjective
from secs.elucidation.optimizers import (
    OPTIMIZER_RESOLVER,
    GraphGAOptimizer,
    MoleculeOptimizer,
    OptimizerResult,
    ScoreOnlyOptimizer,
)

__all__ = [
    "OPTIMIZER_RESOLVER",
    "CacheState",
    "CachedObjective",
    "EmbeddingSimilarity",
    "FormulaPenalty",
    "GraphGAOptimizer",
    "LogBestCallback",
    "MoleculeOptimizer",
    "Objective",
    "OptimizerResult",
    "ProgressCallback",
    "ScoreOnlyOptimizer",
    "ScoringComponent",
    "SmilesEmbedder",
    "SnapshotCallback",
    "SyntheticAccessibility",
    "ValidityPenalty",
    "WeightedObjective",
    "load_model",
    "load_models",
    "spectral_objective",
]
