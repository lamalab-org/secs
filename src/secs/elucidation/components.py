import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity

from secs.elucidation.embedding import SmilesEmbedder
from secs.elucidation.molecules import (
    atom_counts,
    is_radical_charged_or_wrong_valence,
    synthetic_accessibility,
)
from secs.elucidation.objective import ScoringComponent


class EmbeddingSimilarity(ScoringComponent):
    """Mean cosine similarity between candidate SMILES and target spectra embeddings.

    Only modalities present in *both* the embedder and the targets contribute.
    If none do, every candidate scores 0 -- the remaining components then decide
    the ranking, rather than the objective silently collapsing.
    """

    name = "embedding_similarity"

    def __init__(self, embedder: SmilesEmbedder, targets: dict[str, Tensor]) -> None:
        self.embedder = embedder
        self.targets = targets

    def score(self, smiles: list[str]) -> np.ndarray:
        candidate_embeddings = self.embedder.encode(smiles)

        per_modality = []
        for modality, target in self.targets.items():
            embedding = candidate_embeddings.get(modality)
            if embedding is None or embedding.nelement() == 0:
                continue
            similarity = cosine_similarity(target.unsqueeze(0).cpu(), embedding.cpu(), dim=1)
            per_modality.append(similarity.numpy())

        if not per_modality:
            return np.zeros(len(smiles), dtype=float)
        return np.mean(np.stack(per_modality), axis=0)


class FormulaPenalty(ScoringComponent):
    """Penalises deviation from a target molecular formula.

    Returns 0 for an exact match and increasingly negative values as the atom
    counts diverge, normalised by the target's total atom count so the term is
    comparable across molecule sizes. Unparseable SMILES get a large fixed
    penalty.
    """

    name = "formula_penalty"

    def __init__(self, target_counts: dict[str, int], invalid_penalty: float = -1000.0) -> None:
        self.target_counts = target_counts
        self.invalid_penalty = invalid_penalty
        self.total_atoms = sum(target_counts.values()) or 1.0

    def _score_one(self, smiles: str) -> float:
        counts = atom_counts(smiles)
        if not counts:
            return self.invalid_penalty
        deviation = sum(
            abs(counts.get(element, 0) - self.target_counts.get(element, 0)) for element in set(counts) | set(self.target_counts)
        )
        return -deviation / self.total_atoms

    def score(self, smiles: list[str]) -> np.ndarray:
        return np.array([self._score_one(s) for s in smiles], dtype=float)


class ValidityPenalty(ScoringComponent):
    """Scores -1 for radicals, charged species and valence errors, else 0."""

    name = "validity_penalty"

    def __init__(self, penalty: float = -1.0) -> None:
        self.penalty = penalty

    def score(self, smiles: list[str]) -> np.ndarray:
        return np.array(
            [self.penalty if is_radical_charged_or_wrong_valence(s) else 0.0 for s in smiles],
            dtype=float,
        )


class SyntheticAccessibility(ScoringComponent):
    """Rewards easily synthesisable molecules: ``-(SA score) / 10`` in [-1, 0].

    Not part of the original reward; available for composing into an objective
    when synthesisability matters.
    """

    name = "synthetic_accessibility"

    def score(self, smiles: list[str]) -> np.ndarray:
        return np.array([-synthetic_accessibility(s) / 10.0 for s in smiles], dtype=float)


def _ensure_1d(target: Tensor) -> Tensor:
    """Reduce a target embedding to 1-D, mean-pooling a sequence dimension if present."""
    if target.ndim > 1 and target.shape[0] == 1:
        target = target.squeeze(0)
    if target.ndim == 2:
        target = torch.mean(target, dim=0)
    if target.ndim != 1:
        raise ValueError(f"Target embedding must reduce to 1-D, got shape {tuple(target.shape)}.")
    return target


def spectral_objective(
    embedder: SmilesEmbedder,
    targets: dict[str, Tensor],
    target_counts: dict[str, int],
    similarity_weight: float = 1.0,
    formula_weight: float = 1.0,
    validity_weight: float = 1.0,
):
    """The default elucidation objective.

    Equivalent to the original ``reward_function_ga``: mean cosine similarity
    across modalities, plus a molecular-formula penalty, minus a validity
    penalty.
    """
    from secs.elucidation.objective import WeightedObjective  # noqa: PLC0415  (avoids a cycle)

    normalised = {modality: _ensure_1d(target) for modality, target in targets.items()}
    return WeightedObjective(
        [
            (similarity_weight, EmbeddingSimilarity(embedder, normalised)),
            (formula_weight, FormulaPenalty(target_counts)),
            (validity_weight, ValidityPenalty()),
        ]
    )
