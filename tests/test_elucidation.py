import numpy as np
import pytest
import torch

from secs.elucidation import (
    OPTIMIZER_RESOLVER,
    CachedObjective,
    FaissCandidateSource,
    FormulaPenalty,
    OptimizerResult,
    ScoreOnlyOptimizer,
    StaticCandidateSource,
    TrajectoryCallback,
    ValidityPenalty,
    WeightedObjective,
)
from secs.elucidation.components import EmbeddingSimilarity, _ensure_1d
from secs.elucidation.molecules import atom_counts, is_radical_charged_or_wrong_valence
from secs.elucidation.objective import ScoringComponent
from secs.utils.elucidation import (
    build_formula_string,
    gen_close_molformulas_from_seed,
    get_atom_counts_from_formula,
)


class StubEmbedder:
    """Returns a fixed embedding per SMILES, so similarity is predictable."""

    def __init__(self, table: dict[str, list[float]], modality: str = "hnmr") -> None:
        self.table = table
        self.modality = modality

    def encode(self, smiles: list[str]) -> dict[str, torch.Tensor]:
        return {self.modality: torch.tensor([self.table[s] for s in smiles], dtype=torch.float32)}


class ConstantComponent(ScoringComponent):
    name = "constant"

    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, smiles):
        return np.full(len(smiles), self.value, dtype=float)


# --- molecules -------------------------------------------------------------


def test_atom_counts_includes_explicit_hydrogens():
    assert atom_counts("CCO") == {"C": 2, "O": 1, "H": 6}


def test_atom_counts_returns_none_for_unparseable_smiles():
    assert atom_counts("not_a_smiles") is None


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [("CCO", False), ("c1ccccc1", False), ("[CH3]", True), ("CC(=O)[O-]", True)],
)
def test_validity_detection(smiles, expected):
    assert is_radical_charged_or_wrong_valence(smiles) is expected


# --- objective composition -------------------------------------------------


def test_weighted_objective_sums_components():
    objective = WeightedObjective([(1.0, ConstantComponent(2.0)), (0.5, ConstantComponent(4.0))])
    assert objective(["CCO", "CC"]).tolist() == [4.0, 4.0]


def test_weights_scale_contributions():
    objective = WeightedObjective([(3.0, ConstantComponent(2.0))])
    assert objective(["CCO"]).tolist() == [6.0]


def test_empty_input_gives_empty_scores():
    objective = WeightedObjective([(1.0, ConstantComponent(1.0))])
    assert objective([]).size == 0


def test_objective_requires_a_component():
    with pytest.raises(ValueError, match="at least one component"):
        WeightedObjective([])


def test_breakdown_reports_each_component_separately():
    objective = WeightedObjective([(1.0, ConstantComponent(2.0)), (2.0, ValidityPenalty())])
    breakdown = objective.breakdown(["CCO", "[CH3]"])
    assert breakdown["constant"].tolist() == [2.0, 2.0]
    assert breakdown["validity_penalty"].tolist() == [0.0, -2.0]


# --- components ------------------------------------------------------------


def test_formula_penalty_is_zero_for_exact_match():
    penalty = FormulaPenalty(atom_counts("CCO"))
    assert penalty.score(["CCO"])[0] == 0.0


def test_formula_penalty_grows_with_deviation():
    penalty = FormulaPenalty(atom_counts("CCO"))
    close, far = penalty.score(["CCCO", "CCCCCCCCO"])
    assert close > far


def test_formula_penalty_flags_invalid_smiles():
    penalty = FormulaPenalty(atom_counts("CCO"))
    assert penalty.score(["not_a_smiles"])[0] == -1000.0


def test_embedding_similarity_ranks_the_matching_molecule_highest():
    embedder = StubEmbedder({"CCO": [1.0, 0.0], "CCC": [0.0, 1.0]})
    component = EmbeddingSimilarity(embedder, {"hnmr": torch.tensor([1.0, 0.0])})
    aligned, orthogonal = component.score(["CCO", "CCC"])
    assert aligned == pytest.approx(1.0)
    assert orthogonal == pytest.approx(0.0, abs=1e-6)


def test_embedding_similarity_is_zero_when_no_modality_matches():
    embedder = StubEmbedder({"CCO": [1.0, 0.0]}, modality="hnmr")
    component = EmbeddingSimilarity(embedder, {"ir": torch.tensor([1.0, 0.0])})
    assert component.score(["CCO"]).tolist() == [0.0]


def test_ensure_1d_mean_pools_a_sequence_dimension():
    assert _ensure_1d(torch.tensor([[0.0, 0.0], [2.0, 4.0]])).tolist() == [1.0, 2.0]


def test_ensure_1d_squeezes_a_leading_batch_dimension():
    assert _ensure_1d(torch.tensor([[1.0, 2.0]])).tolist() == [1.0, 2.0]


# --- caching ---------------------------------------------------------------


def test_cache_evaluates_each_molecule_once():
    seen = []

    def objective(smiles):
        seen.append(list(smiles))
        return np.arange(len(smiles), dtype=float)

    cached = CachedObjective(objective)
    cached.eval_batch(["CCO", "CCC"])
    cached.eval_batch(["CCO", "CCCC"])
    assert seen == [["CCO", "CCC"], ["CCCC"]]


def test_cached_scores_are_stable_across_batches():
    cached = CachedObjective(lambda s: np.array([len(x) for x in s], dtype=float))
    first = cached.eval_batch(["CCO", "CC"])
    second = cached.eval_batch(["CC", "CCO"])
    assert first == [3.0, 2.0]
    assert second == [2.0, 3.0]


def test_generation_counts_batches_after_the_initial_population():
    cached = CachedObjective(lambda s: np.zeros(len(s)))
    assert cached.state.generation == 0
    cached.eval_batch(["CCO"])
    assert cached.state.generation == 0  # initial population
    cached.eval_batch(["CCC"])
    assert cached.state.generation == 1


def test_callbacks_receive_progress():
    states = []
    cached = CachedObjective(
        lambda s: np.array([len(x) for x in s], dtype=float),
        callbacks=[states.append],
    )
    cached.eval_batch(["CCO", "CC"])
    assert states[-1].n_evaluated == 2
    assert states[-1].best() == ("CCO", 3.0)


# --- optimisers ------------------------------------------------------------


def test_score_only_optimizer_ranks_the_starting_population():
    embedder = StubEmbedder({"CCO": [1.0, 0.0], "CCC": [0.0, 1.0]})
    objective = WeightedObjective([(1.0, EmbeddingSimilarity(embedder, {"hnmr": torch.tensor([1.0, 0.0])}))])
    result = ScoreOnlyOptimizer().run(["CCC", "CCO"], objective)
    assert [smiles for smiles, _ in result.population] == ["CCO", "CCC"]
    assert result.best[0] == "CCO"


def test_optimizer_result_marks_retrieved_molecules():
    result = OptimizerResult(population=[("CCO", 1.0), ("CCC", 0.5)])
    records = result.to_records(retrieved={"CCO"})
    assert records[0]["retrieved"] is True
    assert records[1]["retrieved"] is False


def test_optimizers_are_resolvable_by_name():

    assert type(OPTIMIZER_RESOLVER.make("score_only")).__name__ == "ScoreOnlyOptimizer"
    assert type(OPTIMIZER_RESOLVER.make("graph_ga")).__name__ == "GraphGAOptimizer"


# --- molecular formula candidates ------------------------------------------


def test_candidate_formulas_include_the_seed_first():
    """The target molecule has exactly the seed formula, so it must be reachable."""

    seed = "C6H5Cl"
    canonical = build_formula_string(get_atom_counts_from_formula(seed))
    candidates = gen_close_molformulas_from_seed(seed)
    assert candidates[0] == canonical
    assert canonical in candidates


def test_candidate_formulas_are_unique():

    candidates = gen_close_molformulas_from_seed("C10H12N2O")
    assert len(candidates) == len(set(candidates))


def test_candidate_formulas_include_neighbours():

    candidates = gen_close_molformulas_from_seed("C6H5Cl")
    assert "C7H7Cl" in candidates
    assert "C6H6ClN" in candidates


def test_candidate_formulas_reject_unparseable_input():

    with pytest.raises(ValueError, match="Invalid seed formula"):
        gen_close_molformulas_from_seed("")


# --- candidate retrieval ---------------------------------------------------


def _toy_index(vectors):
    faiss = pytest.importorskip("faiss")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def test_faiss_source_preserves_spectral_ranking_within_the_formula_filter():
    """The neighbour order decides ranking; the formula filter only removes rows."""

    # row 0 matches the target best, row 2 next, row 1 is orthogonal.
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype="float32")
    source = FaissCandidateSource(
        index=_toy_index(vectors),
        smiles=np.array(["best", "orthogonal", "second"]),
        formulas=np.array(["C6H5Cl", "C6H5Cl", "C6H5Cl"]),
        n_neighbours=3,
    )
    got = source.propose(torch.tensor([1.0, 0.0]), "C6H5Cl", n_candidates=3)
    assert got == ["best", "second", "orthogonal"]


def test_faiss_source_excludes_implausible_formulas():

    vectors = np.array([[1.0, 0.0], [0.99, 0.01]], dtype="float32")
    source = FaissCandidateSource(
        index=_toy_index(vectors),
        smiles=np.array(["wrong_formula", "right_formula"]),
        formulas=np.array(["C99H99", "C6H5Cl"]),
        n_neighbours=2,
    )
    assert source.propose(torch.tensor([1.0, 0.0]), "C6H5Cl", n_candidates=5) == ["right_formula"]


def test_faiss_source_keeps_molecules_with_the_exact_target_formula():
    """Regression: the seed formula must survive the filter."""

    vectors = np.array([[1.0, 0.0]], dtype="float32")
    source = FaissCandidateSource(
        index=_toy_index(vectors),
        smiles=np.array(["exact_match"]),
        formulas=np.array(["C6H5Cl"]),
        n_neighbours=1,
    )
    assert source.propose(torch.tensor([1.0, 0.0]), "C6H5Cl") == ["exact_match"]


def test_faiss_source_falls_back_when_no_formula_matches():

    vectors = np.array([[1.0, 0.0]], dtype="float32")
    source = FaissCandidateSource(
        index=_toy_index(vectors),
        smiles=np.array(["only_option"]),
        formulas=np.array(["C99H99"]),
        n_neighbours=1,
    )
    assert source.propose(torch.tensor([1.0, 0.0]), "C6H5Cl") == ["only_option"]


def test_faiss_source_rejects_mismatched_metadata():

    with pytest.raises(ValueError, match="same length"):
        FaissCandidateSource(index=None, smiles=np.array(["a", "b"]), formulas=np.array(["C"]))


def test_static_source_returns_its_list():

    source = StaticCandidateSource(["CCO", "CCC", "CC"])
    assert source.propose(torch.tensor([1.0]), "C2H6O", n_candidates=2) == ["CCO", "CCC"]


# --- trajectory recording --------------------------------------------------


def test_trajectory_records_one_entry_per_batch():
    recorder = TrajectoryCallback()
    cached = CachedObjective(lambda s: np.array([len(x) for x in s], dtype=float), callbacks=[recorder])
    cached.eval_batch(["CC", "CCC"])
    cached.eval_batch(["CCCC"])
    assert len(recorder.history) == 2
    assert recorder.history[-1]["best_smiles"] == "CCCC"
    assert recorder.history[-1]["n_evaluated"] == 3


def test_trajectory_tracks_whether_the_target_was_reached():
    recorder = TrajectoryCallback(target_smiles="CCCC")
    cached = CachedObjective(lambda s: np.array([len(x) for x in s], dtype=float), callbacks=[recorder])
    cached.eval_batch(["CC"])
    assert recorder.history[-1]["target_seen"] is False
    cached.eval_batch(["CCCC"])
    assert recorder.history[-1]["target_seen"] is True
    assert recorder.history[-1]["target_score"] == 4.0


def test_trajectory_annotations_are_recorded():
    recorder = TrajectoryCallback(annotate=lambda s: {"length": len(s)})
    cached = CachedObjective(lambda s: np.zeros(len(s)), callbacks=[recorder])
    cached.eval_batch(["CCO"])
    assert recorder.history[-1]["length"] == 3
