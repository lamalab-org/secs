import numpy as np
import pytest
import requests

from secs.elucidation import (
    CallableSimulator,
    HttpShiftSimulator,
    PeakCountVerifier,
    SimulatedShiftVerifier,
    WeightedObjective,
)
from secs.elucidation.verifiers import VERIFIER_RESOLVER, n_distinct_environments
from secs.elucidation.verifiers.metrics import greedy_peak_distance, hungarian_peak_distance

OBS = np.array([20.0, 21.0, 22.0, 120.0])


# --- assignment metric -----------------------------------------------------


def test_identical_peak_lists_cost_nothing():
    assert hungarian_peak_distance(OBS, OBS) == 0.0


def test_uniform_shift_reports_the_true_mean_error():
    """A 1 ppm shift on every peak must cost exactly 1 ppm."""
    assert hungarian_peak_distance(OBS, OBS + 1.0) == pytest.approx(1.0)


def test_greedy_understates_a_uniform_shift():
    """Why the assignment is worth solving: greedy many-to-one halves the error."""
    assert greedy_peak_distance(OBS, OBS + 1.0) < hungarian_peak_distance(OBS, OBS + 1.0)


def test_collapsed_prediction_is_punished():
    """One line cannot explain four distinct peaks."""
    collapsed = np.full(4, 21.0)
    assert hungarian_peak_distance(OBS, collapsed) > hungarian_peak_distance(OBS, OBS + 1.0)


def test_one_unexplainable_peak_does_not_saturate_the_score():
    """A solvent line must not swamp an otherwise perfect match.

    Observed carries a 40 ppm DMSO peak the simulator never predicts; the
    remaining peaks match exactly. Capping the pairwise cost keeps the total
    bounded so candidates stay distinguishable.
    """
    observed = np.array([40.0, 116.2, 123.7, 129.3])
    good = np.array([116.2, 123.7, 129.3, 130.0])
    bad = np.array([10.0, 60.0, 70.0, 90.0])
    assert hungarian_peak_distance(observed, good, unmatched_penalty=10.0) < 3.0
    assert hungarian_peak_distance(observed, good) < hungarian_peak_distance(observed, bad)


def test_pairwise_cost_is_bounded_by_the_penalty():
    far = np.array([1000.0, 2000.0])
    assert hungarian_peak_distance(np.array([1.0, 2.0]), far, unmatched_penalty=10.0) == pytest.approx(10.0)


def test_matching_is_one_to_one():
    """Two observed peaks cannot both be explained by the same predicted peak."""
    assert hungarian_peak_distance(np.array([10.0, 20.0]), np.array([10.0, 10.0])) > 0.0


def test_surplus_peaks_cost_the_unmatched_penalty():
    # 4 observed vs 5 predicted: 4 match exactly, 1 unmatched at cost 10, over max(4,5).
    predicted = np.append(OBS, 200.0)
    assert hungarian_peak_distance(OBS, predicted, unmatched_penalty=10.0) == pytest.approx(10.0 / 5)


def test_empty_input_is_infinite():
    assert hungarian_peak_distance(np.array([]), OBS) == float("inf")


def test_metric_is_symmetric():
    a, b = np.array([10.0, 20.0, 30.0]), np.array([11.0, 19.0])
    assert hungarian_peak_distance(a, b) == pytest.approx(hungarian_peak_distance(b, a))


# --- environment counting --------------------------------------------------


def test_symmetry_equivalent_carbons_count_once():
    assert n_distinct_environments("c1ccccc1") == 1


def test_substitution_breaks_symmetry():
    assert n_distinct_environments("Cc1ccccc1") == 5


def test_unparseable_smiles_has_no_count():
    assert n_distinct_environments("not_a_smiles") is None


# --- verifiers -------------------------------------------------------------


def test_peak_count_verifier_rejects_molecules_that_cannot_produce_the_peaks():
    verifier = PeakCountVerifier(n_observed_peaks=8)
    enough = verifier.verify("O=c1cnc2ccccc2[nH]1")  # 8 environments
    too_few = verifier.verify("c1ccccc1")  # 1 environment cannot give 8 peaks
    assert enough == pytest.approx(0.0)
    assert too_few < enough


def test_extra_environments_are_free():
    """Peaks may coincide, so more environments than peaks is ordinary overlap."""
    verifier = PeakCountVerifier(n_observed_peaks=8)
    assert verifier.verify("Cc1ccc(C)cc1C(=O)Nc1ccccc1") == pytest.approx(0.0)


def test_shortfall_scales_with_how_far_short_it_falls():
    verifier = PeakCountVerifier(n_observed_peaks=8)
    assert verifier.verify("Cc1ccccc1") > verifier.verify("c1ccccc1")


def test_solvent_tolerance_forgives_surplus_observed_peaks():
    """Solvent signals inflate the observed count through no fault of the candidate."""
    strict = PeakCountVerifier(n_observed_peaks=8, solvent_tolerance=0)
    lenient = PeakCountVerifier(n_observed_peaks=8, solvent_tolerance=2)
    assert lenient.verify("Cc1ccccc1") > strict.verify("Cc1ccccc1")


def test_verifier_scores_are_never_positive():
    verifier = PeakCountVerifier(n_observed_peaks=8)
    assert verifier.score(["O=c1cnc2ccccc2[nH]1", "c1ccccc1", "bad_smiles"]).max() <= 0.0


def test_peak_count_verifier_rejects_a_nonsense_target():
    with pytest.raises(ValueError, match="must be positive"):
        PeakCountVerifier(n_observed_peaks=0)


def test_simulated_shift_verifier_ranks_a_good_simulator_highly():
    simulator = CallableSimulator("c_nmr", lambda smiles: [OBS if s == "right" else OBS + 20.0 for s in smiles])
    verifier = SimulatedShiftVerifier(simulator, observed=OBS, tolerance_ppm=5.0)
    right, wrong = verifier.score(["right", "wrong"])
    assert right == pytest.approx(0.0)
    assert wrong == pytest.approx(-1.0)


def test_simulator_failures_are_penalised_not_ignored():
    simulator = CallableSimulator("c_nmr", lambda smiles: [None for _ in smiles])
    verifier = SimulatedShiftVerifier(simulator, observed=OBS, failure_penalty=-1.0)
    assert verifier.score(["anything"]).tolist() == [-1.0]


def test_verifier_composes_into_an_objective():
    objective = WeightedObjective([(2.0, PeakCountVerifier(n_observed_peaks=8))])
    scores = objective(["O=c1cnc2ccccc2[nH]1", "c1ccccc1"])
    assert scores[0] > scores[1]


def test_verifiers_are_resolvable_by_name():
    assert VERIFIER_RESOLVER.lookup("peak_count") is PeakCountVerifier


# --- remote simulator ------------------------------------------------------


class _StubServer:
    """Minimal stand-in for a shift-prediction service (e.g. CASCADE-2.0)."""

    def __init__(self, table, fail_on=()):
        self.table = table
        self.fail_on = set(fail_on)
        self.batches = []

    def __call__(self, url, json, timeout):  # noqa: ARG002
        batch = json["smiles"]
        self.batches.append(len(batch))
        if any(s in self.fail_on for s in batch):
            raise ConnectionError("simulator down")
        return _StubResponse({"shifts": [self.table.get(s) for s in batch]})


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_http_simulator_returns_predictions_in_order(monkeypatch):

    server = _StubServer({"a": [1.0, 2.0], "b": [3.0]})
    monkeypatch.setattr(requests, "post", server)
    out = HttpShiftSimulator("http://sim").simulate(["a", "b"])
    assert out[0].tolist() == [1.0, 2.0]
    assert out[1].tolist() == [3.0]


def test_http_simulator_marks_unpredictable_molecules_as_none(monkeypatch):

    monkeypatch.setattr(requests, "post", _StubServer({"a": [1.0]}))
    assert HttpShiftSimulator("http://sim").simulate(["a", "unknown"])[1] is None


def test_http_simulator_batches_large_requests(monkeypatch):

    server = _StubServer({f"m{i}": [float(i)] for i in range(10)})
    monkeypatch.setattr(requests, "post", server)
    HttpShiftSimulator("http://sim", batch_size=4).simulate([f"m{i}" for i in range(10)])
    assert server.batches == [4, 4, 2]


def test_a_dead_simulator_does_not_kill_the_search(monkeypatch):
    """The GA must degrade to its other components, not crash."""

    monkeypatch.setattr(requests, "post", _StubServer({}, fail_on=["a"]))
    assert HttpShiftSimulator("http://sim").simulate(["a"]) == [None]


def test_remote_simulator_drives_a_verifier(monkeypatch):
    """End to end: service -> simulator -> verifier -> objective."""

    observed = [20.0, 21.0, 22.0, 120.0]
    monkeypatch.setattr(
        requests,
        "post",
        _StubServer({"right": observed, "wrong": [70.0, 71.0, 72.0, 170.0]}),
    )
    verifier = SimulatedShiftVerifier(HttpShiftSimulator("http://sim"), observed=np.array(observed), tolerance_ppm=5.0)
    right, wrong = verifier.score(["right", "wrong"])
    assert right == pytest.approx(0.0)
    assert wrong < right
