import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_peak_distance(
    observed: np.ndarray,
    predicted: np.ndarray,
    unmatched_penalty: float = 10.0,
) -> float:
    """Mean per-peak cost of the optimal one-to-one assignment, in ppm.

    Peak lists carry no pairing, so matching them is an assignment problem:
    every observed peak should be explained by a *different* predicted peak.
    Solved exactly with the Hungarian algorithm.

    When the lists differ in length the surplus peaks stay unmatched and each
    costs `unmatched_penalty`, so a prediction cannot improve its score by
    omitting hard peaks or by inventing extra ones. Matched pairs are charged
    at most that same penalty, bounding the damage from peaks that have no
    counterpart at all (solvent, impurities).
    """
    observed = np.asarray(observed, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    if observed.size == 0 or predicted.size == 0:
        return float("inf")

    # Costs are capped at `unmatched_penalty`, which makes the assignment
    # robust: a peak nothing can explain -- a solvent line, an impurity, a
    # simulator that skips a nucleus -- costs a bounded amount instead of
    # dominating the total. Without the cap a single DMSO peak at 40 ppm adds
    # ~76 ppm and saturates the score for every candidate alike.
    cost = np.minimum(np.abs(observed[:, None] - predicted[None, :]), unmatched_penalty)
    rows, cols = linear_sum_assignment(cost)

    matched = cost[rows, cols].sum()
    n_unmatched = (observed.size - rows.size) + (predicted.size - cols.size)
    return float((matched + unmatched_penalty * n_unmatched) / max(observed.size, predicted.size))


def greedy_peak_distance(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance, in ppm.

    Cheaper than the Hungarian assignment but permits many-to-one matches, so
    a single predicted peak can explain a whole cluster of observed ones.
    Kept for comparison; prefer `hungarian_peak_distance`.
    """
    observed = np.asarray(observed, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    if observed.size == 0 or predicted.size == 0:
        return float("inf")

    gaps = np.abs(observed[:, None] - predicted[None, :])
    return float(0.5 * (gaps.min(axis=1).mean() + gaps.min(axis=0).mean()))
