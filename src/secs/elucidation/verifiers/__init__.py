from class_resolver import ClassResolver

from secs.elucidation.verifiers.base import CallableSimulator, SpectrumSimulator, Verifier
from secs.elucidation.verifiers.counting import (
    PeakCountVerifier,
    n_distinct_environments,
)
from secs.elucidation.verifiers.metrics import greedy_peak_distance, hungarian_peak_distance
from secs.elucidation.verifiers.remote import HttpShiftSimulator
from secs.elucidation.verifiers.shifts import SimulatedShiftVerifier

VERIFIER_RESOLVER: ClassResolver[Verifier] = ClassResolver(
    [PeakCountVerifier, SimulatedShiftVerifier],
    base=Verifier,
    suffix="Verifier",
)

__all__ = [
    "VERIFIER_RESOLVER",
    "CallableSimulator",
    "HttpShiftSimulator",
    "PeakCountVerifier",
    "SimulatedShiftVerifier",
    "SpectrumSimulator",
    "Verifier",
    "greedy_peak_distance",
    "hungarian_peak_distance",
    "n_distinct_environments",
]
