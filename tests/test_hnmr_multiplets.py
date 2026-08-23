"""Reported multiplet lists must survive the round trip into a spectrum with their integrals intact."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from secs.data.components.hnmr_multiplets import (
    Multiplet,
    _multiplet_lines,
    multiplets_to_spectrum,
    parse_multiplets,
    parse_solvent,
    parse_spectrometer_freq_hz,
)

EXAMPLE = (
    "[('m', [], '2H', 7.27, 7.24), ('m', [], '3H', 7.19, 7.15), ('d', ['3.8Hz'], '2H', 3.68, 3.68), "
    "('s', [], '9H', 1.33, 1.33), ('t', ['7.8Hz'], '9H', 0.98, 0.98), ('q', ['7.9Hz'], '6H', 0.81, 0.81)]"
)


def test_parses_reported_fields():
    multiplets = parse_multiplets(EXAMPLE)
    assert [m.multiplicity for m in multiplets] == ["m", "m", "d", "s", "t", "q"]
    assert [m.n_protons for m in multiplets] == [2, 3, 2, 9, 9, 6]
    assert multiplets[2].j_values_hz == [3.8]
    assert multiplets[0].center_ppm == pytest.approx(7.255)
    assert multiplets[0].range_ppm == pytest.approx(0.03)


def test_parses_j_values_given_as_a_string_repr():
    (multiplet,) = parse_multiplets("[('dd', \"['5.2Hz', '8.4Hz']\", '1H', 7.51, 7.51)]")
    assert multiplet.j_values_hz == [5.2, 8.4]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("s", [0]),
        ("d", [1]),
        ("t", [2]),
        ("q", [3]),
        ("dd", [1, 1]),
        ("td", [2, 1]),
        ("hept", [6]),
        ("dh", [1, 6]),  # isopropyl CH: heptet of doublets
        ("hex", [5]),
        ("app d", [1]),
        ("br dt", [1, 2]),
    ],
)
def test_resolved_multiplicity_labels(label, expected):
    (multiplet,) = parse_multiplets(f"[('{label}', ['7.0Hz'], '1H', 3.0, 3.0)]")
    assert multiplet.splitting == expected


@pytest.mark.parametrize("label", ["m", "AB", "ABq", "br"])
def test_unresolved_multiplicity_labels(label):
    (multiplet,) = parse_multiplets(f"[('{label}', [], '1H', 3.0, 2.9)]")
    assert multiplet.splitting is None


def test_broad_signals_are_flagged():
    (broad,) = parse_multiplets("[('brs', [], '1H', 5.0, 5.0)]")
    (sharp,) = parse_multiplets("[('s', [], '1H', 5.0, 5.0)]")
    assert broad.is_broad
    assert not sharp.is_broad


def test_fractional_integration():
    (multiplet,) = parse_multiplets("[('s', [], '0.5H', 5.0, 5.0)]")
    assert multiplet.n_protons == pytest.approx(0.5)


def test_malformed_input_yields_no_multiplets():
    assert parse_multiplets("not a peak list") == []
    assert parse_multiplets("[('s', [], '1H', None, None)]") == []


def test_metadata_helpers():
    assert parse_spectrometer_freq_hz("400 MHz") == 400e6
    assert parse_spectrometer_freq_hz("not_known") is None
    # messy corpus strings: solvent digits must not be read as the frequency
    assert parse_spectrometer_freq_hz("CDCl3 500 MHz") == 500e6
    assert parse_spectrometer_freq_hz("CDCl3400MHz") == 400e6
    assert parse_spectrometer_freq_hz("C2D2Cl4 500 MHz 100 ˚C") == 500e6
    # unphysical values fall back to None (random draw downstream), never 0
    assert parse_spectrometer_freq_hz("0 MHz") is None
    assert parse_spectrometer_freq_hz(0) is None
    assert parse_solvent("CDCl3") == "CDCl3"
    assert parse_solvent("DMSO-d6") == "DMSO-d6"
    assert parse_solvent("not_known") is None


def test_spectrum_has_the_requested_shape_and_axis():
    x, y = multiplets_to_spectrum(EXAMPLE, num_points=5000)
    assert x.shape == y.shape == (5000,)
    assert (x[0], x[-1]) == (-2.0, 10.0)
    assert np.isfinite(y).all()
    assert y.min() >= 0
    assert y.max() == pytest.approx(1.0)


def test_clean_spectrum_preserves_reported_integrals():
    x, y = multiplets_to_spectrum(EXAMPLE, augment=False, normalize=False, spectrometer_freq_hz=400e6)
    area = float(y.sum() * (x[1] - x[0]))
    assert area == pytest.approx(sum(m.n_protons for m in parse_multiplets(EXAMPLE)), rel=0.02)


def test_clean_quartet_shows_four_lines_with_the_reported_spacing():
    x, y = multiplets_to_spectrum(
        "[('q', ['8.0Hz'], '2H', 2.5, 2.5)]", num_points=40000, augment=False, spectrometer_freq_hz=400e6
    )
    lines, _ = find_peaks(y, height=0.05)
    assert len(lines) == 4
    spacings = np.diff(x[lines])
    assert spacings == pytest.approx(8.0 / 400.0, rel=0.05)  # 8 Hz at 400 MHz is 0.02 ppm


def test_signals_outside_the_window_are_dropped():
    _, y = multiplets_to_spectrum("[('s', [], '1H', 42.0, 42.0)]", augment=False, normalize=False)
    assert y.max() == 0.0


def test_seeded_augmentation_is_reproducible_and_unseeded_is_not():
    kwargs = {"spectrometer_freq_hz": 400e6, "solvent": "CDCl3"}
    first = multiplets_to_spectrum(EXAMPLE, rng=np.random.default_rng(7), **kwargs)[1]
    second = multiplets_to_spectrum(EXAMPLE, rng=np.random.default_rng(7), **kwargs)[1]
    third = multiplets_to_spectrum(EXAMPLE, **kwargs)[1]
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_augmentation_keeps_the_reported_signals_in_place():
    # Lines are ~2 px wide and augmentation jitters them sub-pixel, so compare after modest
    # smoothing; the median over seeds tolerates draws where a legitimately huge water peak dominates.
    clean = gaussian_filter1d(multiplets_to_spectrum(EXAMPLE, augment=False, spectrometer_freq_hz=400e6)[1], 15)
    corr = [
        np.corrcoef(
            clean,
            gaussian_filter1d(multiplets_to_spectrum(EXAMPLE, spectrometer_freq_hz=400e6, rng=np.random.default_rng(s))[1], 15),
        )[0, 1]
        for s in range(12)
    ]
    assert np.median(corr) > 0.5


def test_dd_with_one_reported_j_keeps_two_distinct_splittings():
    # The missing second J must not be copied from the first: that would merge the dd into a triplet.
    _, y = multiplets_to_spectrum(
        "[('dd', ['16.0Hz'], '1H', 6.0, 6.0)]", num_points=40000, augment=False, spectrometer_freq_hz=500e6
    )
    lines, _ = find_peaks(y, height=0.05)
    assert len(lines) == 4


def test_unreported_j_falls_back_to_the_corpus_median():
    x, y = multiplets_to_spectrum("[('d', [], '1H', 6.0, 6.0)]", num_points=40000, augment=False, spectrometer_freq_hz=500e6)
    lines, _ = find_peaks(y, height=0.05)
    spacing_hz = (x[lines][1] - x[lines][0]) * 500
    assert spacing_hz == pytest.approx(7.9, abs=0.3)  # median reported J for a solitary doublet


def test_unresolved_multiplet_is_a_line_pattern_inside_its_reported_range():
    """An 'm' is a real pattern of lines, not a smooth hump: a 12 Hz-wide one at
    600 MHz must show resolved structure, and all of it must sit in the reported range."""
    x, y = multiplets_to_spectrum("[('m', [], '2H', 8.03, 8.01)]", num_points=5000, augment=False, spectrometer_freq_hz=600e6)
    lines, _ = find_peaks(y, height=0.08)
    assert len(lines) >= 2
    occupied = x[y > 0.02]
    assert occupied.min() > 8.01 - 0.01
    assert occupied.max() < 8.03 + 0.01


def test_unresolved_multiplet_without_a_range_gets_a_field_scaled_width():
    _, y300 = multiplets_to_spectrum("[('m', [], '2H', 2.35, 2.35)]", num_points=20000, augment=False, spectrometer_freq_hz=300e6)
    _, y600 = multiplets_to_spectrum("[('m', [], '2H', 2.35, 2.35)]", num_points=20000, augment=False, spectrometer_freq_hz=600e6)
    width = lambda y: np.sum(y > 0.02)  # noqa: E731
    assert 1.6 < width(y300) / width(y600) < 2.4  # the same Hz spread is twice as wide in ppm at 300 MHz


def test_sub_pixel_line_placement_conserves_area_on_a_coarse_grid():
    x, y = multiplets_to_spectrum(
        "[('d', ['3.8Hz'], '2H', 3.68, 3.68)]", num_points=5000, augment=False, normalize=False, spectrometer_freq_hz=500e6
    )
    assert float(y.sum() * (x[1] - x[0])) == pytest.approx(2.0, rel=0.01)


@pytest.mark.parametrize("clip", [True, False])
def test_normalized_output_spans_exactly_zero_to_one(clip):
    _, y = multiplets_to_spectrum(EXAMPLE, spectrometer_freq_hz=400e6, rng=np.random.default_rng(1), clip_negative=clip)
    assert float(y.min()) == 0.0
    assert float(y.max()) == 1.0


def test_zero_frequency_is_survivable():
    """A frequency of 0 (unparseable metadata) must fall back to a random draw, not divide by zero."""
    _, y = multiplets_to_spectrum(EXAMPLE, spectrometer_freq_hz=0, rng=np.random.default_rng(1))
    assert np.isfinite(y).all()


def test_augmented_integrals_carry_relative_plus_absolute_error():
    """A 1H and a 9H signal should both wander by ~0.1 H from noise/rounding, plus a few
    percent relative -- so the 9H varies more in H but less in relative terms."""
    errs = {1: [], 9: []}
    for n, deviations in errs.items():
        for seed in range(200):
            lines = _multiplet_lines(Multiplet("s", [], float(n), 3.0, 3.0), 400e6, np.random.default_rng(seed), jitter=True)
            deviations.append(sum(a for _, a in lines) - n)
    assert 0.06 < np.std(errs[1]) < 0.10  # dominated by the +-0.1 H absolute term
    assert np.std(errs[9]) > np.std(errs[1])  # the relative term adds up on big integrals
    assert np.std(errs[9]) / 9 < np.std(errs[1]) / 1  # but relatively it is tighter

    multiplets = [Multiplet("s", [], 3.0, 2.1, 2.1)]
    _, y = multiplets_to_spectrum(multiplets, augment=False)
    assert y.max() == pytest.approx(1.0)
