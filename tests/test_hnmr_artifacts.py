"""The instrument-artifact stages: each gate forced on must produce its physical signature."""

import numpy as np
import pytest
from scipy.signal import find_peaks

import secs.data.components.hnmr as hnmr
from secs.data.components.hnmr import _Grid, _Phase, apply_instrument_artifacts, augment
from secs.data.components.hnmr_multiplets import _split_lines, multiplets_to_spectrum

GRID = _Grid(5000)
PHASE = _Phase()


def delta_spectrum(idx=2500):
    y = np.zeros(5000)
    y[idx] = 1.0
    return y


def test_reference_peak_lands_at_zero_ppm(monkeypatch):
    monkeypatch.setattr(hnmr, "TMS_PROBABILITY", 1.0)
    out = hnmr._add_reference_peak(np.zeros(5000), GRID, np.random.default_rng(0), PHASE, 1.0)
    assert abs(GRID.idx_to_ppm(int(np.argmax(out)))) < 0.05


def test_known_impurities_appear_at_their_tabulated_shifts(monkeypatch):
    monkeypatch.setattr(hnmr, "H_IMPURITIES", {k: (1.0, v[1]) for k, v in hnmr.H_IMPURITIES.items()})
    out = hnmr._add_known_impurities(np.zeros(5000), GRID, np.random.default_rng(1), PHASE, 1.0, "CDCl3")
    peaks, _ = find_peaks(out, height=0.001)
    found = [GRID.idx_to_ppm(i) for i in peaks]
    for expected in (1.26, 2.17, 5.30):  # grease, acetone, DCM
        assert any(abs(f - expected) < 0.06 for f in found)


def test_known_impurities_need_a_tabulated_solvent(monkeypatch):
    monkeypatch.setattr(hnmr, "H_IMPURITIES", {k: (1.0, v[1]) for k, v in hnmr.H_IMPURITIES.items()})
    out = hnmr._add_known_impurities(np.zeros(5000), GRID, np.random.default_rng(1), PHASE, 1.0, "D2O")
    assert out.max() == 0.0


def test_spinning_sidebands_flank_the_peak_at_the_spinning_rate(monkeypatch):
    monkeypatch.setattr(hnmr, "SPINNING_SIDEBAND_PROBABILITY", 1.0)
    out = hnmr._add_spinning_sidebands(delta_spectrum(), GRID, np.random.default_rng(3), 400e6)
    side = np.where((out > 1e-4) & (delta_spectrum() == 0))[0]
    offsets_hz = np.abs(side - 2500) / GRID.points_per_ppm * 400
    assert len(side) == 2
    lo, hi = hnmr.SPINNING_RATE_HZ_RANGE
    bin_hz = 400 / GRID.points_per_ppm  # rates are quantised to the grid
    assert all(lo - bin_hz <= o <= hi + bin_hz for o in offsets_hz)


def test_shim_asymmetry_grows_a_tail_on_one_side_only(monkeypatch):
    monkeypatch.setattr(hnmr, "SHIM_ASYMMETRY_PROBABILITY", 1.0)
    out = hnmr._apply_shim_asymmetry(delta_spectrum(), GRID, np.random.default_rng(4))
    left, right = out[2450:2500].sum(), out[2501:2551].sum()
    assert min(left, right) < 0.05 * max(left, right)


def test_truncation_wiggles_ring_but_conserve_area(monkeypatch):
    monkeypatch.setattr(hnmr, "TRUNCATION_PROBABILITY", 1.0)
    base = delta_spectrum()
    out = hnmr._apply_truncation_wiggles(base.copy(), GRID, np.random.default_rng(5))
    assert out.min() < 0  # sinc lobes go negative
    assert out.sum() == pytest.approx(base.sum(), rel=1e-6)


def test_suppression_notches_the_water_region_and_boosts_local_noise(monkeypatch):
    monkeypatch.setattr(hnmr, "SUPPRESSION_PROBABILITY", 1.0)
    water_idx = GRID.ppm_to_idx(3.33)  # water in DMSO-d6
    base = np.zeros(5000)
    base[water_idx] = 1.0
    out = hnmr._apply_solvent_suppression(base.copy(), GRID, np.random.default_rng(6), "DMSO-d6", 1.0)
    assert out[water_idx] < 0.75  # attenuated
    near = out[water_idx + 50 : water_idx + 150].std()
    far = out[100:200].std()
    assert near > 3 * far  # noise pedestal is local


def test_suppression_never_fires_in_chloroform(monkeypatch):
    monkeypatch.setattr(hnmr, "SUPPRESSION_PROBABILITY", 1.0)
    base = np.zeros(5000)
    base[GRID.ppm_to_idx(1.56)] = 1.0
    out = hnmr._apply_solvent_suppression(base.copy(), GRID, np.random.default_rng(6), "CDCl3", 1.0)
    assert np.array_equal(out, base)


def test_baseline_noise_is_correlated_not_white():
    vals = []
    for seed in range(30):
        out = hnmr._add_correlated_noise(np.zeros(4000), np.random.default_rng(seed), 1.0)
        out = out - out.mean()
        vals.append(float((out[:-1] * out[1:]).sum() / (out**2).sum()))
    assert np.mean(vals) > 0.3  # white noise would average ~0


def test_signal_jitter_noise_concentrates_at_the_peaks(monkeypatch):
    monkeypatch.setattr(hnmr, "SIGNAL_JITTER_PROBABILITY", 1.0)
    base = np.zeros(5000)
    base[2000:2100] = 1.0  # a broad plateau with sharp edges
    diffs = []
    for seed in range(10):
        out = hnmr._add_signal_jitter_noise(base.copy(), GRID, np.random.default_rng(seed))
        resid = np.abs(out - base)
        diffs.append(resid[1990:2110].mean() / (resid[100:1000].mean() or 1e-12))
    assert np.mean(diffs) > 10  # noise lives where the signal is


def test_roofed_splitting_conserves_area_and_leans():
    flat = _split_lines([(2.5, 1.0)], 3, 0.02)
    tilted = _split_lines([(2.5, 1.0)], 3, 0.02, tilt=0.3)
    assert sum(a for _, a in tilted) == pytest.approx(1.0)
    assert tilted[0][1] < flat[0][1] < flat[-1][1] * (flat[0][1] / flat[-1][1]) + 1e-9
    assert tilted[-1][1] > flat[-1][1]


def test_artifact_tail_is_seed_stable_and_shape_preserving():
    base = delta_spectrum()
    kwargs = {"solvent_name": "CDCl3"}
    a = apply_instrument_artifacts(base.copy(), GRID, np.random.default_rng(7), PHASE, 400e6, **kwargs)
    b = apply_instrument_artifacts(base.copy(), GRID, np.random.default_rng(7), PHASE, 400e6, **kwargs)
    assert np.array_equal(a, b)
    assert a.shape == base.shape
    assert np.isfinite(a).all()
    assert a.min() >= 0  # default clips


def test_artifact_tail_keeps_negatives_when_asked():
    out = apply_instrument_artifacts(
        delta_spectrum(), GRID, np.random.default_rng(2), PHASE, 400e6, solvent_name="CDCl3", clip_negative=False
    )
    assert out.min() < 0


def test_augment_accepts_solvent_and_stays_seed_stable():
    x = np.zeros(10000)
    x[::400] = 1.0
    a = augment(x, rng=np.random.default_rng(5), solvent="DMSO-d6")
    b = augment(x, rng=np.random.default_rng(5), solvent="DMSO-d6")
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()


def test_roofing_is_field_dependent_and_off_when_clean():
    def outer_ratio(y):
        _, props = find_peaks(y, height=0.05)
        h = props["peak_heights"]
        return h[0] / h[-1] if len(h) == 4 else None

    _, clean = multiplets_to_spectrum(
        "[('q', ['8.0Hz'], '2H', 2.5, 2.5)]", num_points=40000, augment=False, spectrometer_freq_hz=300e6
    )
    assert outer_ratio(clean) == pytest.approx(1.0, abs=0.02)
