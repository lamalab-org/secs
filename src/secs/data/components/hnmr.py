"""Realistic augmentation of idealised 1H NMR stick spectra.

The augmentation pipeline takes a clean spectrum sampled on a fixed ppm grid,
detects its peaks and re-synthesises them with experimentally motivated
artefacts: pseudo-Voigt line shapes, J-coupling multiplets, phase errors,
13C satellites, residual solvent and water peaks, impurities, baseline drift
and noise.

All randomness flows through a single :class:`numpy.random.Generator`, which
can be supplied to :func:`augment` for reproducible output.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.special import comb

# --- Configuration Constants (Optimized for Realistic Experimental Data) ---
PPM_MIN = -2.0
PPM_MAX = 10.0  # Expanded range for acidic protons, etc.

# A list of common spectrometer frequencies in Hz. One is chosen randomly if not specified.
COMMON_SPECTROMETER_FREQS_HZ = [300e6, 400e6, 500e6, 600e6, 700e6]

# Peak identification parameters (tuned to be less sensitive to noise)
FINDPKS_MIN_PEAK_PROMINENCE_FRAC = 0.008  # Increased to avoid picking up noise
FINDPKS_MIN_PEAK_DIST_PPM = 0.015  # Increased to treat multiplets as single entities
FINDPKS_MIN_HEIGHT_FRAC = 0.015  # Slightly increased to focus on real peaks

# Peak broadening parameters (wider range for more variability)
PEAK_GAUSSIAN_SIGMA_PPM_MIN = 0.0008
PEAK_GAUSSIAN_SIGMA_PPM_MAX = 0.004
PEAK_LORENTZIAN_FWHM_PPM_MIN = 0.0008
PEAK_LORENTZIAN_FWHM_PPM_MAX = 0.012
VOIGT_ETA_MIN = 0.1
VOIGT_ETA_MAX = 0.9

# Global broadening parameters
FINAL_GLOBAL_BROADENING_SIGMA_PPM_MIN = 0.0002
FINAL_GLOBAL_BROADENING_SIGMA_PPM_MAX = 0.0020

# Chemical shift perturbation (more realistic global shift)
MAX_GLOBAL_SHIFT_PPM = 0.05  # Reduced from 0.12, which is a very large error
MAX_LOCAL_SHIFT_PPM = 0.01

# Intensity variation parameters (significantly increased noise and baseline issues)
INTENSITY_NOISE_FACTOR_RANGE = (0.001, 0.01)
PEAK_INTENSITY_VARIATION = 0.12
BASELINE_DRIFT_AMPLITUDE = 0.04  # Increased for more pronounced baseline roll

# Coupling simulation parameters
J_COUPLING_RANGE_HZ = (1.0, 18.0)
COUPLING_PROBABILITY = 0.2
MAX_COUPLING_PARTNERS = 7  # Allow up to an octet

# Solvent peak parameters (more variability)
SOLVENT_PEAKS = {
    "CDCl3": {"ppm": 7.26, "intensity_factor": 0.1, "width_factor": 1.0, "lorentz_factor": 1.2},
    "DMSO-d6": {"ppm": 2.50, "intensity_factor": 0.08, "width_factor": 1.2, "lorentz_factor": 1.5},
}
SOLVENT_PROBABILITY = 0.85  # Slightly reduced from 1.0; not every spectrum shows it

# Water Peak Parameters
WATER_PEAKS = {
    "CDCl3": {"ppm": 1.56, "width_factor": 2.0, "lorentz_factor": 2.5},
    "DMSO-d6": {"ppm": 3.33, "width_factor": 2.5, "lorentz_factor": 3.0},
}
# Water intensity is a wide random range, not a fixed factor.
WATER_INTENSITY_FACTOR_RANGE = (0.01, 1.0)
WATER_PROBABILITY = 0.65  # Increased probability, as water is very common

# 13C satellite parameters (more likely to appear in good S/N spectra)
C13_SATELLITE_INTENSITY_FACTOR = 0.0055
C13_COUPLING_1JCH_HZ_RANGE = (115.0, 160.0)
C13_SATELLITE_PROBABILITY = 0.6  # Increased probability

# Impurity peak parameters (more likely to have impurities)
IMPURITY_PROBABILITY = 0.40  # Increased probability
NUM_IMPURITY_PEAKS_MAX = 4
IMPURITY_INTENSITY_MAX_FRAC = 0.08
IMPURITY_WIDTH_FACTOR_RANGE = (0.8, 1.8)  # Wider range for varied impurity shapes

# Phase error parameters (allowing for more severe errors)
PHASE_ERROR_PROBABILITY = 0.6
MAX_ZERO_ORDER_PHASE_DEG = 25.0
MAX_FIRST_ORDER_PHASE_DEG_PER_PPM = 3.0

# Intensity below which a peak or line shape is treated as numerically zero.
_EPS = 1e-9


# --- Axis / geometry ---
@dataclass
class _Grid:
    """The discretised ppm axis shared by every peak-generating helper."""

    num_points: int
    ppm_min: float = PPM_MIN
    ppm_max: float = PPM_MAX
    axis: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.axis = np.arange(self.num_points, dtype=float)

    @property
    def points_per_ppm(self) -> float:
        return (self.num_points - 1) / (self.ppm_max - self.ppm_min)

    def ppm_to_idx(self, ppm_val: float) -> int:
        if self.num_points == 0:
            return 0
        if self.ppm_max == self.ppm_min:
            return 0 if ppm_val <= self.ppm_min else self.num_points - 1
        ppm_val = np.clip(ppm_val, self.ppm_min, self.ppm_max)
        return int(((ppm_val - self.ppm_min) / (self.ppm_max - self.ppm_min)) * (self.num_points - 1))

    def idx_to_ppm(self, idx: int) -> float:
        if self.num_points <= 1:
            return self.ppm_min
        return self.ppm_min + (idx / (self.num_points - 1)) * (self.ppm_max - self.ppm_min)


def _hz_to_ppm(hz_val: float, spectrometer_freq_hz: float) -> float:
    if spectrometer_freq_hz == 0:
        return 0.0  # Avoid division by zero
    return hz_val / (spectrometer_freq_hz / 1e6)


# --- Phase ---
@dataclass(frozen=True)
class _Phase:
    """A zero- plus first-order phase error, evaluated anywhere on the ppm axis."""

    phi0_rad: float = 0.0
    phi1_rad_per_ppm: float = 0.0
    ppm_pivot: float = (PPM_MAX + PPM_MIN) / 2.0

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "_Phase":
        """Draws a phase error, or a perfectly phased spectrum with probability 1 - PHASE_ERROR_PROBABILITY."""
        if rng.random() >= PHASE_ERROR_PROBABILITY:
            return cls()
        return cls(
            phi0_rad=np.deg2rad(rng.uniform(-MAX_ZERO_ORDER_PHASE_DEG, MAX_ZERO_ORDER_PHASE_DEG)),
            phi1_rad_per_ppm=np.deg2rad(rng.uniform(-MAX_FIRST_ORDER_PHASE_DEG_PER_PPM, MAX_FIRST_ORDER_PHASE_DEG_PER_PPM)),
        )

    def dispersive_fraction(self, ppm_value: float) -> float:
        """Fraction of dispersive line shape mixed into an absorption peak at `ppm_value`."""
        phase_rad = self.phi0_rad + self.phi1_rad_per_ppm * (ppm_value - self.ppm_pivot)
        return float(np.clip(np.tan(phase_rad), -5, 5))  # Cap effect


# --- Line shapes ---
def _lorentzian(grid: _Grid, center_idx: float, fwhm_points: float, amplitude: float) -> np.ndarray:
    gamma = max(fwhm_points, 0.1) / 2.0
    center_idx = np.clip(center_idx, 0, grid.num_points - 1)
    return amplitude * (gamma**2) / ((grid.axis - center_idx) ** 2 + gamma**2)


def _dispersive_lorentzian(grid: _Grid, center_idx: float, fwhm_points: float, amplitude: float) -> np.ndarray:
    gamma = max(fwhm_points, 0.1) / 2.0
    center_idx = np.clip(center_idx, 0, grid.num_points - 1)
    return amplitude * gamma * (grid.axis - center_idx) / ((grid.axis - center_idx) ** 2 + gamma**2)


def _gaussian(grid: _Grid, center_idx: float, sigma_points: float, amplitude: float) -> np.ndarray:
    sigma_points = max(sigma_points, 0.1)
    center_idx = np.clip(center_idx, 0, grid.num_points - 1)
    return amplitude * np.exp(-((grid.axis - center_idx) ** 2) / (2 * sigma_points**2))


def _delta(grid: _Grid, center_idx: float, amplitude: float) -> np.ndarray:
    """Fallback line shape for widths so small that the Voigt profile collapses."""
    peak = np.zeros(grid.num_points)
    if amplitude > _EPS and 0 <= round(center_idx) < grid.num_points:
        peak[round(center_idx)] = amplitude
    return peak


def _pseudo_voigt(
    grid: _Grid,
    center_idx: float,
    gaussian_sigma_points: float,
    lorentzian_fwhm_points: float,
    amplitude: float,
    eta: float,
    dispersive_fraction: float = 0.0,
) -> np.ndarray:
    """Linear (pseudo-Voigt) combination of a Lorentzian and a Gaussian, optionally phase-distorted.

    Args:
        eta: Lorentzian weight in [0, 1]; the Gaussian carries the remainder.
        dispersive_fraction: Amount of dispersive Lorentzian mixed in by a phase error.
    """
    absorption_lorentzian = _lorentzian(grid, center_idx, lorentzian_fwhm_points, amplitude)
    absorption_gaussian = _gaussian(grid, center_idx, gaussian_sigma_points, amplitude)
    unscaled = eta * absorption_lorentzian + (1 - eta) * absorption_gaussian

    peak_max = np.max(unscaled)
    if peak_max <= _EPS:
        # Widths were degenerate; fall back to a delta at the peak position.
        return _delta(grid, center_idx, amplitude)
    absorption = unscaled * (amplitude / peak_max)

    if abs(dispersive_fraction) <= 1e-6:
        return absorption

    dispersive = _dispersive_lorentzian(grid, center_idx, lorentzian_fwhm_points, amplitude)
    dispersive_max = np.max(np.abs(dispersive))
    if np.max(absorption_lorentzian) <= _EPS or dispersive_max <= _EPS:
        return absorption

    # Scale the dispersive part so its peak magnitude matches the absorptive Lorentzian's.
    dispersive *= np.max(absorption_lorentzian) / dispersive_max
    return absorption + eta * dispersive * dispersive_fraction


def _sample_widths(rng: np.random.Generator, width_factor: float = 1.0, lorentz_factor: float = 1.0) -> tuple[float, float]:
    """Draws (gaussian sigma, lorentzian FWHM) in ppm for a single line."""
    sigma_ppm = rng.uniform(PEAK_GAUSSIAN_SIGMA_PPM_MIN, PEAK_GAUSSIAN_SIGMA_PPM_MAX) * width_factor
    fwhm_ppm = rng.uniform(PEAK_LORENTZIAN_FWHM_PPM_MIN, PEAK_LORENTZIAN_FWHM_PPM_MAX) * width_factor * lorentz_factor
    return sigma_ppm, fwhm_ppm


def _mean_widths(width_factor: float = 1.0, lorentz_factor: float = 1.0) -> tuple[float, float]:
    """Deterministic mid-range widths, used for peaks whose shape is set by the solvent rather than sampled."""
    sigma_ppm = (PEAK_GAUSSIAN_SIGMA_PPM_MIN + PEAK_GAUSSIAN_SIGMA_PPM_MAX) / 2 * width_factor
    fwhm_ppm = (PEAK_LORENTZIAN_FWHM_PPM_MIN + PEAK_LORENTZIAN_FWHM_PPM_MAX) / 2 * width_factor * lorentz_factor
    return sigma_ppm, fwhm_ppm


def _peak_at_ppm(
    grid: _Grid,
    rng: np.random.Generator,
    phase: _Phase,
    ppm: float,
    amplitude: float,
    widths_ppm: tuple[float, float],
) -> np.ndarray:
    """Renders one line at `ppm`, phased according to `phase`. Returns zeros if it falls off the grid."""
    idx = grid.ppm_to_idx(ppm)
    if not 0 <= idx < grid.num_points:
        return np.zeros(grid.num_points)
    sigma_ppm, fwhm_ppm = widths_ppm
    return _pseudo_voigt(
        grid,
        idx,
        sigma_ppm * grid.points_per_ppm,
        fwhm_ppm * grid.points_per_ppm,
        amplitude,
        rng.uniform(VOIGT_ETA_MIN, VOIGT_ETA_MAX),
        phase.dispersive_fraction(ppm),
    )


# --- Peaks of the analyte ---
@dataclass
class _Peak:
    """A peak picked from the input spectrum, with its perturbed position and height."""

    original_idx: int
    original_ppm: float
    perturbed_ppm: float
    height: float
    is_complex_multiplet: bool = False


def _pick_peaks(h_nmr: np.ndarray, grid: _Grid, rng: np.random.Generator, max_intensity: float) -> list[_Peak]:
    """Detects the peaks of the ideal spectrum and perturbs their positions and heights."""
    min_prominence = FINDPKS_MIN_PEAK_PROMINENCE_FRAC * max_intensity
    if not (np.isfinite(min_prominence) and min_prominence > 0):
        min_prominence = 1e-5

    peak_indices, _ = find_peaks(
        h_nmr,
        height=max(0.0, FINDPKS_MIN_HEIGHT_FRAC * max_intensity),
        distance=max(1, int(FINDPKS_MIN_PEAK_DIST_PPM * grid.points_per_ppm)),
        prominence=min_prominence,
    )

    peaks = []
    for p_idx in peak_indices:
        original_ppm = grid.idx_to_ppm(p_idx)
        peaks.append(
            _Peak(
                original_idx=int(p_idx),
                original_ppm=original_ppm,
                perturbed_ppm=original_ppm + rng.uniform(-MAX_LOCAL_SHIFT_PPM, MAX_LOCAL_SHIFT_PPM),
                height=h_nmr[p_idx] * rng.uniform(1 - PEAK_INTENSITY_VARIATION, 1 + PEAK_INTENSITY_VARIATION),
            )
        )
    peaks.sort(key=lambda p: p.perturbed_ppm)
    return peaks


def _simulate_j_coupling(
    grid: _Grid,
    rng: np.random.Generator,
    phase: _Phase,
    center_ppm: float,
    total_intensity: float,
    spectrometer_freq_hz: float,
) -> np.ndarray:
    """Simulates a J-coupling multiplet from the n+1 rule.

    The appearance of the multiplet (in ppm) depends on the spectrometer frequency, and the
    dispersive fraction is evaluated once at the multiplet centre and shared by all its lines.
    """
    n_protons = int(rng.integers(1, MAX_COUPLING_PARTNERS, endpoint=True))
    j_ppm = _hz_to_ppm(rng.uniform(*J_COUPLING_RANGE_HZ), spectrometer_freq_hz)
    widths_ppm = _sample_widths(rng)

    # Relative line intensities follow Pascal's triangle.
    pascal_coeffs = [comb(n_protons, k, exact=True) for k in range(n_protons + 1)]
    total_coeff_sum = sum(pascal_coeffs)
    if total_coeff_sum == 0:
        return np.zeros(grid.num_points)

    multiplet = np.zeros(grid.num_points)
    center_dispersive_fraction = phase.dispersive_fraction(center_ppm)
    for k, coeff in enumerate(pascal_coeffs):
        line_ppm = center_ppm + (k - n_protons / 2.0) * j_ppm
        line_idx = grid.ppm_to_idx(line_ppm)
        if 0 <= line_idx < grid.num_points:
            multiplet += _pseudo_voigt(
                grid,
                line_idx,
                widths_ppm[0] * grid.points_per_ppm,
                widths_ppm[1] * grid.points_per_ppm,
                total_intensity * (coeff / total_coeff_sum),
                rng.uniform(VOIGT_ETA_MIN, VOIGT_ETA_MAX),
                center_dispersive_fraction,
            )
    return multiplet


def _reconstruct_peaks(
    peaks: list[_Peak], grid: _Grid, rng: np.random.Generator, phase: _Phase, spectrometer_freq_hz: float
) -> np.ndarray:
    """Re-synthesises every picked peak as either a single line or a J-coupled multiplet."""
    spectrum = np.zeros(grid.num_points)
    for peak in peaks:
        if rng.random() < COUPLING_PROBABILITY:
            spectrum += _simulate_j_coupling(grid, rng, phase, peak.perturbed_ppm, peak.height, spectrometer_freq_hz)
            peak.is_complex_multiplet = True
        else:
            spectrum += _peak_at_ppm(grid, rng, phase, peak.perturbed_ppm, peak.height, _sample_widths(rng))
    return spectrum


# --- Additional spectral features ---
def _add_c13_satellites(
    spectrum: np.ndarray,
    peaks: list[_Peak],
    grid: _Grid,
    rng: np.random.Generator,
    phase: _Phase,
    max_intensity: float,
    spectrometer_freq_hz: float,
) -> np.ndarray:
    if not peaks or rng.random() > C13_SATELLITE_PROBABILITY:
        return spectrum

    satellites = np.zeros(grid.num_points)
    satellite_offset_ppm = _hz_to_ppm(rng.uniform(*C13_COUPLING_1JCH_HZ_RANGE) / 2.0, spectrometer_freq_hz)

    for peak in peaks:
        if peak.height <= 0.05 * max_intensity or peak.is_complex_multiplet:
            continue
        intensity = peak.height * C13_SATELLITE_INTENSITY_FACTOR
        for side in (-1, 1):
            satellite_ppm = peak.perturbed_ppm + side * satellite_offset_ppm
            satellites += _peak_at_ppm(grid, rng, phase, satellite_ppm, intensity, _sample_widths(rng))
    return spectrum + satellites


def _add_solvent_and_water_peaks(
    spectrum: np.ndarray, grid: _Grid, rng: np.random.Generator, phase: _Phase, max_intensity: float
) -> np.ndarray:
    """Adds a residual solvent peak and, with some probability, the matching water peak."""
    if rng.random() > SOLVENT_PROBABILITY:
        return spectrum

    solvent_names = list(SOLVENT_PEAKS)
    solvent_name = solvent_names[rng.integers(len(solvent_names))]
    solvent_info = SOLVENT_PEAKS[solvent_name]

    intensity_scale = max_intensity if max_intensity > 0 else 1.0
    spectrum = spectrum + _peak_at_ppm(
        grid,
        rng,
        phase,
        solvent_info["ppm"] + rng.uniform(-0.02, 0.02),
        intensity_scale * solvent_info["intensity_factor"] * rng.uniform(0.7, 1.3),
        _mean_widths(solvent_info["width_factor"], solvent_info["lorentz_factor"]),
    )

    if rng.random() < WATER_PROBABILITY and solvent_name in WATER_PEAKS:
        water_info = WATER_PEAKS[solvent_name]
        spectrum = spectrum + _peak_at_ppm(
            grid,
            rng,
            phase,
            water_info["ppm"] + rng.uniform(-0.05, 0.05),  # Water peaks can shift more
            intensity_scale * rng.uniform(*WATER_INTENSITY_FACTOR_RANGE),
            _mean_widths(water_info["width_factor"], water_info["lorentz_factor"]),
        )
    return spectrum


def _add_impurity_peaks(
    spectrum: np.ndarray, grid: _Grid, rng: np.random.Generator, phase: _Phase, max_intensity: float
) -> np.ndarray:
    if rng.random() > IMPURITY_PROBABILITY or max_intensity == 0:
        return spectrum

    impurities = np.zeros(grid.num_points)
    for _ in range(int(rng.integers(1, NUM_IMPURITY_PEAKS_MAX, endpoint=True))):
        impurities += _peak_at_ppm(
            grid,
            rng,
            phase,
            rng.uniform(PPM_MIN, PPM_MAX),
            rng.uniform(0.001, IMPURITY_INTENSITY_MAX_FRAC) * max_intensity,
            _sample_widths(rng, width_factor=rng.uniform(*IMPURITY_WIDTH_FACTOR_RANGE)),
        )
    return spectrum + impurities


# --- Baseline artefacts ---
def _add_baseline_noise(spectrum: np.ndarray, rng: np.random.Generator, max_intensity: float) -> np.ndarray:
    """Adds baseline noise with a randomly selected factor from the configured range."""
    noise_factor = rng.uniform(*INTENSITY_NOISE_FACTOR_RANGE)
    noise_level = noise_factor * max_intensity if max_intensity > 0 else noise_factor
    return spectrum + rng.normal(0, noise_level, len(spectrum))


def _add_baseline_drift(
    spectrum: np.ndarray, rng: np.random.Generator, drift_amplitude: float, max_intensity: float
) -> np.ndarray:
    x = np.linspace(0, 1, len(spectrum))
    # Ensure drift calculation doesn't fail for very short spectra
    max_order = min(5, max(1, len(spectrum) - 1) if len(spectrum) > 1 else 1)
    poly_order = int(rng.integers(2, max(2, max_order), endpoint=True))
    drift = np.polynomial.polynomial.polyval(x, rng.normal(0, 1, poly_order))

    drift_span = np.max(drift) - np.min(drift)
    drift = (drift - np.min(drift)) / drift_span - 0.5 if drift_span > _EPS else np.zeros_like(drift)

    scale = max_intensity if max_intensity > 0 else 1.0
    return spectrum + drift * drift_amplitude * scale


def _apply_global_shift(spectrum: np.ndarray, grid: _Grid, rng: np.random.Generator) -> np.ndarray:
    """Rolls the whole spectrum by a small referencing error, zero-filling the wrapped edge."""
    shift_idx = round(rng.uniform(-MAX_GLOBAL_SHIFT_PPM, MAX_GLOBAL_SHIFT_PPM) * grid.points_per_ppm)
    if shift_idx == 0 or grid.num_points == 0:
        return spectrum
    spectrum = np.roll(spectrum, shift_idx)
    if shift_idx > 0:
        spectrum[:shift_idx] = 0
    else:
        spectrum[shift_idx:] = 0
    return spectrum


def _apply_global_broadening(spectrum: np.ndarray, grid: _Grid, rng: np.random.Generator) -> np.ndarray:
    sigma_ppm = rng.uniform(FINAL_GLOBAL_BROADENING_SIGMA_PPM_MIN, FINAL_GLOBAL_BROADENING_SIGMA_PPM_MAX)
    sigma_points = sigma_ppm * grid.points_per_ppm
    if sigma_points < 0.3:  # Sub-pixel broadening is a no-op
        return spectrum
    return gaussian_filter1d(spectrum, sigma=sigma_points, mode="reflect")


def augment(h_nmr: np.ndarray, spectrometer_freq_hz: float | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Augments an NMR spectrum with realistic effects, including variable
    spectrometer frequencies and diverse water peak intensities.

    Args:
        h_nmr (np.ndarray): The input ideal NMR spectrum, sampled on the [PPM_MIN, PPM_MAX] grid.
        spectrometer_freq_hz (float, optional): The spectrometer frequency in Hz.
            If None, a random frequency from COMMON_SPECTROMETER_FREQS_HZ is chosen.
        rng (np.random.Generator, optional): Source of randomness. If None, a freshly
            seeded generator is used, which keeps forked dataloader workers independent.

    Returns:
        np.ndarray: The augmented NMR spectrum, clipped to non-negative intensities.
    """
    if rng is None:
        rng = np.random.default_rng()
    if spectrometer_freq_hz is None:
        spectrometer_freq_hz = COMMON_SPECTROMETER_FREQS_HZ[rng.integers(len(COMMON_SPECTROMETER_FREQS_HZ))]

    num_points = len(h_nmr)
    max_intensity_input = np.max(h_nmr) if h_nmr.size > 0 else 0.0

    if num_points <= 1 or PPM_MAX <= PPM_MIN:
        # Degenerate axis: nothing to re-synthesise, so only add noise.
        noise_ref_intensity = max_intensity_input if max_intensity_input > 0 else 1.0
        return np.maximum(_add_baseline_noise(h_nmr.copy(), rng, noise_ref_intensity), 0)

    grid = _Grid(num_points)
    peaks = _pick_peaks(h_nmr, grid, rng, max_intensity_input)
    phase = _Phase.sample(rng) if peaks else _Phase()

    spectrum = _reconstruct_peaks(peaks, grid, rng, phase, spectrometer_freq_hz) if peaks else h_nmr.copy()

    max_signal_intensity = np.max(spectrum) if spectrum.size > 0 else 0.0
    max_intensity_for_additions = max_signal_intensity or max_intensity_input or 1e-5

    spectrum = _add_c13_satellites(spectrum, peaks, grid, rng, phase, max_intensity_for_additions, spectrometer_freq_hz)
    spectrum = _add_solvent_and_water_peaks(spectrum, grid, rng, phase, max_intensity_for_additions)
    spectrum = _add_impurity_peaks(spectrum, grid, rng, phase, max_intensity_for_additions)

    spectrum = _apply_global_shift(spectrum, grid, rng)
    spectrum = _apply_global_broadening(spectrum, grid, rng)

    scale_for_baseline = (np.max(spectrum) if spectrum.size > 0 else 0.0) or max_intensity_input or 1e-5
    spectrum = _add_baseline_noise(spectrum, rng, scale_for_baseline)
    spectrum = _add_baseline_drift(spectrum, rng, BASELINE_DRIFT_AMPLITUDE, scale_for_baseline)
    spectrum = np.maximum(spectrum, 0)

    if rng.random() < 0.2:
        spectrum = spectrum * rng.uniform(0.85, 1.15)
    return spectrum
