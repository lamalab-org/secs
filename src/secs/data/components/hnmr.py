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
# Water intensity is drawn log-uniformly: most samples show a small water line,
# a wet one occasionally dominates. (A uniform draw made water the tallest feature
# in a third of all CDCl3 spectra.)
WATER_INTENSITY_FACTOR_RANGE = (0.005, 0.8)
WATER_PROBABILITY = 0.65  # Increased probability, as water is very common

# 13C satellite parameters (more likely to appear in good S/N spectra)
C13_SATELLITE_INTENSITY_FACTOR = 0.0055
C13_COUPLING_1JCH_HZ_RANGE = (115.0, 160.0)
C13_SATELLITE_PROBABILITY = 0.6  # Increased probability

# Impurity peak parameters (more likely to have impurities)
IMPURITY_PROBABILITY = 0.15  # Unknown junk only; known contaminants are drawn from H_IMPURITIES
NUM_IMPURITY_PEAKS_MAX = 4
IMPURITY_INTENSITY_MAX_FRAC = 0.08
IMPURITY_WIDTH_FACTOR_RANGE = (0.8, 1.8)  # Wider range for varied impurity shapes

# Phase error parameters (allowing for more severe errors)
PHASE_ERROR_PROBABILITY = 0.6
# Published SI spectra are phased by hand; residual errors are a few degrees, and
# the dispersive tails of a tall peak cluster become a visible baseline roll well
# before 20 degrees (checked visually on corpus renders).
MAX_ZERO_ORDER_PHASE_DEG = 12.0
MAX_FIRST_ORDER_PHASE_DEG_PER_PPM = 1.5

# Reference compound (TMS at 0 ppm)
TMS_PROBABILITY = 0.35
TMS_INTENSITY_RANGE = (0.01, 0.2)

# Known contaminants at their literature 1H shifts (Gottlieb 1997; Fulmer 2010),
# as (probability, {solvent: [(ppm, relative area), ...]}). A contaminant contributes
# all of its lines or none of them. The random-position impurity stage below stays,
# at reduced probability, for the junk no table covers.
H_IMPURITIES = {
    "grease": (0.35, {"CDCl3": [(0.86, 0.5), (1.26, 1.0)], "DMSO-d6": [(0.82, 0.5), (1.24, 1.0)]}),
    "ethyl_acetate": (
        0.12,
        {"CDCl3": [(1.26, 3.0), (2.05, 3.0), (4.12, 2.0)], "DMSO-d6": [(1.17, 3.0), (1.99, 3.0), (4.03, 2.0)]},
    ),
    "acetone": (0.12, {"CDCl3": [(2.17, 1.0)], "DMSO-d6": [(2.09, 1.0)]}),
    "dcm": (0.10, {"CDCl3": [(5.30, 1.0)], "DMSO-d6": [(5.76, 1.0)]}),
    "methanol": (0.08, {"CDCl3": [(3.49, 1.0)], "DMSO-d6": [(3.16, 1.0)]}),
    "ethanol": (0.06, {"CDCl3": [(1.25, 3.0), (3.72, 2.0)], "DMSO-d6": [(1.06, 3.0), (3.44, 2.0)]}),
    "diethyl_ether": (0.05, {"CDCl3": [(1.21, 6.0), (3.48, 4.0)], "DMSO-d6": [(1.09, 6.0), (3.38, 4.0)]}),
    "dmf": (0.05, {"CDCl3": [(2.88, 3.0), (2.96, 3.0), (8.02, 1.0)], "DMSO-d6": [(2.73, 3.0), (2.89, 3.0), (7.95, 1.0)]}),
    "dmso": (0.05, {"CDCl3": [(2.62, 1.0)], "DMSO-d6": [(2.54, 1.0)]}),
    "thf": (0.04, {"CDCl3": [(1.85, 4.0), (3.76, 4.0)], "DMSO-d6": [(1.76, 4.0), (3.60, 4.0)]}),
}
KNOWN_IMPURITY_SCALE_RANGE = (0.005, 0.10)

# Shimming / B0 inhomogeneity: an exponential tail on one side of every line
SHIM_ASYMMETRY_PROBABILITY = 0.35
SHIM_TAIL_PPM_RANGE = (0.003, 0.02)
SHIM_TAIL_FRACTION_RANGE = (0.05, 0.30)

# Spinning sidebands at +- the spinning rate around every peak
SPINNING_SIDEBAND_PROBABILITY = 0.2
SPINNING_RATE_HZ_RANGE = (12.0, 25.0)
SPINNING_SIDEBAND_FRACTION_RANGE = (0.002, 0.02)

# Truncation (sinc) wiggles from a too-short FID
TRUNCATION_PROBABILITY = 0.12
TRUNCATION_PERIOD_PPM_RANGE = (0.004, 0.015)
TRUNCATION_AMPLITUDE_RANGE = (0.03, 0.15)

# Solvent-suppression notch over the water region, with locally boosted noise.
# Only where people actually suppress water: samples in protic/hygroscopic
# solvents. Nobody runs presaturation on a CDCl3 sample.
SUPPRESSION_PROBABILITY = 0.15
SUPPRESSION_SOLVENTS = ("DMSO-d6",)
SUPPRESSION_DEPTH_RANGE = (0.4, 0.95)
SUPPRESSION_WIDTH_PPM_RANGE = (0.05, 0.3)
SUPPRESSION_NOISE_BOOST_RANGE = (2.0, 8.0)

# Noise texture: correlation length of the baseline noise (apodization smooths
# noise together with the signal), and scan-to-scan instability that converts
# signal into noise localized at the peaks.
NOISE_CORRELATION_SIGMA_POINTS = (0.0, 2.5)
SIGNAL_JITTER_PROBABILITY = 0.5
AMP_JITTER_RANGE = (0.002, 0.02)
FREQ_JITTER_POINTS_RANGE = (0.05, 0.6)

# Receiver-side baseline: slow sinusoidal ripple (clipped-FID signature) and DC offset
BASELINE_RIPPLE_PROBABILITY = 0.15
BASELINE_RIPPLE_AMPLITUDE_RANGE = (0.002, 0.02)
BASELINE_RIPPLE_PERIODS_RANGE = (1.0, 6.0)
BASELINE_DC_OFFSET_SIGMA = 0.003

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

    def ppm_to_pos(self, ppm_val: float) -> float:
        """Like ppm_to_idx but fractional, for line shapes placed between grid points."""
        if self.num_points <= 1 or self.ppm_max == self.ppm_min:
            return 0.0
        ppm_val = np.clip(ppm_val, self.ppm_min, self.ppm_max)
        return ((ppm_val - self.ppm_min) / (self.ppm_max - self.ppm_min)) * (self.num_points - 1)

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
        return float(np.clip(np.tan(phase_rad), -1.5, 1.5))  # Cap effect


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
    spectrum: np.ndarray,
    grid: _Grid,
    rng: np.random.Generator,
    phase: _Phase,
    max_intensity: float,
    solvent_name: str | None = None,
) -> np.ndarray:
    """Adds a residual solvent peak and, with some probability, the matching water peak.

    Args:
        solvent_name: Key of SOLVENT_PEAKS to use (resolved by `apply_instrument_artifacts`);
            if None or unknown, one is drawn at random.
    """
    if rng.random() > SOLVENT_PROBABILITY:
        return spectrum

    if solvent_name not in SOLVENT_PEAKS:
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
        lo, hi = WATER_INTENSITY_FACTOR_RANGE
        spectrum = spectrum + _peak_at_ppm(
            grid,
            rng,
            phase,
            water_info["ppm"] + rng.uniform(-0.05, 0.05),  # Water peaks can shift more
            intensity_scale * float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
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
            rng.uniform(grid.ppm_min, grid.ppm_max),
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


def _rolled(spectrum: np.ndarray, shift_idx: int) -> np.ndarray:
    """np.roll with the wrapped edge zero-filled instead of cycled."""
    out = np.roll(spectrum, shift_idx)
    if shift_idx > 0:
        out[:shift_idx] = 0
    elif shift_idx < 0:
        out[shift_idx:] = 0
    return out


def _add_reference_peak(
    spectrum: np.ndarray, grid: _Grid, rng: np.random.Generator, phase: _Phase, max_intensity: float
) -> np.ndarray:
    """TMS at 0 ppm; most literature spectra are referenced to it and many still show it."""
    if rng.random() > TMS_PROBABILITY:
        return spectrum
    intensity = rng.uniform(*TMS_INTENSITY_RANGE) * max_intensity
    return spectrum + _peak_at_ppm(grid, rng, phase, rng.uniform(-0.01, 0.01), intensity, _sample_widths(rng))


def _add_known_impurities(
    spectrum: np.ndarray, grid: _Grid, rng: np.random.Generator, phase: _Phase, max_intensity: float, solvent_name: str
) -> np.ndarray:
    """Common lab contaminants at their tabulated shifts for this solvent, all lines or none."""
    addition = np.zeros(grid.num_points)
    for prob, lines_by_solvent in H_IMPURITIES.values():
        lines = lines_by_solvent.get(solvent_name)
        if not lines or rng.random() > prob:
            continue
        scale = rng.uniform(*KNOWN_IMPURITY_SCALE_RANGE) * max_intensity
        largest_area = max(area for _, area in lines)
        for ppm, area in lines:
            addition += _peak_at_ppm(
                grid, rng, phase, ppm + rng.uniform(-0.02, 0.02), scale * area / largest_area, _sample_widths(rng)
            )
    return spectrum + addition


def _add_spinning_sidebands(
    spectrum: np.ndarray, grid: _Grid, rng: np.random.Generator, spectrometer_freq_hz: float
) -> np.ndarray:
    """Small mirror images of every peak at +- the spinning rate, from magnet/tube inhomogeneity."""
    if rng.random() > SPINNING_SIDEBAND_PROBABILITY:
        return spectrum
    offset_idx = round(_hz_to_ppm(rng.uniform(*SPINNING_RATE_HZ_RANGE), spectrometer_freq_hz) * grid.points_per_ppm)
    if offset_idx < 1:
        return spectrum
    up = rng.uniform(*SPINNING_SIDEBAND_FRACTION_RANGE)
    down = rng.uniform(*SPINNING_SIDEBAND_FRACTION_RANGE)  # the two sides need not match
    return spectrum + up * _rolled(spectrum, offset_idx) + down * _rolled(spectrum, -offset_idx)


def _apply_solvent_suppression(
    spectrum: np.ndarray, grid: _Grid, rng: np.random.Generator, solvent_name: str, max_intensity: float
) -> np.ndarray:
    """A suppression notch over the water region: attenuated signal, locally boosted noise."""
    if solvent_name not in SUPPRESSION_SOLVENTS or solvent_name not in WATER_PEAKS or rng.random() > SUPPRESSION_PROBABILITY:
        return spectrum
    center_idx = grid.ppm_to_pos(WATER_PEAKS[solvent_name]["ppm"])
    width_points = rng.uniform(*SUPPRESSION_WIDTH_PPM_RANGE) * grid.points_per_ppm
    profile = np.exp(-0.5 * ((grid.axis - center_idx) / max(width_points, 1.0)) ** 2)
    spectrum = spectrum * (1 - rng.uniform(*SUPPRESSION_DEPTH_RANGE) * profile)
    noise_level = np.mean(INTENSITY_NOISE_FACTOR_RANGE) * max_intensity * rng.uniform(*SUPPRESSION_NOISE_BOOST_RANGE)
    return spectrum + profile * rng.normal(0, noise_level, grid.num_points)


def _apply_shim_asymmetry(spectrum: np.ndarray, grid: _Grid, rng: np.random.Generator) -> np.ndarray:
    """Bad shimming: every line grows an exponential tail on one side of the axis."""
    if rng.random() > SHIM_ASYMMETRY_PROBABILITY:
        return spectrum
    tau = rng.uniform(*SHIM_TAIL_PPM_RANGE) * grid.points_per_ppm / 3.0
    half = int(6 * tau)
    if half < 1 or tau <= 0:
        return spectrum
    kernel = np.zeros(2 * half + 1)
    kernel[half] = 1.0
    tail = rng.uniform(*SHIM_TAIL_FRACTION_RANGE) * np.exp(-np.arange(1, half + 1) / tau)
    if rng.random() < 0.5:
        kernel[half + 1 :] = tail
    else:
        kernel[:half] = tail[::-1]
    return np.convolve(spectrum, kernel / kernel.sum(), mode="same")


def _apply_truncation_wiggles(spectrum: np.ndarray, grid: _Grid, rng: np.random.Generator) -> np.ndarray:
    """Sinc ringing around every line, from Fourier-transforming a truncated FID."""
    if rng.random() > TRUNCATION_PROBABILITY:
        return spectrum
    period = rng.uniform(*TRUNCATION_PERIOD_PPM_RANGE) * grid.points_per_ppm
    half = int(6 * period)
    if half < 2:
        return spectrum
    n = np.arange(-half, half + 1, dtype=float)
    kernel = rng.uniform(*TRUNCATION_AMPLITUDE_RANGE) * np.sinc(n / period) * np.exp(-np.abs(n) / (4 * period))
    kernel[half] = 1.0
    return np.convolve(spectrum, kernel / kernel.sum(), mode="same")


def _add_correlated_noise(spectrum: np.ndarray, rng: np.random.Generator, max_intensity: float) -> np.ndarray:
    """Baseline noise with a finite correlation length.

    Added BEFORE the final broadening and drawn smooth: apodization filters the
    noise together with the signal, so strictly white per-pixel noise is a
    synthetic fingerprint a model can key on.
    """
    noise_factor = rng.uniform(*INTENSITY_NOISE_FACTOR_RANGE)
    level = noise_factor * max_intensity if max_intensity > 0 else noise_factor
    noise = rng.normal(0, 1.0, len(spectrum))
    sigma = rng.uniform(*NOISE_CORRELATION_SIGMA_POINTS)
    if sigma >= 0.3:
        noise = gaussian_filter1d(noise, sigma)
    std = noise.std()
    return spectrum + noise * (level / std) if std > _EPS else spectrum


def _add_signal_jitter_noise(spectrum: np.ndarray, grid: _Grid, rng: np.random.Generator) -> np.ndarray:
    """Scan-to-scan instability: noise proportional to the signal, localized at the peaks.

    Amplitude jitter between co-added scans multiplies the signal by a slowly
    varying factor; frequency jitter adds derivative-shaped noise, the fuzz seen
    on the flanks of tall solvent lines (the 1D analogue of t1-noise).
    """
    if rng.random() > SIGNAL_JITTER_PROBABILITY:
        return spectrum
    slow = gaussian_filter1d(rng.normal(0, 1.0, grid.num_points), 5.0)
    std = slow.std()
    if std > _EPS:
        spectrum = spectrum * (1 + rng.uniform(*AMP_JITTER_RANGE) * slow / std)
    kappa = rng.uniform(*FREQ_JITTER_POINTS_RANGE)
    return spectrum + kappa * rng.normal(0, 1.0, grid.num_points) * np.gradient(spectrum)


def _add_baseline_ripple(spectrum: np.ndarray, rng: np.random.Generator, max_intensity: float) -> np.ndarray:
    """Receiver-side baseline: a small DC offset always, and the slow sinusoidal roll
    of a clipped FID with some probability."""
    spectrum = spectrum + rng.normal(0, BASELINE_DC_OFFSET_SIGMA) * max_intensity
    if rng.random() > BASELINE_RIPPLE_PROBABILITY:
        return spectrum
    x = np.linspace(0, 1, len(spectrum))
    amplitude = rng.uniform(*BASELINE_RIPPLE_AMPLITUDE_RANGE) * max_intensity
    periods = rng.uniform(*BASELINE_RIPPLE_PERIODS_RANGE)
    return spectrum + amplitude * np.sin(2 * np.pi * periods * x + rng.uniform(0, 2 * np.pi))


def apply_instrument_artifacts(
    spectrum: np.ndarray,
    grid: _Grid,
    rng: np.random.Generator,
    phase: _Phase,
    spectrometer_freq_hz: float,
    solvent_name: str | None = None,
    drift_amplitude: float = BASELINE_DRIFT_AMPLITUDE,
    clip_negative: bool = True,
) -> np.ndarray:
    """Everything the tube, magnet, and receiver do to a rendered spectrum.

    The shared back half of both augmentation paths (`augment` on rasterised
    spectra, `multiplets_to_spectrum` on reported multiplet lists): solvent and
    contaminant peaks, spinning sidebands, suppression, lineshape defects,
    referencing error, and the noise model. Every stage draws from `rng` only,
    so a seeded generator reproduces the spectrum bit for bit.

    Args:
        solvent_name: Key of SOLVENT_PEAKS; if None or unknown, one is drawn at random.
        drift_amplitude: Scale of the polynomial baseline roll.
        clip_negative: Clip the result at zero. Pass False to keep the negative
            baseline excursions and dispersive lobes a real spectrum has.
    """
    if solvent_name not in SOLVENT_PEAKS:
        solvent_names = list(SOLVENT_PEAKS)
        solvent_name = solvent_names[rng.integers(len(solvent_names))]

    max_intensity = (np.max(spectrum) if spectrum.size > 0 else 0.0) or 1e-5

    # Things in the tube besides the analyte
    spectrum = _add_solvent_and_water_peaks(spectrum, grid, rng, phase, max_intensity, solvent_name)
    spectrum = _add_reference_peak(spectrum, grid, rng, phase, max_intensity)
    spectrum = _add_known_impurities(spectrum, grid, rng, phase, max_intensity, solvent_name)
    spectrum = _add_impurity_peaks(spectrum, grid, rng, phase, max_intensity)

    # Instrument response, applied to everything at once
    spectrum = _add_spinning_sidebands(spectrum, grid, rng, spectrometer_freq_hz)
    spectrum = _apply_solvent_suppression(spectrum, grid, rng, solvent_name, max_intensity)
    spectrum = _apply_shim_asymmetry(spectrum, grid, rng)
    spectrum = _apply_truncation_wiggles(spectrum, grid, rng)
    spectrum = _apply_global_shift(spectrum, grid, rng)

    # Noise before the final broadening, so apodization smooths signal and noise together
    scale = (np.max(spectrum) if spectrum.size > 0 else 0.0) or max_intensity
    spectrum = _add_correlated_noise(spectrum, rng, scale)
    spectrum = _apply_global_broadening(spectrum, grid, rng)
    spectrum = _add_signal_jitter_noise(spectrum, grid, rng)

    spectrum = _add_baseline_drift(spectrum, rng, drift_amplitude, scale)
    spectrum = _add_baseline_ripple(spectrum, rng, scale)

    if clip_negative:
        spectrum = np.maximum(spectrum, 0)
    if rng.random() < 0.2:
        spectrum = spectrum * rng.uniform(0.85, 1.15)
    return spectrum


def augment(
    h_nmr: np.ndarray,
    spectrometer_freq_hz: float | None = None,
    rng: np.random.Generator | None = None,
    solvent: str | None = None,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Augments an NMR spectrum with realistic effects, including variable
    spectrometer frequencies and diverse water peak intensities.

    Args:
        h_nmr (np.ndarray): The input ideal NMR spectrum, sampled on the [PPM_MIN, PPM_MAX] grid.
        spectrometer_freq_hz (float, optional): The spectrometer frequency in Hz.
            If None, a random frequency from COMMON_SPECTROMETER_FREQS_HZ is chosen.
        rng (np.random.Generator, optional): Source of randomness. If None, a freshly
            seeded generator is used, which keeps forked dataloader workers independent.
        solvent (str, optional): Key of SOLVENT_PEAKS steering the residual solvent,
            water, and known-contaminant peaks; if None or unknown, one is drawn at random.
        clip_negative (bool): Clip the result at zero (the default). Pass False to keep
            the negative baseline excursions and dispersive lobes a real spectrum has.

    Returns:
        np.ndarray: The augmented NMR spectrum.
    """
    if rng is None:
        rng = np.random.default_rng()
    if spectrometer_freq_hz is None or spectrometer_freq_hz <= 0:
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
    return apply_instrument_artifacts(
        spectrum, grid, rng, phase, spectrometer_freq_hz, solvent_name=solvent, clip_negative=clip_negative
    )
