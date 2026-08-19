import random

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
# NEW: Water intensity is now a wide random range, not a fixed factor.
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


# --- Helper Functions ---
def _ppm_to_idx(ppm_val, num_points, ppm_min, ppm_max):
    if num_points == 0:
        return 0
    if ppm_max == ppm_min:
        return 0 if ppm_val <= ppm_min else num_points - 1
    ppm_val = np.clip(ppm_val, ppm_min, ppm_max)
    return int(((ppm_val - ppm_min) / (ppm_max - ppm_min)) * (num_points - 1))


def _idx_to_ppm(idx, num_points, ppm_min, ppm_max):
    if num_points <= 1:
        return ppm_min
    return ppm_min + (idx / (num_points - 1)) * (ppm_max - ppm_min)


def _hz_to_ppm(hz_val, spectrometer_freq_hz):
    if spectrometer_freq_hz == 0:
        return 0.0  # Avoid division by zero
    return hz_val / (spectrometer_freq_hz / 1e6)


def _generate_lorentzian(x_axis_len, center_idx, fwhm_points, amplitude):
    if fwhm_points < 0.1:
        fwhm_points = 0.1
    center_idx = np.clip(center_idx, 0, x_axis_len - 1)
    x = np.arange(x_axis_len)
    gamma = fwhm_points / 2.0
    return amplitude * (gamma**2) / ((x - center_idx) ** 2 + gamma**2)


def _generate_dispersive_lorentzian(x_axis_len, center_idx, fwhm_points, amplitude):
    if fwhm_points < 0.1:
        fwhm_points = 0.1
    center_idx = np.clip(center_idx, 0, x_axis_len - 1)
    x = np.arange(x_axis_len)
    gamma = fwhm_points / 2.0
    # Avoid division by zero if gamma is extremely small, though fwhm_points check helps
    if gamma < 1e-6:
        return np.zeros(x_axis_len)
    return amplitude * gamma * (x - center_idx) / ((x - center_idx) ** 2 + gamma**2)


def _generate_gaussian(x_axis_len, center_idx, sigma_points, amplitude):
    if sigma_points < 0.1:
        sigma_points = 0.1
    center_idx = np.clip(center_idx, 0, x_axis_len - 1)
    x = np.arange(x_axis_len)
    # Avoid division by zero if sigma_points is extremely small
    if sigma_points < 1e-6:
        peak = np.zeros(x_axis_len)
        if 0 <= round(center_idx) < x_axis_len:
            peak[round(center_idx)] = amplitude  # delta function
        return peak
    return amplitude * np.exp(-((x - center_idx) ** 2) / (2 * sigma_points**2))


def _generate_voigt_approx(
    x_axis_len, center_idx, gaussian_sigma_points, lorentzian_fwhm_points, amplitude, dispersive_fraction=0.0
):
    lorentzian_amp = amplitude
    gaussian_amp = amplitude

    abs_lorentzian = _generate_lorentzian(x_axis_len, center_idx, lorentzian_fwhm_points, lorentzian_amp)
    abs_gaussian = _generate_gaussian(x_axis_len, center_idx, gaussian_sigma_points, gaussian_amp)

    eta = random.uniform(VOIGT_ETA_MIN, VOIGT_ETA_MAX)  # Mixing parameter for Lorentzian contribution
    absorption_peak_unscaled = eta * abs_lorentzian + (1 - eta) * abs_gaussian

    max_abs_peak_unscaled = np.max(absorption_peak_unscaled)
    if max_abs_peak_unscaled > 1e-9:
        absorption_peak = absorption_peak_unscaled * (amplitude / max_abs_peak_unscaled)
    else:  # Handle case where unscaled peak is zero or tiny (e.g. extreme widths)
        absorption_peak = np.zeros_like(absorption_peak_unscaled)
        # If amplitude is non-zero, put a delta-like function at center_idx if widths were too small
        if amplitude > 1e-9 and 0 <= round(center_idx) < x_axis_len:
            absorption_peak[round(center_idx)] = amplitude

    if abs(dispersive_fraction) > 1e-6:
        disp_lorentzian_base = _generate_dispersive_lorentzian(x_axis_len, center_idx, lorentzian_fwhm_points, lorentzian_amp)

        if np.max(abs_lorentzian) > 1e-9 and np.max(np.abs(disp_lorentzian_base)) > 1e-9:
            # Scale disp_lorentzian_base so its peak magnitude matches abs_lorentzian's peak
            disp_lorentzian_scaled_to_abs = disp_lorentzian_base * (np.max(abs_lorentzian) / np.max(np.abs(disp_lorentzian_base)))
            dispersive_component = eta * disp_lorentzian_scaled_to_abs * dispersive_fraction
        else:
            dispersive_component = np.zeros_like(absorption_peak)

        final_peak = absorption_peak + dispersive_component
    else:
        final_peak = absorption_peak
    return final_peak


def _add_baseline_noise(spectrum, max_spectrum_intensity):
    """Adds baseline noise with a randomly selected factor from the configured range."""
    noise_factor = random.uniform(*INTENSITY_NOISE_FACTOR_RANGE)
    noise_level = noise_factor * max_spectrum_intensity if max_spectrum_intensity > 0 else noise_factor
    noise = np.random.normal(0, noise_level, len(spectrum))
    return spectrum + noise


def _add_baseline_drift(spectrum, drift_amplitude, max_spectrum_intensity):
    x = np.linspace(0, 1, len(spectrum))
    # Ensure drift calculation doesn't fail for very short spectra
    poly_order = random.randint(2, min(5, max(1, len(spectrum) - 1) if len(spectrum) > 1 else 1))
    coeffs = np.random.normal(0, 1, poly_order)
    drift = np.polynomial.polynomial.polyval(x, coeffs)

    if np.max(drift) - np.min(drift) > 1e-9:
        drift_scaled = (drift - np.min(drift)) / (np.max(drift) - np.min(drift)) - 0.5
    else:
        drift_scaled = np.zeros_like(drift)

    drift_final = (
        drift_scaled * drift_amplitude * max_spectrum_intensity if max_spectrum_intensity > 0 else drift_scaled * drift_amplitude
    )
    return spectrum + drift_final


def _calculate_dispersive_fraction_at_ppm(ppm_value, phase_params: dict):
    if not phase_params or "phi0_rad" not in phase_params:
        return 0.0

    phi0_rad = phase_params["phi0_rad"]
    phi1_rad_per_ppm = phase_params["phi1_rad_per_ppm"]
    ppm_pivot = phase_params["ppm_pivot"]

    phase_at_ppm_rad = phi0_rad + phi1_rad_per_ppm * (ppm_value - ppm_pivot)
    dispersive_fraction = np.tan(phase_at_ppm_rad)
    return np.clip(dispersive_fraction, -5, 5)  # Cap effect


def _add_c13_satellites(
    spectrum, peak_details, points_per_ppm, num_points, max_intensity_overall, spectrometer_freq_hz, phase_params: dict
):
    if not peak_details or random.random() > C13_SATELLITE_PROBABILITY:
        return spectrum

    satellite_spectrum_addition = np.zeros_like(spectrum)
    j_1ch_hz = random.uniform(*C13_COUPLING_1JCH_HZ_RANGE)
    satellite_offset_ppm = _hz_to_ppm(j_1ch_hz / 2.0, spectrometer_freq_hz)

    for peak in peak_details:
        if peak["height"] > 0.05 * max_intensity_overall and not peak.get("is_complex_multiplet", False):
            satellite_intensity = peak["height"] * C13_SATELLITE_INTENSITY_FACTOR

            for side in [-1, 1]:
                satellite_ppm = peak["perturbed_ppm"] + side * satellite_offset_ppm
                satellite_idx = _ppm_to_idx(satellite_ppm, num_points, PPM_MIN, PPM_MAX)

                if 0 <= satellite_idx < num_points:
                    gauss_sigma_ppm = random.uniform(PEAK_GAUSSIAN_SIGMA_PPM_MIN, PEAK_GAUSSIAN_SIGMA_PPM_MAX)
                    lorentz_fwhm_ppm = random.uniform(PEAK_LORENTZIAN_FWHM_PPM_MIN, PEAK_LORENTZIAN_FWHM_PPM_MAX)
                    gauss_sigma_points = gauss_sigma_ppm * points_per_ppm
                    lorentz_fwhm_points = lorentz_fwhm_ppm * points_per_ppm

                    satellite_dispersive_fraction = _calculate_dispersive_fraction_at_ppm(satellite_ppm, phase_params)

                    satellite_peak_shape = _generate_voigt_approx(
                        num_points,
                        satellite_idx,
                        gauss_sigma_points,
                        lorentz_fwhm_points,
                        satellite_intensity,
                        satellite_dispersive_fraction,
                    )
                    satellite_spectrum_addition += satellite_peak_shape
    return spectrum + satellite_spectrum_addition


def _add_solvent_and_water_peaks(spectrum, points_per_ppm, num_points, max_intensity_overall, phase_params: dict):
    """
    Adds a residual solvent peak and potentially a corresponding water peak.
    """
    if random.random() > SOLVENT_PROBABILITY:
        return spectrum

    # --- Add Residual Solvent Peak ---
    solvent_name = random.choice(list(SOLVENT_PEAKS.keys()))
    solvent_info = SOLVENT_PEAKS[solvent_name]

    solvent_ppm = solvent_info["ppm"] + random.uniform(-0.02, 0.02)
    solvent_idx = _ppm_to_idx(solvent_ppm, num_points, PPM_MIN, PPM_MAX)

    intensity_scale = max_intensity_overall if max_intensity_overall > 0 else 1.0
    solvent_intensity = intensity_scale * solvent_info["intensity_factor"] * random.uniform(0.7, 1.3)

    base_gauss_sigma = (PEAK_GAUSSIAN_SIGMA_PPM_MIN + PEAK_GAUSSIAN_SIGMA_PPM_MAX) / 2
    base_lorentz_fwhm = (PEAK_LORENTZIAN_FWHM_PPM_MIN + PEAK_LORENTZIAN_FWHM_PPM_MAX) / 2

    gauss_sigma_ppm = base_gauss_sigma * solvent_info["width_factor"]
    lorentz_fwhm_ppm = base_lorentz_fwhm * solvent_info["width_factor"] * solvent_info["lorentz_factor"]

    gauss_sigma_points = gauss_sigma_ppm * points_per_ppm
    lorentz_fwhm_points = lorentz_fwhm_ppm * points_per_ppm

    solvent_dispersive_fraction = _calculate_dispersive_fraction_at_ppm(solvent_ppm, phase_params)

    solvent_peak_shape = _generate_voigt_approx(
        num_points, solvent_idx, gauss_sigma_points, lorentz_fwhm_points, solvent_intensity, solvent_dispersive_fraction
    )
    spectrum += solvent_peak_shape

    # --- Add Corresponding Water Peak ---
    if random.random() < WATER_PROBABILITY and solvent_name in WATER_PEAKS:
        water_info = WATER_PEAKS[solvent_name]
        water_ppm = water_info["ppm"] + random.uniform(-0.05, 0.05)  # Water peaks can shift more
        water_idx = _ppm_to_idx(water_ppm, num_points, PPM_MIN, PPM_MAX)

        # MODIFIED: Use a wide random range for the water intensity factor for more realism.
        water_intensity_factor = random.uniform(*WATER_INTENSITY_FACTOR_RANGE)
        water_intensity = intensity_scale * water_intensity_factor

        gauss_sigma_ppm_water = base_gauss_sigma * water_info["width_factor"]
        lorentz_fwhm_ppm_water = base_lorentz_fwhm * water_info["width_factor"] * water_info["lorentz_factor"]

        gauss_sigma_points_water = gauss_sigma_ppm_water * points_per_ppm
        lorentz_fwhm_points_water = lorentz_fwhm_ppm_water * points_per_ppm

        water_dispersive_fraction = _calculate_dispersive_fraction_at_ppm(water_ppm, phase_params)

        water_peak_shape = _generate_voigt_approx(
            num_points,
            water_idx,
            gauss_sigma_points_water,
            lorentz_fwhm_points_water,
            water_intensity,
            water_dispersive_fraction,
        )
        spectrum += water_peak_shape

    return spectrum


def _add_impurity_peaks(spectrum, points_per_ppm, num_points, max_intensity_overall, phase_params: dict):
    if random.random() > IMPURITY_PROBABILITY or max_intensity_overall == 0:
        return spectrum

    num_impurities = random.randint(1, NUM_IMPURITY_PEAKS_MAX)
    impurity_spectrum_addition = np.zeros_like(spectrum)

    for _ in range(num_impurities):
        imp_ppm = random.uniform(PPM_MIN, PPM_MAX)
        imp_idx = _ppm_to_idx(imp_ppm, num_points, PPM_MIN, PPM_MAX)
        imp_intensity = random.uniform(0.001, IMPURITY_INTENSITY_MAX_FRAC) * max_intensity_overall

        width_factor = random.uniform(*IMPURITY_WIDTH_FACTOR_RANGE)
        gauss_sigma_ppm = random.uniform(PEAK_GAUSSIAN_SIGMA_PPM_MIN, PEAK_GAUSSIAN_SIGMA_PPM_MAX) * width_factor
        lorentz_fwhm_ppm = random.uniform(PEAK_LORENTZIAN_FWHM_PPM_MIN, PEAK_LORENTZIAN_FWHM_PPM_MAX) * width_factor

        gauss_sigma_points = gauss_sigma_ppm * points_per_ppm
        lorentz_fwhm_points = lorentz_fwhm_ppm * points_per_ppm

        imp_dispersive_fraction = _calculate_dispersive_fraction_at_ppm(imp_ppm, phase_params)

        impurity_peak_shape = _generate_voigt_approx(
            num_points, imp_idx, gauss_sigma_points, lorentz_fwhm_points, imp_intensity, imp_dispersive_fraction
        )
        impurity_spectrum_addition += impurity_peak_shape
    return spectrum + impurity_spectrum_addition


def _apply_phase_error_to_peak_details(peak_details, ppm_axis):
    dispersive_fraction_map_for_peaks = {}  # From original peak index to its specific dispersive fraction
    # Global phase parameters dict that defines the phase error across the spectrum
    phase_parameters = {"phi0_rad": 0.0, "phi1_rad_per_ppm": 0.0, "ppm_pivot": (PPM_MAX + PPM_MIN) / 2.0}

    if random.random() < PHASE_ERROR_PROBABILITY:
        phi0_deg = random.uniform(-MAX_ZERO_ORDER_PHASE_DEG, MAX_ZERO_ORDER_PHASE_DEG)
        phi1_deg_per_ppm = random.uniform(-MAX_FIRST_ORDER_PHASE_DEG_PER_PPM, MAX_FIRST_ORDER_PHASE_DEG_PER_PPM)

        phi0_rad = np.deg2rad(phi0_deg)
        phi1_rad_per_ppm = np.deg2rad(phi1_deg_per_ppm)
        ppm_pivot = (PPM_MAX + PPM_MIN) / 2.0

        phase_parameters.update({"phi0_rad": phi0_rad, "phi1_rad_per_ppm": phi1_rad_per_ppm, "ppm_pivot": ppm_pivot})

        for peak in peak_details:
            peak_ppm = peak["perturbed_ppm"]
            # Calculate dispersive fraction for this specific peak using the global phase parameters
            # This is stored mainly for direct use on original peaks before they become complex multiplets.
            # For new peaks (solvent, etc.) or multiplet lines, phase is recalculated at their specific ppm.
            dispersive_fraction = _calculate_dispersive_fraction_at_ppm(peak_ppm, phase_parameters)
            dispersive_fraction_map_for_peaks[peak["original_idx"]] = dispersive_fraction

    return dispersive_fraction_map_for_peaks, phase_parameters


def _simulate_j_coupling(
    center_ppm,
    total_intensity,
    j_hz,
    n_protons,
    points_per_ppm,
    num_points,
    spectrometer_freq_hz,
    dispersive_fraction,
    gauss_sigma_ppm,
    lorentz_fwhm_ppm,
):
    """
    NEW: Simulates a J-coupling multiplet based on the n+1 rule.
    The appearance of the multiplet (in ppm) depends on the spectrometer frequency.
    """
    multiplet_spectrum = np.zeros(num_points)
    num_lines = n_protons + 1
    j_ppm = _hz_to_ppm(j_hz, spectrometer_freq_hz)

    # Get relative intensities from Pascal's triangle
    pascal_coeffs = [comb(n_protons, k, exact=True) for k in range(num_lines)]
    total_coeff_sum = sum(pascal_coeffs)

    if total_coeff_sum == 0:
        return multiplet_spectrum  # Avoid division by zero

    # Calculate positions of each line in the multiplet
    line_positions_ppm = [center_ppm + (k - n_protons / 2.0) * j_ppm for k in range(num_lines)]

    gauss_sigma_points = gauss_sigma_ppm * points_per_ppm
    lorentz_fwhm_points = lorentz_fwhm_ppm * points_per_ppm

    for i, pos_ppm in enumerate(line_positions_ppm):
        line_intensity = total_intensity * (pascal_coeffs[i] / total_coeff_sum)
        line_idx = _ppm_to_idx(pos_ppm, num_points, PPM_MIN, PPM_MAX)

        if 0 <= line_idx < num_points:
            # The dispersive fraction is calculated once for the multiplet's center
            # and applied to all its lines, which is a reasonable approximation.
            line_shape = _generate_voigt_approx(
                num_points,
                line_idx,
                gauss_sigma_points,
                lorentz_fwhm_points,
                line_intensity,
                dispersive_fraction,
            )
            multiplet_spectrum += line_shape

    return multiplet_spectrum


def augment(h_nmr: np.array, spectrometer_freq_hz: float = None) -> np.array:
    """
    Augments an NMR spectrum with realistic effects, now including variable
    spectrometer frequencies and more diverse water peak intensities.

    Args:
        h_nmr (np.array): The input ideal NMR spectrum.
        spectrometer_freq_hz (float, optional): The spectrometer frequency in Hz.
            If None, a random frequency from COMMON_SPECTROMETER_FREQS_HZ is chosen.

    Returns:
        np.array: The augmented NMR spectrum.
    """
    # MODIFIED: Select a spectrometer frequency if not provided.
    if spectrometer_freq_hz is None:
        spectrometer_freq_hz = random.choice(COMMON_SPECTROMETER_FREQS_HZ)

    num_points = len(h_nmr)
    max_intensity_input = np.max(h_nmr) if h_nmr.size > 0 else 0.0

    if num_points > 1 and PPM_MAX > PPM_MIN:
        points_per_ppm = (num_points - 1) / (PPM_MAX - PPM_MIN)
        ppm_axis = np.linspace(PPM_MIN, PPM_MAX, num_points)
    else:
        current_spectrum = h_nmr.copy()
        if max_intensity_input > 0 or any(f > 0 for f in INTENSITY_NOISE_FACTOR_RANGE):
            noise_ref_intensity = max_intensity_input if max_intensity_input > 0 else 1.0
            current_spectrum = _add_baseline_noise(current_spectrum, noise_ref_intensity)
        return np.maximum(current_spectrum, 0)

    current_min_peak_prominence = FINDPKS_MIN_PEAK_PROMINENCE_FRAC * max_intensity_input
    if not (np.isfinite(current_min_peak_prominence) and current_min_peak_prominence > 0):
        current_min_peak_prominence = 1e-5
    min_peak_height_threshold = max(0.0, FINDPKS_MIN_HEIGHT_FRAC * max_intensity_input)
    min_peak_dist_points = max(1, int(FINDPKS_MIN_PEAK_DIST_PPM * points_per_ppm))

    peak_indices, _ = find_peaks(
        h_nmr,
        height=min_peak_height_threshold,
        distance=min_peak_dist_points,
        prominence=current_min_peak_prominence,
    )

    peak_details = []
    if len(peak_indices) > 0:
        for p_idx in peak_indices:
            intensity_variation = random.uniform(1 - PEAK_INTENSITY_VARIATION, 1 + PEAK_INTENSITY_VARIATION)
            original_ppm = _idx_to_ppm(p_idx, num_points, PPM_MIN, PPM_MAX)
            local_shift = random.uniform(-MAX_LOCAL_SHIFT_PPM, MAX_LOCAL_SHIFT_PPM)
            perturbed_ppm = original_ppm + local_shift
            peak_details.append(
                {
                    "original_idx": p_idx,
                    "original_ppm": original_ppm,
                    "perturbed_ppm": perturbed_ppm,
                    "height": h_nmr[p_idx] * intensity_variation,
                    "is_complex_multiplet": False,
                }
            )
        peak_details.sort(key=lambda p: p["perturbed_ppm"])

    dispersive_fraction_map_for_peaks, phase_parameters = {}, {}  # Default empty
    if peak_details:
        dispersive_fraction_map_for_peaks, phase_parameters = _apply_phase_error_to_peak_details(peak_details, ppm_axis)

    reconstructed_spectrum = np.zeros_like(h_nmr)
    if peak_details:
        for peak in peak_details:
            peak_dispersive_fraction = dispersive_fraction_map_for_peaks.get(peak["original_idx"], 0.0)

            if random.random() < COUPLING_PROBABILITY:
                j_value_hz = random.uniform(*J_COUPLING_RANGE_HZ)
                num_coupled_protons = random.randint(1, MAX_COUPLING_PARTNERS)

                multiplet_gauss_sigma_ppm = random.uniform(PEAK_GAUSSIAN_SIGMA_PPM_MIN, PEAK_GAUSSIAN_SIGMA_PPM_MAX)
                multiplet_lorentz_fwhm_ppm = random.uniform(PEAK_LORENTZIAN_FWHM_PPM_MIN, PEAK_LORENTZIAN_FWHM_PPM_MAX)

                current_peak_center_dispersive_fraction = _calculate_dispersive_fraction_at_ppm(
                    peak["perturbed_ppm"], phase_parameters
                )

                # MODIFIED: Call the new _simulate_j_coupling function, passing the chosen spectrometer frequency.
                coupled_spectrum_part = _simulate_j_coupling(
                    peak["perturbed_ppm"],
                    peak["height"],
                    j_value_hz,
                    num_coupled_protons,
                    points_per_ppm,
                    num_points,
                    spectrometer_freq_hz,
                    current_peak_center_dispersive_fraction,
                    multiplet_gauss_sigma_ppm,
                    multiplet_lorentz_fwhm_ppm,
                )
                reconstructed_spectrum += coupled_spectrum_part
                peak["is_complex_multiplet"] = True
            else:
                gauss_sigma_ppm = random.uniform(PEAK_GAUSSIAN_SIGMA_PPM_MIN, PEAK_GAUSSIAN_SIGMA_PPM_MAX)
                lorentz_fwhm_ppm = random.uniform(PEAK_LORENTZIAN_FWHM_PPM_MIN, PEAK_LORENTZIAN_FWHM_PPM_MAX)
                gauss_sigma_points = gauss_sigma_ppm * points_per_ppm
                lorentz_fwhm_points = lorentz_fwhm_ppm * points_per_ppm
                perturbed_idx = _ppm_to_idx(peak["perturbed_ppm"], num_points, PPM_MIN, PPM_MAX)

                voigt_peak = _generate_voigt_approx(
                    num_points, perturbed_idx, gauss_sigma_points, lorentz_fwhm_points, peak["height"], peak_dispersive_fraction
                )
                reconstructed_spectrum += voigt_peak
        current_spectrum = reconstructed_spectrum
    else:
        current_spectrum = h_nmr.copy()

    max_signal_intensity = np.max(current_spectrum) if current_spectrum.size > 0 else 0.0
    max_intensity_for_additions = (
        max_signal_intensity if max_signal_intensity > 0 else (max_intensity_input if max_intensity_input > 0 else 1e-5)
    )

    # MODIFIED: Pass the spectrometer frequency to functions that depend on it.
    current_spectrum = _add_c13_satellites(
        current_spectrum,
        peak_details,
        points_per_ppm,
        num_points,
        max_intensity_for_additions,
        spectrometer_freq_hz,
        phase_parameters,
    )
    current_spectrum = _add_solvent_and_water_peaks(
        current_spectrum, points_per_ppm, num_points, max_intensity_for_additions, phase_parameters
    )
    current_spectrum = _add_impurity_peaks(
        current_spectrum, points_per_ppm, num_points, max_intensity_for_additions, phase_parameters
    )

    global_shift_ppm = random.uniform(-MAX_GLOBAL_SHIFT_PPM, MAX_GLOBAL_SHIFT_PPM)
    shift_idx = round(global_shift_ppm * points_per_ppm)
    if shift_idx != 0 and num_points > 0:
        current_spectrum = np.roll(current_spectrum, shift_idx)
        if shift_idx > 0:
            current_spectrum[:shift_idx] = 0
        elif shift_idx < 0:
            current_spectrum[shift_idx:] = 0

    final_broadening_sigma_ppm = random.uniform(FINAL_GLOBAL_BROADENING_SIGMA_PPM_MIN, FINAL_GLOBAL_BROADENING_SIGMA_PPM_MAX)
    final_broadening_sigma_points = final_broadening_sigma_ppm * points_per_ppm
    if final_broadening_sigma_points >= 0.3:
        current_spectrum = gaussian_filter1d(current_spectrum, sigma=final_broadening_sigma_points, mode="reflect")

    current_max_after_all_mods = np.max(current_spectrum) if current_spectrum.size > 0 else 0.0
    scale_for_baseline = (
        current_max_after_all_mods
        if current_max_after_all_mods > 0
        else (max_intensity_input if max_intensity_input > 0 else 1e-5)
    )

    current_spectrum = _add_baseline_noise(current_spectrum, scale_for_baseline)
    current_spectrum = _add_baseline_drift(current_spectrum, BASELINE_DRIFT_AMPLITUDE, scale_for_baseline)
    current_spectrum = np.maximum(current_spectrum, 0)

    if random.random() < 0.2:
        current_spectrum *= random.uniform(0.85, 1.15)
    return current_spectrum
