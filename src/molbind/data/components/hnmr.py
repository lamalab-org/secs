import random

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# --- Configuration Constants ---
PPM_MIN = -2.0
PPM_MAX = 10.0
SPECTROMETER_FREQ_HZ = 400.0e6  # Spectrometer frequency in Hz (e.g., 400 MHz)

# Peak identification parameters
FINDPKS_MIN_PEAK_PROMINENCE_FRAC = 0.0002
FINDPKS_MIN_PEAK_DIST_PPM = 0.005
FINDPKS_MIN_HEIGHT_FRAC = 0.005

# Peak broadening parameters (Voigt components)
PEAK_GAUSSIAN_SIGMA_PPM_MIN = 0.001  # Gaussian component (inhomogeneity, etc.)
PEAK_GAUSSIAN_SIGMA_PPM_MAX = 0.003
PEAK_LORENTZIAN_FWHM_PPM_MIN = 0.001  # Lorentzian component (T2 relaxation)
PEAK_LORENTZIAN_FWHM_PPM_MAX = 0.008
VOIGT_ETA_MIN = 0.1  # Min Lorentzian contribution factor for pseudo-Voigt
VOIGT_ETA_MAX = 0.9  # Max Lorentzian contribution factor for pseudo-Voigt


# Global broadening parameters
FINAL_GLOBAL_BROADENING_SIGMA_PPM_MIN = 0.0003
FINAL_GLOBAL_BROADENING_SIGMA_PPM_MAX = 0.0015

# Chemical shift perturbation
MAX_GLOBAL_SHIFT_PPM = 0.10
MAX_LOCAL_SHIFT_PPM = 0.02

# Intensity variation parameters
INTENSITY_NOISE_FACTOR = 0.0001
PEAK_INTENSITY_VARIATION = 0.20
BASELINE_DRIFT_AMPLITUDE = 0.003

# Coupling simulation parameters
J_COUPLING_RANGE_HZ = (1.0, 18.0)  # Hz
COUPLING_PROBABILITY = 0.35
MAX_COUPLING_PARTNERS = 6  # Max n for (n+1) multiplicity (e.g., 6 for a septet)

# Solvent peak parameters
SOLVENT_PEAKS = {
    "CDCl3": {"ppm": 7.26, "intensity_factor": 0.1, "width_factor": 1.0, "lorentz_factor": 1.2},
    "DMSO-d6": {"ppm": 2.50, "intensity_factor": 0.08, "width_factor": 1.2, "lorentz_factor": 1.5},
    "D2O": {"ppm": 4.79, "intensity_factor": 0.12, "width_factor": 1.5, "lorentz_factor": 2.0},
    "CD3OD": {"ppm": 3.31, "intensity_factor": 0.09, "width_factor": 1.1, "lorentz_factor": 1.3},
    "TMS": {"ppm": 0.0, "intensity_factor": 0.05, "width_factor": 0.8, "lorentz_factor": 0.8},
}
SOLVENT_PROBABILITY = 0.6

# 13C satellite parameters
C13_SATELLITE_INTENSITY_FACTOR = 0.0055  # Per satellite (total ~1.1%)
C13_COUPLING_1JCH_HZ_RANGE = (115.0, 160.0)
C13_SATELLITE_PROBABILITY = 0.4

# Impurity peak parameters
IMPURITY_PROBABILITY = 0.25
NUM_IMPURITY_PEAKS_MAX = 3
IMPURITY_INTENSITY_MAX_FRAC = 0.05
IMPURITY_WIDTH_FACTOR_RANGE = (0.8, 1.5)

# Phase error parameters
PHASE_ERROR_PROBABILITY = 0.5
MAX_ZERO_ORDER_PHASE_DEG = 15.0
MAX_FIRST_ORDER_PHASE_DEG_PER_PPM = 2.0


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
        # The dispersive component is conventionally associated with the Lorentzian part.
        # Generate a dispersive Lorentzian with the same 'lorentzian_amp'.
        # Its effective amplitude relative to the absorptive Lorentzian needs care.
        # Max of L_abs is lorentzian_amp. Max of L_disp is lorentzian_amp/2.
        # We want D_final / A_final_L_part = dispersive_fraction (tan(phi))
        # A_final_L_part is approx eta * amplitude (if Gaussian part is small or similarly peaked)

        # Generate base dispersive Lorentzian, its amplitude is 'lorentzian_amp' in the formula,
        # but its actual peak value is lorentzian_amp / 2.
        disp_lorentzian_base = _generate_dispersive_lorentzian(x_axis_len, center_idx, lorentzian_fwhm_points, lorentzian_amp)

        # The absorptive Lorentzian part of 'absorption_peak' has an effective amplitude contributing 'eta * amplitude'
        # to the peak height 'amplitude' (ignoring Gaussian contribution for a moment for scaling D).
        # So, the amplitude of the LORENTZIAN PART of the scaled absorption_peak is roughly (eta * lorentzian_amp / max_abs_peak_unscaled) * amplitude
        # This is complex. Simpler: dispersive_fraction is ratio of D_max to L_max.
        # Peak of abs_lorentzian generated with lorentzian_amp is lorentzian_amp.
        # Peak of disp_lorentzian generated with lorentzian_amp is lorentzian_amp/2.
        # So, D_L_base needs to be scaled by 2 to match L_abs_base's amplitude scale.
        # Then, its contribution to Voigt is via eta.
        # Amplitude of dispersive component should be: (eta * amplitude_of_absorptive_Lorentzian) * dispersive_fraction

        # Estimate amplitude of the Lorentzian component in the final 'absorption_peak'
        # This is roughly eta * amplitude, assuming L and G peak at the same point and eta is for height contribution.
        effective_lorentzian_amplitude_in_absorption_peak = eta * amplitude

        # Scale the base dispersive Lorentzian. Max of disp_lorentzian_base is lorentzian_amp/2.
        # We want final_dispersive_peak = effective_lorentzian_amplitude_in_absorption_peak * dispersive_fraction
        # So, disp_lorentzian_base * scale_factor = effective_lorentzian_amplitude_in_absorption_peak * dispersive_fraction
        # scale_factor = (effective_lorentzian_amplitude_in_absorption_peak * dispersive_fraction) / (lorentzian_amp/2)
        #              = (eta * amplitude * dispersive_fraction) / (lorentzian_amp/2)
        # Since lorentzian_amp == amplitude initially:
        # scale_factor = (eta * dispersive_fraction) / (1/2) = 2 * eta * dispersive_fraction

        # However, if dispersive_fraction = tan(phi), and final signal is A*cos(phi) + D*sin(phi)
        # where A is (eta*L + (1-eta)*G) and D is (eta*L_disp).
        # If 'amplitude' is for A*cos(phi), then D*sin(phi) = D * cos(phi)*tan(phi)
        # = (eta*L_disp_scaled_like_L_abs) * cos(phi)*tan(phi)
        # This implies the current code is more correct than my re-derivation above if amplitude is post-phasing.
        # The original code's logic for scaling the dispersive component was:
        # disp_L_scaled_to_abs_L_peak = disp_L_base * (max(abs_L_base) / max(abs(disp_L_base)))
        #                           = disp_L_base * (lorentzian_amp / (lorentzian_amp/2)) = disp_L_base * 2
        # dispersive_component = eta * disp_L_scaled_to_abs_L_peak * dispersive_fraction
        #                      = eta * (disp_L_base * 2) * dispersive_fraction
        # This seems to make the dispersive amplitude proportional to `eta * max_abs_L_peak * dispersive_fraction`.
        # This is a sound approach for pseudo-Voigt phasing.

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


def _add_baseline_noise(spectrum, noise_factor, max_spectrum_intensity):
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


def _generate_pascal_triangle_intensities(num_equivalent_protons):
    n = num_equivalent_protons
    if n < 0:
        n = 0
    line = [1.0]  # Ensure float
    for _ in range(n):
        new_line = [1.0]
        for i in range(len(line) - 1):
            new_line.append(line[i] + line[i + 1])
        new_line.append(1.0)
        line = new_line
    current_sum = np.sum(line)
    return np.array(line) / current_sum if current_sum > 0 else np.array([1.0])


def _simulate_j_coupling(
    peak_ppm,
    peak_height,
    j_value_hz,
    num_equivalent_protons,
    points_per_ppm,
    num_points,
    spectrometer_freq_hz,
    dispersive_fraction,  # Common dispersive character for the multiplet
    multiplet_gauss_sigma_ppm,  # Common Gaussian width for the multiplet lines
    multiplet_lorentz_fwhm_ppm,  # Common Lorentzian width for the multiplet lines
):
    j_value_ppm = _hz_to_ppm(j_value_hz, spectrometer_freq_hz)
    intensities = _generate_pascal_triangle_intensities(num_equivalent_protons)
    multiplicity = len(intensities)
    split_spectrum = np.zeros(num_points)

    # Use the pre-determined linewidths for all lines in this multiplet
    gauss_sigma_points = multiplet_gauss_sigma_ppm * points_per_ppm
    lorentz_fwhm_points = multiplet_lorentz_fwhm_ppm * points_per_ppm

    for i, intensity_fraction in enumerate(intensities):
        offset_ppm = (i - (multiplicity - 1) / 2.0) * j_value_ppm
        split_ppm = peak_ppm + offset_ppm
        split_idx = _ppm_to_idx(split_ppm, num_points, PPM_MIN, PPM_MAX)

        peak_contribution = _generate_voigt_approx(
            num_points, split_idx, gauss_sigma_points, lorentz_fwhm_points, peak_height * intensity_fraction, dispersive_fraction
        )
        split_spectrum += peak_contribution
    return split_spectrum


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


def _add_solvent_peak(spectrum, points_per_ppm, num_points, max_intensity_overall, phase_params: dict):
    if random.random() > SOLVENT_PROBABILITY:
        return spectrum

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
    return spectrum + solvent_peak_shape


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


def augment(h_nmr: np.array) -> np.array:
    num_points = len(h_nmr)
    max_intensity_input = np.max(h_nmr) if h_nmr.size > 0 else 0.0

    if num_points > 1 and PPM_MAX > PPM_MIN:
        points_per_ppm = (num_points - 1) / (PPM_MAX - PPM_MIN)
        ppm_axis = np.linspace(PPM_MIN, PPM_MAX, num_points)
    else:
        current_spectrum = h_nmr.copy()
        if max_intensity_input > 0 or INTENSITY_NOISE_FACTOR > 0:  # Add noise even to zero input if noise factor is non-zero
            # Use a small default intensity for noise if input is all zero
            noise_ref_intensity = max_intensity_input if max_intensity_input > 0 else 1.0
            current_spectrum = _add_baseline_noise(current_spectrum, INTENSITY_NOISE_FACTOR, noise_ref_intensity)
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
            # Get the specific dispersive fraction if it was a simple peak affected by phase error
            # For complex multiplets, individual lines will have their phase calculated if phase_parameters is active
            peak_dispersive_fraction = dispersive_fraction_map_for_peaks.get(peak["original_idx"], 0.0)

            if random.random() < COUPLING_PROBABILITY:
                j_value_hz = random.uniform(*J_COUPLING_RANGE_HZ)
                num_coupled_protons = random.randint(1, MAX_COUPLING_PARTNERS)

                # Determine intrinsic linewidths for this multiplet ONCE
                multiplet_gauss_sigma_ppm = random.uniform(PEAK_GAUSSIAN_SIGMA_PPM_MIN, PEAK_GAUSSIAN_SIGMA_PPM_MAX)
                multiplet_lorentz_fwhm_ppm = random.uniform(PEAK_LORENTZIAN_FWHM_PPM_MIN, PEAK_LORENTZIAN_FWHM_PPM_MAX)

                # For multiplets, the phase of each line is calculated based on its specific PPM
                # So, pass phase_parameters and let _simulate_j_coupling handle it IF it were to calculate phase per line.
                # Current _simulate_j_coupling takes a single dispersive_fraction for the whole multiplet.
                # This means the multiplet as a whole has phase determined by its center.
                # Or, we use the global phase_parameters to determine dispersion at peak["perturbed_ppm"]
                # Let's use the globally determined phase character at the peak's center for the whole multiplet.
                current_peak_center_dispersive_fraction = _calculate_dispersive_fraction_at_ppm(
                    peak["perturbed_ppm"], phase_parameters
                )

                coupled_spectrum_part = _simulate_j_coupling(
                    peak["perturbed_ppm"],
                    peak["height"],
                    j_value_hz,
                    num_coupled_protons,
                    points_per_ppm,
                    num_points,
                    SPECTROMETER_FREQ_HZ,
                    current_peak_center_dispersive_fraction,  # Pass the calculated fraction for this multiplet's center
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

                # Use the pre-calculated dispersive fraction for this specific peak
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

    current_spectrum = _add_c13_satellites(
        current_spectrum,
        peak_details,
        points_per_ppm,
        num_points,
        max_intensity_for_additions,
        SPECTROMETER_FREQ_HZ,
        phase_parameters,
    )
    current_spectrum = _add_solvent_peak(
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

    current_spectrum = _add_baseline_noise(current_spectrum, INTENSITY_NOISE_FACTOR, scale_for_baseline)
    current_spectrum = _add_baseline_drift(current_spectrum, BASELINE_DRIFT_AMPLITUDE, scale_for_baseline)
    current_spectrum = np.maximum(current_spectrum, 0)

    if random.random() < 0.2:
        current_spectrum *= random.uniform(0.85, 1.15)
    return current_spectrum


def augment_with_global_t2_effect(h_nmr: np.array, t2_time_s_range=(0.01, 0.5)) -> np.array:
    # Note: This applies a *global* T2 effect often simulating very fast relaxation or severe broadening
    # ON TOP of any per-peak Lorentzian widths already applied by the main augment() function.
    # t2_time_s_range default is changed to represent more significant broadening.
    augmented_spectrum = augment(h_nmr)
    num_points = len(augmented_spectrum)
    if num_points <= 1 or PPM_MAX == PPM_MIN or SPECTROMETER_FREQ_HZ == 0:
        return augmented_spectrum

    points_per_ppm = (num_points - 1) / (PPM_MAX - PPM_MIN)
    t2_s = random.uniform(*t2_time_s_range)
    if t2_s <= 1e-6:
        return augmented_spectrum  # Avoid division by zero for extremely small T2

    line_broadening_hz = 1.0 / (np.pi * t2_s)
    line_broadening_ppm = _hz_to_ppm(line_broadening_hz, SPECTROMETER_FREQ_HZ)

    # Approximate Lorentzian FWHM broadening with a Gaussian filter.
    # Sigma for Gaussian approx of Lorentzian FWHM = FWHM / (2 * sqrt(2 * ln(2))) approx FWHM / 2.355
    # Using FWHM / 2 as a rough sigma for the Gaussian filter.
    broadening_sigma_points = (line_broadening_ppm * points_per_ppm) / 2.0

    if broadening_sigma_points >= 0.3:
        augmented_spectrum = gaussian_filter1d(augmented_spectrum, sigma=broadening_sigma_points, mode="reflect")
    return np.maximum(augmented_spectrum, 0)
