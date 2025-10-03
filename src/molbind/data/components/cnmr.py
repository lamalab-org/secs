import random

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# --- Simplified Configuration for 13C NMR ---
PPM_MIN = -5.0
PPM_MAX = 230.0

# --- Main Augmentation Function ---


def augment_13c(c_nmr: np.array) -> np.array:
    """
    Augments an ideal 13C NMR spectrum with basic realistic effects,
    including peak shifts and impurities.

    Args:
        c_nmr (np.array): The input ideal 13C NMR spectrum.

    Returns:
        np.array: The augmented 13C NMR spectrum.
    """
    num_points = len(c_nmr)
    if num_points == 0:
        return c_nmr

    # Helper to convert ppm to index
    def ppm_to_idx(ppm):
        return int(((ppm - PPM_MIN) / (PPM_MAX - PPM_MIN)) * (num_points - 1))

    # 1. Identify original peaks and apply shifts
    # Find peaks in the ideal spectrum
    peak_indices, _ = find_peaks(c_nmr, height=0.01 * np.max(c_nmr, initial=1))

    # Create a new empty spectrum to reconstruct shifted peaks
    reconstructed_spectrum = np.zeros_like(c_nmr)

    # Apply a random global shift to all peaks
    global_shift_ppm = random.uniform(-0.05, 0.05)

    for p_idx in peak_indices:
        original_ppm = PPM_MIN + (p_idx / (num_points - 1)) * (PPM_MAX - PPM_MIN)

        # Apply a small, random local shift to each peak
        local_shift_ppm = random.uniform(-0.02, 0.02)

        # Calculate the new position
        new_ppm = original_ppm + global_shift_ppm + local_shift_ppm
        new_idx = ppm_to_idx(new_ppm)

        if 0 <= new_idx < num_points:
            # Place the peak with its original height at the new position
            reconstructed_spectrum[new_idx] = c_nmr[p_idx]

    # 2. Broaden the reconstructed peaks
    # Apply a Gaussian filter for simple peak broadening
    broadening_sigma = random.uniform(0.5, 2.0)
    augmented_spectrum = gaussian_filter1d(reconstructed_spectrum, sigma=broadening_sigma)

    # Helper to add a new peak at a specific ppm
    def add_peak(spectrum, ppm, intensity, width):
        peak_position = ppm_to_idx(ppm)
        if 0 <= peak_position < num_points:
            # Generate a simple Gaussian peak
            x = np.arange(num_points)
            peak = intensity * np.exp(-((x - peak_position) ** 2) / (2 * width**2))
            return spectrum + peak
        return spectrum

    max_intensity = np.max(augmented_spectrum) if augmented_spectrum.size > 0 else 1.0

    # 3. Add a TMS peak (sharp singlet near 0 ppm)
    if random.random() < 0.85:  # High probability of TMS
        tms_intensity = max_intensity * random.uniform(0.2, 0.6)
        tms_ppm = random.uniform(-0.1, 0.1) + global_shift_ppm  # Also shift the TMS peak
        augmented_spectrum = add_peak(augmented_spectrum, tms_ppm, tms_intensity, random.uniform(0.5, 1.5))

    # 4. Add a solvent peak (e.g., CDCl3 triplet around 77 ppm)
    if random.random() < 0.90:  # High probability of solvent peak
        solvent_intensity = max_intensity * random.uniform(0.1, 0.4)
        solvent_width = random.uniform(1.0, 2.0)
        solvent_center_ppm = 77.16 + global_shift_ppm  # Also shift the solvent peak
        # Add a simplified triplet for CDCl3
        augmented_spectrum = add_peak(augmented_spectrum, solvent_center_ppm - 0.1, solvent_intensity / 2, solvent_width)
        augmented_spectrum = add_peak(augmented_spectrum, solvent_center_ppm, solvent_intensity, solvent_width)
        augmented_spectrum = add_peak(augmented_spectrum, solvent_center_ppm + 0.1, solvent_intensity / 2, solvent_width)

    # 5. Add impurity peaks
    if random.random() < 0.40:  # Probability of having impurities
        num_impurities = random.randint(1, 4)
        for _ in range(num_impurities):
            imp_ppm = random.uniform(PPM_MIN, PPM_MAX)
            imp_intensity = max_intensity * random.uniform(0.01, 0.15)
            imp_width = random.uniform(1.0, 3.0)
            augmented_spectrum = add_peak(augmented_spectrum, imp_ppm, imp_intensity, imp_width)

    # 6. Add baseline noise
    noise_level = max_intensity * random.uniform(0.01, 0.05)
    noise = np.random.normal(0, noise_level, num_points)
    augmented_spectrum += noise

    # 7. Add a simple baseline drift
    drift_amplitude = max_intensity * random.uniform(0.01, 0.05)
    baseline_drift = drift_amplitude * np.linspace(0, 1, num_points) ** 2
    augmented_spectrum += baseline_drift

    return np.maximum(augmented_spectrum, 0)
