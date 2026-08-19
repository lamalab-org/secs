import copy
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

HANDLERS = {
    ".csv": pd.read_csv,
    ".pickle": pd.read_pickle,
    ".pkl": pd.read_pickle,
    ".parquet": pd.read_parquet,
    "": lambda x: load_dataset(x).to_pandas(),
}


def rename_keys_with_prefix(d: dict, prefix: str = "model.") -> dict:
    new_dict = {}
    for key, value in d.items():
        if key.startswith(prefix):
            # remove the prefix
            new_key = key[len(prefix) :]
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    return new_dict


def select_device() -> str:
    """Selects the device to use for the model."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def find_all_pairs_in_list(lst: list[Any]) -> list[tuple[Any, Any]]:
    """Finds all pairs in a list."""
    return [(lst[i], lst[j]) for i in range(len(lst)) for j in range(i + 1, len(lst))]


def augment_peak_jitter(peak, h_ppm_stddev=0.01, c_ppm_stddev=0.1):
    augmented_peak = copy.deepcopy(peak)
    h_jitter = random.gauss(0, h_ppm_stddev)
    c_jitter = random.gauss(0, c_ppm_stddev)

    augmented_peak["1H_centroid"] += h_jitter
    augmented_peak["1H_min"] += h_jitter
    augmented_peak["1H_max"] += h_jitter

    augmented_peak["13C_centroid"] += c_jitter
    augmented_peak["13C_min"] += c_jitter
    augmented_peak["13C_max"] += c_jitter
    return augmented_peak


def augment_peak_width(peak, width_scale_min=0.8, width_scale_max=1.2):
    augmented_peak = copy.deepcopy(peak)

    h_width = augmented_peak["1H_max"] - augmented_peak["1H_min"]
    c_width = augmented_peak["13C_max"] - augmented_peak["13C_min"]

    h_scale = random.uniform(width_scale_min, width_scale_max)
    c_scale = random.uniform(width_scale_min, width_scale_max)

    new_h_width = max(h_width * h_scale, 0.001)  # Ensure non-zero positive width
    new_c_width = max(c_width * c_scale, 0.01)  # Ensure non-zero positive width

    augmented_peak["1H_min"] = augmented_peak["1H_centroid"] - new_h_width / 2
    augmented_peak["1H_max"] = augmented_peak["1H_centroid"] + new_h_width / 2

    augmented_peak["13C_min"] = augmented_peak["13C_centroid"] - new_c_width / 2
    augmented_peak["13C_max"] = augmented_peak["13C_centroid"] + new_c_width / 2

    # Ensure min < max (could happen if centroid is too close to edge of original small peak and width shrinks a lot)
    if augmented_peak["1H_min"] >= augmented_peak["1H_max"]:
        augmented_peak["1H_min"] = augmented_peak["1H_max"] - 0.001  # a tiny separation
    if augmented_peak["13C_min"] >= augmented_peak["13C_max"]:
        augmented_peak["13C_min"] = augmented_peak["13C_max"] - 0.01

    return augmented_peak


def augment_peak_intensity(peak, intensity_scale_min=0.7, intensity_scale_max=1.3):
    augmented_peak = copy.deepcopy(peak)
    scale = random.uniform(intensity_scale_min, intensity_scale_max)
    augmented_peak["nH"] = max(augmented_peak["nH"] * scale, 0.01)  # Ensure positive intensity
    return augmented_peak


def generate_hsqc_matrix(
    peaks, matrix_size=512, h_ppm_range=(-2.0, 10.0), c_ppm_range=(0.0, 300.0), sigma_factor=4.0, min_sigma_pixels=1.0
):
    """
    Generate a 2D HSQC matrix from a list of peak data.

    Parameters:
    -----------
    peaks : list of dict
        List of peak dictionaries, each containing:
        - '1H_centroid': 1H chemical shift centroid (ppm)
        - '13C_centroid': 13C chemical shift centroid (ppm)
        - '1H_min', '1H_max': 1H chemical shift range (ppm)
        - '13C_min', '13C_max': 13C chemical shift range (ppm)
        - 'nH': number of protons (intensity)

    matrix_size : int, optional
        Size of the output square matrix (default: 512)

    h_ppm_range : tuple, optional
        (min_ppm, max_ppm) range for 1H dimension (default: (-2.0, 10.0))

    c_ppm_range : tuple, optional
        (min_ppm, max_ppm) range for 13C dimension (default: (0.0, 300.0))

    sigma_factor : float, optional
        Factor to convert peak width to standard deviation (default: 4.0)
        Assumes (max-min) range covers approximately this many std devs

    min_sigma_pixels : float, optional
        Minimum standard deviation in pixel units (default: 1.0)

    Returns:
    --------
    numpy.ndarray
        512x512 (or matrix_size x matrix_size) array representing the HSQC spectrum
    """
    new_peaks = []
    for original_peak in peaks:
        # Apply a sequence of augmentations
        temp_peak = augment_peak_jitter(original_peak, h_ppm_stddev=0.02, c_ppm_stddev=0.2)
        temp_peak = augment_peak_width(temp_peak, width_scale_min=0.9, width_scale_max=1.1)
        augmented_peak = augment_peak_intensity(temp_peak, intensity_scale_min=0.8, intensity_scale_max=1.2)
        new_peaks.append(augmented_peak)
    peaks = new_peaks

    # Unpack ranges
    plot_min_1h_ppm, plot_max_1h_ppm = h_ppm_range
    plot_min_13c_ppm, plot_max_13c_ppm = c_ppm_range

    # Calculate PPM spans and per-pixel resolution
    h_ppm_span = plot_max_1h_ppm - plot_min_1h_ppm
    c_ppm_span = plot_max_13c_ppm - plot_min_13c_ppm

    h_ppm_per_pixel = h_ppm_span / (matrix_size - 1)
    c_ppm_per_pixel = c_ppm_span / (matrix_size - 1)

    # Initialize matrix
    hsqc_matrix = np.zeros((matrix_size, matrix_size), dtype=float)

    # Mapping functions (ppm to float pixel coordinate)
    def map_1h_to_col_float(ppm_val):
        """Maps 1H ppm to column index. Higher ppm = lower column index (left)."""
        col = (plot_max_1h_ppm - ppm_val) / h_ppm_span * (matrix_size - 1)
        return np.clip(col, 0, matrix_size - 1)

    def map_13c_to_row_float(ppm_val):
        """Maps 13C ppm to row index. Higher ppm = lower row index (top)."""
        row = (plot_max_13c_ppm - ppm_val) / c_ppm_span * (matrix_size - 1)
        return np.clip(row, 0, matrix_size - 1)

    # Create coordinate grids for Gaussian calculation
    y_coords, x_coords = np.ogrid[0:matrix_size, 0:matrix_size]

    # Process each peak
    for peak in peaks:
        h_centroid_ppm = peak["1H_centroid"]
        c_centroid_ppm = peak["13C_centroid"]
        intensity = float(peak["nH"])

        # Map centroid to float pixel coordinates
        center_col_float = map_1h_to_col_float(h_centroid_ppm)
        center_row_float = map_13c_to_row_float(c_centroid_ppm)

        # Estimate sigmas in PPM units from min/max
        h_width_ppm = peak["1H_max"] - peak["1H_min"]
        c_width_ppm = peak["13C_max"] - peak["13C_min"]

        # Calculate sigma in ppm, ensuring it's not too small
        sigma_h_ppm = h_width_ppm / sigma_factor
        if sigma_h_ppm < (h_ppm_per_pixel / 2.0) or sigma_h_ppm == 0:
            sigma_h_ppm = h_ppm_per_pixel * 0.75

        sigma_c_ppm = c_width_ppm / sigma_factor
        if sigma_c_ppm < (c_ppm_per_pixel / 2.0) or sigma_c_ppm == 0:
            sigma_c_ppm = c_ppm_per_pixel * 0.75

        # Convert sigmas from PPM to pixel units
        sigma_h_pixels = max(sigma_h_ppm / h_ppm_per_pixel, min_sigma_pixels)
        sigma_c_pixels = max(sigma_c_ppm / c_ppm_per_pixel, min_sigma_pixels)

        # Avoid division by zero
        if sigma_h_pixels <= 1e-6:
            sigma_h_pixels = 1e-6
        if sigma_c_pixels <= 1e-6:
            sigma_c_pixels = 1e-6

        # Calculate 2D Gaussian contribution
        exp_term_h = ((x_coords - center_col_float) ** 2) / (2 * sigma_h_pixels**2)
        exp_term_c = ((y_coords - center_row_float) ** 2) / (2 * sigma_c_pixels**2)
        gaussian_peak_contribution = intensity * np.exp(-(exp_term_h + exp_term_c))

        hsqc_matrix += gaussian_peak_contribution

    return hsqc_matrix
