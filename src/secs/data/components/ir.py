import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.special import voigt_profile


def augment_md_to_real_ir(y, wavenumbers=None, seed=42):
    """
    Augments MD-simulated IR spectra to mimic real experimental FTIR/ATR-FTIR spectra,
    returning a 0-to-1 scaled spectrum.

    Args:
        y (np.array): 1D array of simulated Absorbance values.
        wavenumbers (np.array, optional): 1D array of wavenumbers (e.g., 400 to 4000).
        seed (int, optional): Random seed.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    x_norm = np.linspace(0, 1, n)

    A = np.copy(y)

    # ---------------------------------------------------------
    # 1. MD CORRECTIONS (Fixing LAMMPS/GAFF artifacts)
    # ---------------------------------------------------------

    # A. Non-linear X-Axis Warping (Anharmonicity correction)
    warp_magnitude = rng.uniform(0.005, 0.03)
    warp_field = gaussian_filter1d(rng.standard_normal(n), sigma=n / 10)
    warp_field = (warp_field / np.max(np.abs(warp_field))) * warp_magnitude
    new_x = np.clip(x_norm + warp_field, 0, 1)
    A = np.interp(x_norm, new_x, A)

    # B. Local Relative Intensity Variation
    intensity_mod = 1.0 + gaussian_filter1d(rng.standard_normal(n), sigma=n / 8) * rng.uniform(0.1, 0.4)
    A = A * np.clip(intensity_mod, 0.5, 2.0)

    # ---------------------------------------------------------
    # 2. INSTRUMENTAL EFFECTS (Broadening, ATR, Resolution)
    # ---------------------------------------------------------

    # C. Variable Voigt Broadening
    kx = np.arange(-15, 16)
    sigma = rng.uniform(0.5, 3.0)
    gamma = rng.uniform(0.5, 3.0)
    kernel = voigt_profile(kx, sigma, gamma)
    A = convolve1d(A, kernel / kernel.sum())

    # D. ATR-FTIR Penetration Depth Bias
    if rng.random() > 0.5:
        atr_tilt = np.linspace(rng.uniform(1.0, 1.5), rng.uniform(0.5, 1.0), n)
        A = A * atr_tilt

    # E. Global Scaling & Saturation (Detector limits)
    A = A * rng.uniform(0.5, 2.5)
    A = np.clip(A, a_min=0, a_max=rng.uniform(1.5, 3.0))

    # ---------------------------------------------------------
    # 3. ENVIRONMENTAL & BASELINE ARTIFACTS
    # ---------------------------------------------------------

    # F. Composite Baseline (Scattering + Wander)
    scatter = rng.uniform(0.01, 0.1) * np.exp(-rng.uniform(2, 6) * x_norm)
    wander = np.sin(x_norm * np.pi + rng.uniform(0, 2 * np.pi)) * rng.uniform(0.01, 0.1)

    # G. Atmospheric Interference (CO2 and H2O vapor)
    atm_noise = np.zeros(n)
    if wavenumbers is not None and rng.random() > 0.3:
        co2_mask = (wavenumbers > 2300) & (wavenumbers < 2400)
        h2o_mask = ((wavenumbers > 3500) & (wavenumbers < 3900)) | ((wavenumbers > 1300) & (wavenumbers < 1900))
        atm_noise[co2_mask] = rng.standard_normal(np.sum(co2_mask)) * rng.uniform(0.01, 0.05)
        atm_noise[h2o_mask] = rng.standard_normal(np.sum(h2o_mask)) * rng.uniform(0.005, 0.02)
    else:
        fringe_freq = rng.uniform(50, 200)
        atm_noise = np.sin(x_norm * fringe_freq) * rng.uniform(0.0, 0.01)

    # H. Heteroscedastic (Signal-dependent) + Pink Noise
    shot_noise = rng.standard_normal(n) * np.sqrt(np.abs(A) + 0.01) * rng.uniform(0.001, 0.01)
    read_noise = rng.standard_normal(n) * rng.uniform(0.002, 0.01)

    # Combine everything
    A_final = A + scatter + wander + atm_noise + shot_noise + read_noise

    # ---------------------------------------------------------
    # 4. MIN-MAX SCALING (0 to 1)
    # ---------------------------------------------------------
    a_min = np.min(A_final)
    a_max = np.max(A_final)

    if a_max > a_min:
        A_final = (A_final - a_min) / (a_max - a_min)
    else:
        # Fallback in case of a completely flat spectrum to avoid division by zero
        A_final = np.zeros_like(A_final)

    return A_final.astype(np.float32)
