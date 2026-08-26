import ast
import re
from dataclasses import dataclass

import numpy as np
from scipy.special import comb

from secs.data.components.hnmr import (
    COMMON_SPECTROMETER_FREQS_HZ,
    PPM_MAX,
    PPM_MIN,
    VOIGT_ETA_MAX,
    VOIGT_ETA_MIN,
    _add_c13_satellites,
    _dispersive_lorentzian,
    _gaussian,
    _Grid,
    _hz_to_ppm,
    _lorentzian,
    _Peak,
    _Phase,
    apply_instrument_artifacts,
)

SPLIT_TOKENS = {
    "s": 0,
    "d": 1,
    "t": 2,
    "q": 3,
    "p": 4,
    "quint": 4,
    "pent": 4,
    "h": 6,
    "hex": 5,
    "sex": 5,
    "sext": 5,
    "hept": 6,
    "sept": 6,
    "spt": 6,
    "o": 7,
    "oct": 7,
    "non": 8,
    "nonet": 8,
}
_SORTED_TOKENS = sorted(SPLIT_TOKENS, key=len, reverse=True)

_BROAD_PREFIXES = ("broad", "br")
_NOISE_WORDS = ("apparent", "app", "obscured", "overlapping", "very")


J_DECILES_HZ = {
    1: [1.0, 1.8, 3.7, 6.0, 7.2, 7.9, 8.2, 8.6, 10.1, 14.0, 18.0],
    2: [1.5, 3.4, 6.1, 6.9, 7.2, 7.3, 7.5, 7.6, 7.8, 8.3, 10.0],
    3: [1.5, 3.6, 6.4, 6.8, 7.0, 7.1, 7.2, 7.2, 7.5, 8.0, 10.0],
}
DEFAULT_SPECTROMETER_FREQ_HZ = 400e6

LINE_WIDTH_HZ_RANGE = (0.8, 2.5)
BROAD_LINE_WIDTH_HZ_RANGE = (6.0, 30.0)
MIN_LINE_FWHM_POINTS = 2.0
UNRESOLVED_DEFAULT_SPAN_HZ_RANGE = (12.0, 40.0)
UNRESOLVED_MULTI_SIGNAL_MIN_HZ = 20.0
UNRESOLVED_SINGLE_COUPLING_MAX_HZ = 16.0
MIN_RESOLVED_RANGE_PPM = 0.004

SHIFT_JITTER_PPM = 0.004
J_JITTER_HZ = 0.3
ROOF_TILT_MAX = 0.35  # at 300 MHz
INTEGRAL_VARIATION = 0.03
INTEGRAL_ABSOLUTE_VARIATION_H = 0.1
MIN_RENDERED_AREA_H = 0.05
BASELINE_DRIFT_AMPLITUDE_REPORTED = 0.02

_SOLVENT_ALIASES = {
    "cdcl3": "CDCl3",
    "chloroform-d": "CDCl3",
    "dmso-d6": "DMSO-d6",
    "dmso": "DMSO-d6",
    "(cd3)2so": "DMSO-d6",
}


@dataclass
class Multiplet:
    multiplicity: str
    j_values_hz: list[float]
    n_protons: float
    shift_max: float
    shift_min: float
    is_broad: bool = False

    @property
    def center_ppm(self) -> float:
        return (self.shift_max + self.shift_min) / 2.0

    @property
    def range_ppm(self) -> float:
        return abs(self.shift_max - self.shift_min)

    @property
    def splitting(self) -> list[int] | None:
        return _parse_multiplicity(self.multiplicity)


def _parse_multiplicity(label: str) -> list[int] | None:
    text = label.strip().lower()
    for word in _NOISE_WORDS:
        text = text.replace(word, " ")
    text = re.sub(r"[\s.\-~]", "", text)
    for prefix in _BROAD_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break

    if not text or text == "m" or "'" in label or re.search(r"[AB]", label):
        return None

    splitting = []
    while text:
        for token in _SORTED_TOKENS:
            if text.startswith(token):
                splitting.append(SPLIT_TOKENS[token])
                text = text[len(token) :]
                break
        else:
            return None
    return splitting


def _parse_j_values(raw) -> list[float]:
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = [raw]
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]

    values = []
    for item in raw:
        match = re.search(r"-?\d+(?:\.\d+)?", str(item))
        if match and np.isfinite(value := abs(float(match.group()))):
            values.append(value)
    return values


def _parse_integration(raw) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", str(raw))
    return float(match.group()) if match else 1.0


def parse_multiplets(nmr_processed: str | list) -> list[Multiplet]:
    records = nmr_processed
    if isinstance(records, str):
        try:
            records = ast.literal_eval(records)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(records, (list, tuple)):
        return []

    multiplets = []
    for record in records:
        if not isinstance(record, (list, tuple)) or len(record) != 5:
            continue
        label, j_raw, integration, shift_a, shift_b = record
        if shift_a is None or shift_b is None:
            continue
        try:
            shift_a, shift_b = float(shift_a), float(shift_b)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(shift_a) and np.isfinite(shift_b)):
            continue
        label = str(label)
        multiplets.append(
            Multiplet(
                multiplicity=label,
                j_values_hz=_parse_j_values(j_raw),
                n_protons=_parse_integration(integration),
                shift_max=max(shift_a, shift_b),
                shift_min=min(shift_a, shift_b),
                is_broad=label.strip().lower().startswith(_BROAD_PREFIXES),
            )
        )
    return multiplets


_FREQ_SANITY_HZ = (40e6, 1.3e9)


def parse_spectrometer_freq_hz(text: str | float | None) -> float | None:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        freq = float(text)
        return freq if _FREQ_SANITY_HZ[0] <= freq <= _FREQ_SANITY_HZ[1] else None
    match = re.search(r"(\d+(?:\.\d+)?)\s*MHz", str(text), flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"\d+(?:\.\d+)?", str(text))
    if match is None:
        return None
    number = match.group(1) if match.lastindex else match.group()
    while number and "." not in number:
        if _FREQ_SANITY_HZ[0] <= float(number) * 1e6 <= _FREQ_SANITY_HZ[1]:
            break
        number = number[1:]
    freq = float(number) * 1e6 if number else None
    return freq if freq is not None and _FREQ_SANITY_HZ[0] <= freq <= _FREQ_SANITY_HZ[1] else None


def parse_solvent(text: str | None) -> str | None:
    if not text:
        return None
    return _SOLVENT_ALIASES.get(str(text).strip().lower())


def _draw_j_hz(n_partners: int, rng: np.random.Generator | None) -> float:
    deciles = J_DECILES_HZ[min(n_partners, max(J_DECILES_HZ))]
    if rng is None:
        return deciles[len(deciles) // 2]
    return float(np.interp(rng.random(), np.linspace(0.0, 1.0, len(deciles)), deciles))


def _multiplet_lines(
    multiplet: Multiplet, spectrometer_freq_hz: float, rng: np.random.Generator, jitter: bool
) -> list[tuple[float, float]]:
    area = multiplet.n_protons
    if jitter:
        area *= rng.uniform(1 - INTEGRAL_VARIATION, 1 + INTEGRAL_VARIATION)
        area += rng.uniform(-INTEGRAL_ABSOLUTE_VARIATION_H, INTEGRAL_ABSOLUTE_VARIATION_H)
        area = max(area, MIN_RENDERED_AREA_H)

    center = multiplet.center_ppm
    if jitter:
        center += rng.uniform(-SHIFT_JITTER_PPM, SHIFT_JITTER_PPM)

    splitting = multiplet.splitting
    if splitting is None:
        return _unresolved_lines(multiplet, center, area, rng, jitter, spectrometer_freq_hz)

    j_queue = list(multiplet.j_values_hz)
    lines = [(center, area)]
    for n_partners in [*splitting, *([1] * len(j_queue[len(splitting) :]))]:
        if n_partners == 0:
            continue
        j_hz = j_queue.pop(0) if j_queue else _draw_j_hz(n_partners, rng if jitter else None)
        if jitter:
            j_hz += rng.uniform(-J_JITTER_HZ, J_JITTER_HZ)
        j_hz = max(j_hz, 0.2)
        tilt = rng.uniform(-1, 1) * ROOF_TILT_MAX * (300e6 / spectrometer_freq_hz) if jitter else 0.0
        lines = _split_lines(lines, n_partners, _hz_to_ppm(j_hz, spectrometer_freq_hz), tilt)
    return lines


def _split_lines(lines: list[tuple[float, float]], n_partners: int, j_ppm: float, tilt: float = 0.0) -> list[tuple[float, float]]:
    weights = [float(comb(n_partners, k, exact=True)) for k in range(n_partners + 1)]
    if tilt:
        half = n_partners / 2.0
        weights = [w * max(0.05, 1 + tilt * (k - half) / max(half, 0.5)) for k, w in enumerate(weights)]
    total = sum(weights)
    split = []
    for ppm, area in lines:
        for k, weight in enumerate(weights):
            split.append((ppm + (k - n_partners / 2.0) * j_ppm, area * weight / total))
    return split


def _unresolved_lines(
    multiplet: Multiplet,
    center: float,
    area: float,
    rng: np.random.Generator,
    jitter: bool,
    spectrometer_freq_hz: float,
) -> list[tuple[float, float]]:
    if multiplet.is_broad:
        return [(center, area)]

    mhz = spectrometer_freq_hz / 1e6
    local = rng if jitter else np.random.default_rng(int(abs(center) * 1e4) * 7 + int(area * 10))

    span_ppm = multiplet.range_ppm
    if span_ppm < MIN_RESOLVED_RANGE_PPM:
        span_ppm = (local.uniform(*UNRESOLVED_DEFAULT_SPAN_HZ_RANGE) if jitter else 25.0) / mhz
    span_hz = span_ppm * mhz

    n_signals = 1
    if area > 2 and span_hz > UNRESOLVED_MULTI_SIGNAL_MIN_HZ:
        n_signals = int(local.integers(1, 3, endpoint=True))

    lines: list[tuple[float, float]] = []
    for _ in range(n_signals):
        k = 1 if span_hz / n_signals < UNRESOLVED_SINGLE_COUPLING_MAX_HZ else int(local.integers(2, 3, endpoint=True))
        partners = local.integers(1, 2, size=k, endpoint=True)
        ratios = np.sort(local.uniform(1.0, 3.0, size=k))[::-1]
        if n_signals == 1:
            sub_span_hz, sub_center = span_hz, center
        else:
            sub_span_hz = span_hz * local.uniform(0.3, 0.6)
            slack_ppm = span_ppm - sub_span_hz / mhz
            sub_center = center + local.uniform(-0.5, 0.5) * slack_ppm
        j_hz = ratios / float(np.sum(ratios * partners)) * sub_span_hz
        sub = [(sub_center, area / n_signals)]
        for n_partners, j in zip(partners.tolist(), j_hz.tolist(), strict=True):
            tilt = local.uniform(-1, 1) * ROOF_TILT_MAX * (300e6 / spectrometer_freq_hz) if jitter else 0.0
            sub = _split_lines(sub, n_partners, j / mhz, tilt)
        lines += sub
    return lines


def _render_lines(
    lines: list[tuple[float, float]],
    grid: _Grid,
    phase: _Phase,
    width_hz: float,
    spectrometer_freq_hz: float,
    eta: float,
) -> np.ndarray:
    fwhm_points = _hz_to_ppm(width_hz, spectrometer_freq_hz) * grid.points_per_ppm
    fwhm_points = max(fwhm_points, MIN_LINE_FWHM_POINTS)
    sigma_points = fwhm_points / 2.3548
    unit_area = eta * np.pi * fwhm_points / 2 + (1 - eta) * sigma_points * np.sqrt(2 * np.pi)
    ppm_per_point = 1.0 / grid.points_per_ppm

    spectrum = np.zeros(grid.num_points)
    for ppm, area in lines:
        if not grid.ppm_min <= ppm <= grid.ppm_max or area <= 0:
            continue
        height = area / (unit_area * ppm_per_point)
        pos = grid.ppm_to_pos(ppm)
        absorption = eta * _lorentzian(grid, pos, fwhm_points, height) + (1 - eta) * _gaussian(grid, pos, sigma_points, height)
        spectrum += absorption
        dispersive_fraction = phase.dispersive_fraction(ppm)
        if abs(dispersive_fraction) > 1e-6:
            spectrum += eta * dispersive_fraction * _dispersive_lorentzian(grid, pos, fwhm_points, height)
    return spectrum


def multiplets_to_spectrum(
    nmr_processed: str | list | list[Multiplet],
    num_points: int = 5000,
    spectrometer_freq_hz: float | None = None,
    solvent: str | None = None,
    ppm_range: tuple[float, float] = (PPM_MIN, PPM_MAX),
    augment: bool = True,
    normalize: bool = True,
    rng: np.random.Generator | None = None,
    clip_negative: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    if spectrometer_freq_hz is not None and spectrometer_freq_hz <= 0:
        spectrometer_freq_hz = None
    if spectrometer_freq_hz is None:
        spectrometer_freq_hz = (
            COMMON_SPECTROMETER_FREQS_HZ[rng.integers(len(COMMON_SPECTROMETER_FREQS_HZ))]
            if augment
            else DEFAULT_SPECTROMETER_FREQ_HZ
        )

    multiplets = nmr_processed
    if not (isinstance(multiplets, list) and all(isinstance(m, Multiplet) for m in multiplets)):
        multiplets = parse_multiplets(multiplets)

    ppm_min, ppm_max = ppm_range
    grid = _Grid(num_points, ppm_min=ppm_min, ppm_max=ppm_max)
    x = np.linspace(ppm_min, ppm_max, num_points)
    spectrum = np.zeros(num_points)

    phase = _Phase.sample(rng) if augment and multiplets else _Phase()
    peaks: list[_Peak] = []

    for multiplet in multiplets:
        lines = _multiplet_lines(multiplet, spectrometer_freq_hz, rng, jitter=augment)
        width_range = BROAD_LINE_WIDTH_HZ_RANGE if multiplet.is_broad else LINE_WIDTH_HZ_RANGE
        width_hz = rng.uniform(*width_range) if augment else float(np.mean(width_range))
        eta = rng.uniform(VOIGT_ETA_MIN, VOIGT_ETA_MAX) if augment else 0.5

        contribution = _render_lines(lines, grid, phase, width_hz, spectrometer_freq_hz, eta)
        spectrum += contribution

        center = multiplet.center_ppm
        if ppm_min <= center <= ppm_max:
            peaks.append(
                _Peak(
                    original_idx=grid.ppm_to_idx(center),
                    original_ppm=center,
                    perturbed_ppm=center,
                    height=float(np.max(contribution)),
                    is_complex_multiplet=len(lines) > 1,
                )
            )

    if augment:
        max_intensity = float(np.max(spectrum)) if spectrum.size else 0.0
        max_intensity = max_intensity or 1e-5
        spectrum = _add_c13_satellites(spectrum, peaks, grid, rng, phase, max_intensity, spectrometer_freq_hz)
        spectrum = apply_instrument_artifacts(
            spectrum,
            grid,
            rng,
            phase,
            spectrometer_freq_hz,
            solvent_name=solvent,
            drift_amplitude=BASELINE_DRIFT_AMPLITUDE_REPORTED,
            clip_negative=clip_negative,
        )

    if normalize and spectrum.size:
        lo, hi = float(np.min(spectrum)), float(np.max(spectrum))
        if hi > lo:
            spectrum = (spectrum - lo) / (hi - lo)
    return x, spectrum
