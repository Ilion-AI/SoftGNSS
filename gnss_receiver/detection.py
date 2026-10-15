"""
Detection module for adaptive threshold and signal presence decision.

Implements SNR estimation from correlation function and
signal detection criteria for GNSS acquisition.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from .config import Settings, CA_CHIPPING_RATE

logger = logging.getLogger("gnss_receiver.detection")


@dataclass
class DetectionResult:
    """Result of signal detection for one PRN."""
    detected: bool = False
    snr: float = 0.0
    doppler_hz: float = 0.0
    code_phase_samples: int = 0
    code_phase_chips: float = 0.0
    code_phase_meters: float = 0.0
    peak_value: float = 0.0
    noise_mean: float = 0.0
    noise_std: float = 0.0
    peak_to_second_ratio: float = 0.0
    flag: str = "ok"  # "ok" or "weak"


def compute_snr(
    correlation: np.ndarray,
    peak_index: int,
    fs: float,
    exclude_chips: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Compute SNR from correlation function using adaptive method.
    
    The SNR is computed as: SNR = (peak - mean_noise) / std_noise
    
    The noise floor is estimated from the correlation values
    excluding a region around the peak (±exclude_chips chips).
    
    Parameters
    ----------
    correlation : np.ndarray
        1D array of correlation magnitudes (|C(tau)|).
    peak_index : int
        Index of the correlation peak.
    fs : float
        Sampling frequency.
    exclude_chips : float
        Number of chips to exclude around peak for noise estimation.
    
    Returns
    -------
    tuple
        (snr, noise_mean, noise_std)
    """
    peak_value = correlation[peak_index]
    
    # Compute exclusion range in samples
    samples_per_chip = fs / CA_CHIPPING_RATE
    exclude_samples = int(np.ceil(exclude_chips * samples_per_chip))
    
    # Create mask excluding region around peak
    n = len(correlation)
    mask = np.ones(n, dtype=bool)
    
    start = max(0, peak_index - exclude_samples)
    end = min(n, peak_index + exclude_samples + 1)
    mask[start:end] = False
    
    # Compute noise statistics
    noise_values = correlation[mask]
    
    if len(noise_values) == 0:
        return 0.0, 0.0, 0.0
    
    noise_mean = np.mean(noise_values)
    noise_std = np.std(noise_values)
    
    # SNR
    if noise_std > 0:
        snr = (peak_value - noise_mean) / noise_std
    else:
        snr = 0.0
    
    return snr, noise_mean, noise_std


def compute_peak_ratio(
    correlation: np.ndarray,
    peak_index: int,
    fs: float,
) -> float:
    """
    Compute the ratio of the first peak to the second highest peak.
    
    The second peak is chosen at least 1 chip away from the first peak.
    If the ratio is < 1.5, the detection is considered unreliable.
    
    Parameters
    ----------
    correlation : np.ndarray
        1D array of correlation magnitudes.
    peak_index : int
        Index of the first (highest) peak.
    fs : float
        Sampling frequency.
    
    Returns
    -------
    float
        Peak-to-second-peak ratio.
    """
    peak_value = correlation[peak_index]
    n = len(correlation)
    
    # Exclude ±1 chip around peak
    samples_per_chip = fs / CA_CHIPPING_RATE
    exclude = int(np.ceil(samples_per_chip))
    
    mask = np.ones(n, dtype=bool)
    start = max(0, peak_index - exclude)
    end = min(n, peak_index + exclude + 1)
    mask[start:end] = False
    
    if not np.any(mask):
        return float("inf")
    
    second_peak = np.max(correlation[mask])
    
    if second_peak > 0:
        return peak_value / second_peak
    else:
        return float("inf")


def detect_signal(
    correlation: np.ndarray,
    doppler_bin: int,
    doppler_range: float,
    doppler_step: float,
    fs: float,
    threshold: float = 2.5,
    c: float = 299_792_458.0,
) -> DetectionResult:
    """
    Detect the presence of a GPS signal in the correlation result.
    
    Parameters
    ----------
    correlation : np.ndarray
        2D array of shape (n_doppler, n_code_phases) with correlation magnitudes.
    doppler_bin : int
        Index of the Doppler bin to analyze.
    doppler_range : float
        Doppler search range in Hz.
    doppler_step : float
        Doppler search step in Hz.
    fs : float
        Sampling frequency.
    threshold : float
        SNR threshold for detection.
    c : float
        Speed of light in m/s.
    
    Returns
    -------
    DetectionResult
        Detection result with SNR, code phase, Doppler, etc.
    """
    result = DetectionResult()
    
    # Find the peak in the correlation for this Doppler bin
    corr_1d = correlation[doppler_bin, :]
    peak_index = np.argmax(corr_1d)
    peak_value = corr_1d[peak_index]
    
    # Compute SNR
    snr, noise_mean, noise_std = compute_snr(corr_1d, peak_index, fs)
    
    # Compute peak-to-second ratio
    peak_ratio = compute_peak_ratio(corr_1d, peak_index, fs)
    
    # Compute Doppler frequency
    doppler_hz = -doppler_range + doppler_step * doppler_bin
    
    # Compute code phase in various units
    code_phase_samples = peak_index
    code_phase_chips = code_phase_samples * CA_CHIPPING_RATE / fs
    code_phase_meters = code_phase_chips * (c / CA_CHIPPING_RATE)
    
    # Determine detection
    detected = snr >= threshold
    
    # Flag quality
    flag = "ok"
    if peak_ratio < 1.5:
        flag = "weak"
    
    result.detected = detected
    result.snr = snr
    result.doppler_hz = doppler_hz
    result.code_phase_samples = code_phase_samples
    result.code_phase_chips = code_phase_chips
    result.code_phase_meters = code_phase_meters
    result.peak_value = peak_value
    result.noise_mean = noise_mean
    result.noise_std = noise_std
    result.peak_to_second_ratio = peak_ratio
    result.flag = flag
    
    return result


def detect_signal_full(
    correlation_2d: np.ndarray,
    fs: float,
    doppler_range: float,
    doppler_step: float,
    threshold: float = 2.5,
) -> DetectionResult:
    """
    Detect the strongest signal across ALL Doppler bins for one PRN.
    
    This is the recommended detection method: find the global maximum
    of the 2D correlation surface, then evaluate SNR at that point.
    
    Parameters
    ----------
    correlation_2d : np.ndarray
        2D correlation surface (n_doppler x n_code_phases).
    fs : float
        Sampling frequency.
    doppler_range : float
        Doppler search range.
    doppler_step : float
        Doppler step.
    threshold : float
        SNR threshold.
    
    Returns
    -------
    DetectionResult
        Detection result.
    """
    # Find global peak
    peak_idx = np.unravel_index(np.argmax(correlation_2d), correlation_2d.shape)
    doppler_bin = peak_idx[0]
    
    return detect_signal(
        correlation_2d,
        doppler_bin,
        doppler_range,
        doppler_step,
        fs,
        threshold,
    )
