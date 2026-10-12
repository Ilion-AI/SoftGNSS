"""
GPS C/A PRN code generator.

Generates Gold codes for GPS satellites (PRN 1-32) using
the standard LFSR-based algorithm from IS-GPS-200.
"""

import logging
from typing import Optional

import numpy as np

from .config import CA_CODE_LENGTH, CA_CHIPPING_RATE, Settings

logger = logging.getLogger("gnss_receiver.prn_generator")

# G2 tap assignments for PRN 1-32 (0-indexed: PRN-1)
# These define the phase shift of the G2 code for each satellite
G2_TAPS = [
    5, 6, 7, 8, 17, 18, 139, 140, 141, 251,
    252, 254, 255, 256, 257, 258, 469, 470, 471, 472,
    473, 474, 509, 512, 513, 514, 515, 516, 859, 860,
    861, 862,
]


def generate_ca_code(prn: int) -> np.ndarray:
    """
    Generate a GPS C/A code for the given PRN number.
    
    Parameters
    ----------
    prn : int
        Satellite PRN number (1-32) or 0-based index (0-31).
    
    Returns
    -------
    np.ndarray
        C/A code sequence of length 1023 with values {-1, +1}.
    """
    # Strictly 0-based indexing (0..31).
    # Matches the original SoftGNSS generateCAcode() convention.
    if not (0 <= prn <= 31):
        raise ValueError(f"Invalid PRN index: {prn}. Must be 0-31.")
    idx = prn
    
    g2_shift = G2_TAPS[idx]
    
    # Generate G1 code (10-stage LFSR)
    # Polynomial: 1 + x^3 + x^10
    g1 = np.zeros(CA_CODE_LENGTH)
    reg = -1 * np.ones(10, dtype=np.float64)
    
    for i in range(CA_CODE_LENGTH):
        g1[i] = reg[9]  # Output from last stage
        feedback = reg[2] * reg[9]  # XOR (multiply for ±1 representation)
        reg[1:] = reg[:-1]
        reg[0] = feedback
    
    # Generate G2 code (10-stage LFSR)
    # Polynomial: 1 + x^2 + x^3 + x^6 + x^8 + x^9 + x^10
    g2 = np.zeros(CA_CODE_LENGTH)
    reg = -1 * np.ones(10, dtype=np.float64)
    
    for i in range(CA_CODE_LENGTH):
        g2[i] = reg[9]  # Output from last stage
        feedback = (reg[1] * reg[2] * reg[5] * reg[7] * reg[8] * reg[9])
        reg[1:] = reg[:-1]
        reg[0] = feedback
    
    # Apply G2 phase shift (delay)
    g2 = np.r_[g2[CA_CODE_LENGTH - g2_shift:], g2[:CA_CODE_LENGTH - g2_shift]]
    
    # Combine G1 and G2 to form C/A code
    ca_code = -g1 * g2
    
    return ca_code


def resample_code(
    ca_code: np.ndarray,
    fs: float,
    code_freq: float = CA_CHIPPING_RATE,
) -> np.ndarray:
    """
    Resample (upsample) a C/A code to match the sampling frequency.
    
    Parameters
    ----------
    ca_code : np.ndarray
        C/A code chip sequence (length 1023).
    fs : float
        Target sampling frequency in Hz.
    code_freq : float
        C/A code chipping rate (default: 1.023 MHz).
    
    Returns
    -------
    np.ndarray
        Resampled code with length = fs * 0.001 (samples per code period).
    """
    samples_per_code = int(np.round(fs / (code_freq / CA_CODE_LENGTH)))
    
    ts = 1.0 / fs
    tc = 1.0 / code_freq
    
    # Compute sample-to-chip mapping
    code_value_index = np.ceil(ts * np.arange(1, samples_per_code + 1) / tc).astype(np.int64) - 1
    code_value_index = np.clip(code_value_index, 0, CA_CODE_LENGTH - 1)
    code_value_index[-1] = CA_CODE_LENGTH - 1  # Ensure last sample maps to last chip
    
    return ca_code[code_value_index]


def make_ca_table(settings: Settings) -> np.ndarray:
    """
    Generate a table of resampled C/A codes for all 32 satellites.
    
    Parameters
    ----------
    settings : Settings
        Receiver settings (provides fs).
    
    Returns
    -------
    np.ndarray
        Array of shape (32, samples_per_code) containing
        resampled C/A codes for PRN 1-32.
    """
    samples_per_code = settings.samples_per_code
    ca_table = np.zeros((32, samples_per_code))
    
    for prn_idx in range(32):
        ca_code = generate_ca_code(prn_idx)
        ca_table[prn_idx] = resample_code(ca_code, settings.fs)
    
    logger.debug(
        f"Generated C/A code table: 32 PRNs x {samples_per_code} samples"
    )
    
    return ca_table


def make_ca_table_fft(settings: Settings) -> np.ndarray:
    """
    Generate a table of FFT-transformed C/A codes for all 32 satellites.
    
    The FFT of the codes is precomputed for efficient correlation
    in the acquisition algorithm (Parallel Code Phase Search).
    
    Parameters
    ----------
    settings : Settings
        Receiver settings.
    
    Returns
    -------
    np.ndarray
        Array of shape (32, samples_per_code) containing
        conjugate FFT of resampled C/A codes.
    """
    ca_table = make_ca_table(settings)
    # FFT and conjugate for correlation
    ca_table_fft = np.fft.fft(ca_table, axis=1).conj()
    
    logger.debug("Precomputed FFT of C/A code table for acquisition")
    
    return ca_table_fft
