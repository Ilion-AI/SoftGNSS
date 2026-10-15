"""
Data loader module for reading I/Q samples from binary files.

Supports chunked reading of large files (up to 30+ GB) without
loading the entire file into memory. Handles int8, int16, and
float32 interleaved I/Q formats.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from .config import Settings

logger = logging.getLogger("gnss_receiver.data_loader")


def load_iq_chunk(
    file_path: str,
    offset: int = 0,
    num_samples: int = 50000,
    dtype: np.dtype = np.dtype("<i2"),
) -> np.ndarray:
    """
    Load a chunk of I/Q data from a binary file.
    
    Parameters
    ----------
    file_path : str
        Path to the binary data file.
    offset : int
        Starting complex sample index (not byte offset).
    num_samples : int
        Number of complex I/Q samples to read.
    dtype : np.dtype
        Data type of individual samples (e.g., int8, int16, float32).
    
    Returns
    -------
    np.ndarray
        Complex I/Q samples as complex128 array of shape (num_samples,).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Byte offset = offset * 2 (I+Q) * bytes_per_sample
    byte_offset = offset * 2 * dtype.itemsize
    num_scalars = num_samples * 2  # I and Q for each sample
    
    # Read raw data
    raw = np.fromfile(str(file_path), dtype=dtype, count=num_scalars, offset=byte_offset)
    
    if len(raw) == 0:
        raise ValueError(
            f"No data read from file at offset {offset}. "
            f"File may be too short."
        )
    
    # Handle incomplete pairs
    if len(raw) % 2 != 0:
        logger.warning(
            f"Odd number of scalar samples ({len(raw)}). "
            f"Dropping last sample."
        )
        raw = raw[:-1]
    
    # Reshape to I/Q pairs and convert to complex
    iq = raw.reshape(-1, 2)
    signal = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
    
    return signal


def validate_chunk(data: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Validate an I/Q data chunk for NaN, Inf, and quality issues.
    
    Parameters
    ----------
    data : np.ndarray
        Complex I/Q samples.
    
    Returns
    -------
    tuple
        (cleaned_data, statistics_dict)
    """
    total = len(data)
    nan_mask = np.isnan(data)
    inf_mask = np.isinf(data)
    invalid_mask = nan_mask | inf_mask
    
    n_nan = np.sum(nan_mask)
    n_inf = np.sum(inf_mask)
    n_invalid = np.sum(invalid_mask)
    
    stats = {
        "total_samples": total,
        "nan_count": int(n_nan),
        "inf_count": int(n_inf),
        "invalid_count": int(n_invalid),
        "invalid_percent": 100.0 * n_invalid / total if total > 0 else 0.0,
        "mean_magnitude": float(np.abs(data[~invalid_mask]).mean()) if np.any(~invalid_mask) else 0.0,
        "std_magnitude": float(np.abs(data[~invalid_mask]).std()) if np.any(~invalid_mask) else 0.0,
    }
    
    if n_invalid > 0:
        if stats["invalid_percent"] > 10.0:
            logger.error(
                f"Data quality critical: {stats['invalid_percent']:.1f}% invalid samples "
                f"(>10% threshold). Data may be corrupted."
            )
        else:
            logger.warning(
                f"Found {n_invalid} invalid samples ({stats['invalid_percent']:.2f}%). "
                f"Replacing with zeros."
            )
        
        # Replace invalid values with zero
        data = data.copy()
        data[invalid_mask] = 0.0
    
    return data, stats


def load_signal_for_acquisition(
    settings: Settings,
    num_ms: int = 2,
    sample_offset: int = 0,
) -> np.ndarray:
    """
    Load enough I/Q data for acquisition (typically 2 ms).
    
    Parameters
    ----------
    settings : Settings
        Receiver settings.
    num_ms : int
        Number of milliseconds to load.
    sample_offset : int
        Starting sample offset.
    
    Returns
    -------
    np.ndarray
        Complex I/Q samples.
    """
    num_samples = int(settings.fs * num_ms * 1e-3)
    
    logger.info(
        f"Loading {num_ms} ms ({num_samples} samples) from "
        f"'{settings.file_name}' at offset {sample_offset}"
    )
    
    signal = load_iq_chunk(
        settings.file_name,
        offset=sample_offset,
        num_samples=num_samples,
        dtype=settings.dtype,
    )
    
    signal, stats = validate_chunk(signal)
    
    logger.info(
        f"Loaded {len(signal)} complex samples. "
        f"Mean |signal| = {stats['mean_magnitude']:.2f}, "
        f"Std |signal| = {stats['std_magnitude']:.2f}"
    )
    
    return signal


def get_file_info(settings: Settings) -> dict:
    """
    Get information about the data file.
    
    Returns dict with file size, number of samples, duration, etc.
    """
    file_path = Path(settings.file_name)
    
    if not file_path.exists():
        return {"error": f"File not found: {file_path}"}
    
    file_size = file_path.stat().st_size
    bytes_per_pair = settings.bytes_per_iq_pair
    total_samples = file_size // bytes_per_pair
    duration_s = total_samples / settings.fs
    duration_ms = duration_s * 1000
    
    return {
        "file_name": str(file_path),
        "file_size_bytes": file_size,
        "data_type": settings.data_type,
        "bytes_per_iq_pair": bytes_per_pair,
        "total_samples": total_samples,
        "duration_seconds": duration_s,
        "duration_ms": duration_ms,
        "sampling_freq_mhz": settings.fs / 1e6,
        "samples_per_code_period": settings.samples_per_code,
    }
