"""
GPS L1 C/A signal acquisition using Parallel Code Phase Search (PCPS).

The PCPS algorithm uses FFT-based correlation to efficiently search
all code phases simultaneously for each trial Doppler frequency.
This reduces the complexity from O(N²) to O(N·log(N)).
"""

import logging
import time
from typing import List, Optional, Dict

import numpy as np

from .config import Settings, C, CA_CHIPPING_RATE
from .prn_generator import generate_ca_code, make_ca_table_fft, resample_code
from .detection import detect_signal_full, DetectionResult
from .data_loader import load_signal_for_acquisition, validate_chunk

logger = logging.getLogger("gnss_receiver.acquisition")


class AcquisitionResult:
    """Container for acquisition results for all satellites."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.results: Dict[int, DetectionResult] = {}
        self.correlation_surfaces: Dict[int, np.ndarray] = {}
        self.execution_time: float = 0.0
    
    @property
    def detected_prns(self) -> List[int]:
        """List of PRNs (1-based) with detected signals."""
        return sorted([
            prn for prn, res in self.results.items()
            if res.detected
        ])
    
    @property
    def strong_prns(self) -> List[int]:
        """List of PRNs with strong (non-weak) detections."""
        return sorted([
            prn for prn, res in self.results.items()
            if res.detected and res.flag == "ok"
        ])
    
    @property
    def weak_prns(self) -> List[int]:
        """List of PRNs with weak detections."""
        return sorted([
            prn for prn, res in self.results.items()
            if res.detected and res.flag == "weak"
        ])
    
    def summary(self) -> dict:
        """Return a summary dictionary."""
        detected = self.detected_prns
        return {
            "total_detected": len(detected),
            "strong": len(self.strong_prns),
            "weak": len(self.weak_prns),
            "detected_prns": detected,
            "execution_time_sec": self.execution_time,
        }
    
    def get_channel_list(self) -> List[dict]:
        """
        Get list of detected channels sorted by signal strength.
        
        Returns list of dicts with keys:
        prn, acquired_freq, code_phase, status
        """
        channels = []
        for prn in self.detected_prns:
            res = self.results[prn]
            channels.append({
                "prn": prn,
                "acquired_freq": self.settings.f_if + res.doppler_hz,
                "code_phase": res.code_phase_samples,
                "status": "T",  # Tracking
                "snr": res.snr,
            })
        
        # Sort by SNR (strongest first)
        channels.sort(key=lambda x: x["snr"], reverse=True)
        
        return channels


def acquire_single_prn(
    signal: np.ndarray,
    ca_code_fft_conj: np.ndarray,
    fs: float,
    f_if: float,
    doppler_range: float,
    doppler_step: float,
    threshold: float,
) -> tuple:
    """
    Acquire a single satellite using Parallel Code Phase Search.
    
    Parameters
    ----------
    signal : np.ndarray
        1 ms of complex I/Q signal (samples_per_code samples).
    ca_code_fft_conj : np.ndarray
        Conjugate FFT of the resampled C/A code.
    fs : float
        Sampling frequency.
    f_if : float
        Intermediate frequency.
    doppler_range : float
        Doppler search range.
    doppler_step : float
        Doppler search step.
    threshold : float
        SNR detection threshold.
    
    Returns
    -------
    tuple
        (correlation_2d, detection_result)
    """
    samples_per_code = len(signal)
    ts = 1.0 / fs
    
    # Number of Doppler frequency bins
    n_doppler = int(np.round(doppler_range * 2 / doppler_step) + 1)
    
    # Phase points for carrier mixing
    time_axis = np.arange(samples_per_code) * ts
    
    # 2D correlation surface: Doppler x Code Phase
    correlation = np.zeros((n_doppler, samples_per_code))
    
    # Remove DC component
    signal_dc = signal - signal.mean()
    
    for fi in range(n_doppler):
        # Trial Doppler frequency
        fd = -doppler_range + doppler_step * fi
        freq = f_if + fd
        
        # Mix signal to baseband (remove carrier)
        carrier = np.exp(-1j * 2 * np.pi * freq * time_axis)
        baseband = signal_dc * carrier
        
        # FFT-based correlation (PCPS)
        baseband_fft = np.fft.fft(baseband)
        corr = np.abs(np.fft.ifft(baseband_fft * ca_code_fft_conj)) ** 2
        
        correlation[fi, :] = corr
    
    # Detect signal
    det_result = detect_signal_full(
        correlation, fs, doppler_range, doppler_step, threshold
    )
    
    return correlation, det_result


def acquire_single_prn_noncoherent(
    signal: np.ndarray,
    ca_code_fft_conj: np.ndarray,
    fs: float,
    f_if: float,
    doppler_range: float,
    doppler_step: float,
    threshold: float,
    n_noncoherent: int = 1,
    coherent_ms: int = 1,
) -> tuple:
    """
    Acquire a single satellite with non-coherent integration.
    
    Multiple coherent correlations are summed incoherently
    (sum of magnitudes squared) to improve sensitivity.
    
    Parameters
    ----------
    signal : np.ndarray
        Complex I/Q signal (must be long enough for all integrations).
    ca_code_fft_conj : np.ndarray
        Conjugate FFT of the resampled C/A code.
    fs, f_if, doppler_range, doppler_step, threshold : see acquire_single_prn
    n_noncoherent : int
        Number of non-coherent integrations.
    coherent_ms : int
        Coherent integration time in ms.
    
    Returns
    -------
    tuple
        (correlation_2d, detection_result)
    """
    samples_per_code = int(fs * 1e-3)  # samples per 1 ms
    samples_needed = samples_per_code * coherent_ms
    ts = 1.0 / fs
    
    n_doppler = int(np.round(doppler_range * 2 / doppler_step) + 1)
    
    # Accumulated correlation (non-coherent sum)
    correlation_acc = np.zeros((n_doppler, samples_per_code))
    
    total_needed = samples_needed * n_noncoherent
    if len(signal) < total_needed:
        logger.warning(
            f"Signal too short for {n_noncoherent} non-coherent integrations "
            f"of {coherent_ms} ms. Using {len(signal) // samples_needed} integrations."
        )
        n_noncoherent = len(signal) // samples_needed
    
    for nc in range(n_noncoherent):
        start = nc * samples_needed
        end = start + samples_needed
        
        if coherent_ms == 1:
            sig_block = signal[start:end]
        else:
            # For coherent_ms > 1, concatenate multiple code periods
            # (must handle potential bit transitions)
            sig_block = signal[start:end]
        
        sig_dc = sig_block - sig_block.mean()
        time_axis = np.arange(len(sig_dc)) * ts
        
        for fi in range(n_doppler):
            fd = -doppler_range + doppler_step * fi
            freq = f_if + fd
            
            carrier = np.exp(-1j * 2 * np.pi * freq * time_axis)
            baseband = sig_dc * carrier
            
            baseband_fft = np.fft.fft(baseband)
            corr = np.abs(np.fft.ifft(baseband_fft * ca_code_fft_conj)) ** 2
            
            # Non-coherent accumulation
            if coherent_ms == 1:
                correlation_acc[fi, :] += corr
            else:
                # For multi-ms coherent: fold back to 1 ms
                correlation_acc[fi, :] += corr[:samples_per_code]
    
    det_result = detect_signal_full(
        correlation_acc, fs, doppler_range, doppler_step, threshold
    )
    
    return correlation_acc, det_result


def acquire_all_prns(
    settings: Settings,
    signal: np.ndarray,
) -> AcquisitionResult:
    """
    Run acquisition for all satellites in the PRN list.
    
    Parameters
    ----------
    settings : Settings
        Receiver settings.
    signal : np.ndarray
        Complex I/Q signal data.
    
    Returns
    -------
    AcquisitionResult
        Acquisition results for all satellites.
    """
    result = AcquisitionResult(settings)
    
    logger.info(
        f"Starting acquisition: {len(settings.acq_satellite_list)} PRNs, "
        f"Doppler ±{settings.acq_doppler_range:.0f} Hz, "
        f"step {settings.acq_doppler_step:.0f} Hz, "
        f"threshold {settings.acq_threshold:.1f}, "
        f"coherent {settings.acq_coherent_ms} ms, "
        f"noncoherent {settings.acq_noncoherent}"
    )
    
    # Precompute FFT of all C/A codes
    ca_table_fft = make_ca_table_fft(settings)
    
    t_start = time.time()
    
    detected_summary = []
    
    for prn in settings.acq_satellite_list:
        prn_idx = prn - 1  # 0-based index for table lookup
        
        if settings.acq_noncoherent > 1 or settings.acq_coherent_ms > 1:
            correlation, det = acquire_single_prn_noncoherent(
                signal,
                ca_table_fft[prn_idx],
                settings.fs,
                settings.f_if,
                settings.acq_doppler_range,
                settings.acq_doppler_step,
                settings.acq_threshold,
                settings.acq_noncoherent,
                settings.acq_coherent_ms,
            )
        else:
            correlation, det = acquire_single_prn(
                signal[:settings.samples_per_code],
                ca_table_fft[prn_idx],
                settings.fs,
                settings.f_if,
                settings.acq_doppler_range,
                settings.acq_doppler_step,
                settings.acq_threshold,
            )
        
        result.results[prn] = det
        
        if settings.save_correlation:
            result.correlation_surfaces[prn] = correlation
        
        if det.detected:
            detected_summary.append(f"{prn:02d}({det.snr:.1f})")
        else:
            detected_summary.append(".")
    
    result.execution_time = time.time() - t_start
    
    logger.info("Satellite scan: " + " ".join(detected_summary))
    logger.info(f"Acquisition completed in {result.execution_time:.1f} s")
    logger.info(
        f"Detected {len(result.detected_prns)} satellites: "
        f"{result.detected_prns}"
    )
    
    return result


def prepare_for_tracking(result: AcquisitionResult, max_channels: int = 8) -> List[dict]:
    """
    Prepare acquisition results for the tracking stage.
    
    Sorts detected satellites by signal strength and selects
    the strongest ones (up to max_channels).
    
    Parameters
    ----------
    result : AcquisitionResult
        Acquisition results.
    max_channels : int
        Maximum number of tracking channels.
    
    Returns
    -------
    list
        Channel initialization data sorted by signal strength.
    """
    channels = result.get_channel_list()
    
    # Limit to max_channels
    channels = channels[:max_channels]
    
    logger.info(f"Prepared {len(channels)} channels for tracking:")
    logger.info(
        f"{'Channel':>7} {'PRN':>4} {'Frequency':>14} {'Doppler':>8} "
        f"{'Code Phase':>11} {'Status':>6} {'SNR':>6}"
    )
    logger.info("-" * 60)
    
    for i, ch in enumerate(channels):
        doppler = ch["acquired_freq"] - result.settings.f_if
        logger.info(
            f"{i:>7} {ch['prn']:>4} {ch['acquired_freq']:>14.1f} "
            f"{doppler:>8.0f} {ch['code_phase']:>11} "
            f"{ch['status']:>6} {ch['snr']:>6.1f}"
        )
    
    return channels
