"""
Configuration management for the GNSS receiver.

Handles loading settings from JSON metadata, CLI arguments,
and providing defaults for GPS L1 C/A signal processing.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Physical constants
C = 299_792_458.0          # Speed of light, m/s
GPS_PI = 3.1415926535898   # Pi used in GPS coordinate system

# GPS L1 C/A signal constants
GPS_L1_FREQ = 1575.42e6    # GPS L1 carrier frequency, Hz
CA_CODE_LENGTH = 1023      # Chips in one C/A code period
CA_CHIPPING_RATE = 1.023e6 # C/A code chipping rate, Hz
CA_CODE_PERIOD = 1e-3      # C/A code period, seconds (1 ms)
NAV_BIT_RATE = 50          # Navigation message bit rate, bits/s
NAV_BIT_PERIOD = 20e-3     # Navigation bit period, seconds (20 ms)


@dataclass
class Settings:
    """Receiver configuration settings."""

    # ===== Signal parameters =====
    fs: float = 50_000_000.0           # Sampling frequency, Hz
    f_if: float = 0.0                  # Intermediate frequency, Hz (0 = baseband)
    data_type: str = "int16"           # Data type of samples in .bin file
    file_name: str = ""                # Path to .bin data file
    
    # ===== Acquisition parameters =====
    acq_doppler_range: float = 10_000  # Max Doppler search range, Hz
    acq_doppler_step: float = 500      # Doppler search step, Hz
    acq_threshold: float = 2.5         # SNR detection threshold
    acq_coherent_ms: int = 1           # Coherent integration time, ms
    acq_noncoherent: int = 1           # Number of non-coherent integrations
    acq_satellite_list: List[int] = field(default_factory=lambda: list(range(1, 33)))

    # ===== Tracking loop parameters =====
    dll_damping_ratio: float = 0.7
    dll_noise_bandwidth: float = 2.0
    dll_correlator_spacing: float = 0.5
    pll_damping_ratio: float = 0.7
    pll_noise_bandwidth: float = 25.0
    
    # ===== Navigation parameters =====
    ms_to_process: float = 37_000      # Milliseconds of data to process
    nav_sol_period: float = 500        # Navigation solution period, ms
    elevation_mask: float = 10.0       # Elevation mask, degrees
    use_trop_corr: bool = True
    number_of_channels: int = 8
    skip_acquisition: bool = False
    skip_number_of_bytes: int = 0
    
    # ===== Processing =====
    offset: int = 0                    # Starting sample offset in file
    blocks: int = 1                    # Number of blocks to process
    
    # ===== Output =====
    log_level: str = "INFO"
    output_path: Optional[str] = None
    save_correlation: bool = False
    plot_tracking: bool = False
    
    # ===== Derived properties =====
    @property
    def samples_per_code(self) -> int:
        """Number of samples per C/A code period."""
        return int(np.round(self.fs * CA_CODE_PERIOD))
    
    @property
    def ts(self) -> float:
        """Sampling period."""
        return 1.0 / self.fs
    
    @property
    def dtype(self) -> np.dtype:
        """NumPy data type for reading samples."""
        if self.data_type == "int8":
            return np.dtype("int8")
        elif self.data_type == "int16":
            return np.dtype("<i2")  # little-endian int16
        elif self.data_type == "float32":
            return np.dtype("<f4")  # little-endian float32
        elif self.data_type == "complex64":
            return np.dtype("complex64")
        else:
            return np.dtype(self.data_type)
    
    @property
    def bytes_per_sample(self) -> int:
        """Bytes per scalar sample."""
        return self.dtype.itemsize
    
    @property
    def bytes_per_iq_pair(self) -> int:
        """Bytes per complex I/Q pair."""
        return 2 * self.bytes_per_sample

    @property
    def start_offset(self) -> float:
        return 68.802

    def calc_loop_coef(self, lbw: float, zeta: float, k: float):
        """Calculate loop filter coefficients for PLL/DLL."""
        wn = lbw * 8.0 * zeta / (4.0 * zeta ** 2 + 1)
        tau1 = k / (wn * wn)
        tau2 = 2.0 * zeta / wn
        return tau1, tau2


def parse_zenodo_json(json_path: str, bin_path: Optional[str] = None) -> dict:
    """
    Parse Zenodo metadata JSON to extract signal parameters.
    
    Returns dict with keys: fs, bin_name, bin_size, signal_type, format_type
    """
    logger = logging.getLogger("gnss_receiver.config")
    
    with open(json_path, "r") as f:
        metadata = json.load(f)
    
    result = {
        "fs": None,
        "bin_name": None,
        "bin_size": None,
        "signal_type": "GPS L1",
        "format_type": "interleaved 32-bit float",
    }
    
    # Extract sampling frequency from description
    description = metadata.get("metadata", {}).get("description", "")
    
    # Try to find "XX MSps" pattern
    match = re.search(r"(\d+(?:\.\d+)?)\s*MSps", description, re.IGNORECASE)
    if match:
        result["fs"] = float(match.group(1)) * 1e6
        logger.info(f"Extracted sampling frequency: {result['fs']/1e6:.1f} MSps")
    
    # Extract .bin file info from entries
    entries = metadata.get("files", {}).get("entries", {})
    for key, entry in entries.items():
        if key.endswith(".bin"):
            result["bin_name"] = key
            result["bin_size"] = entry.get("size", 0)
            break
    
    # If a local bin_path is provided, use its size for validation
    if bin_path and Path(bin_path).exists():
        actual_size = Path(bin_path).stat().st_size
        if result["bin_size"] and actual_size != result["bin_size"]:
            logger.warning(
                f"File size mismatch: expected {result['bin_size']} from JSON, "
                f"but actual file is {actual_size} bytes. "
                f"File may be a cropped version."
            )
        result["local_bin_size"] = actual_size
    
    return result


def detect_data_format(bin_path: str) -> str:
    """
    Auto-detect the data format of a .bin file.
    
    Checks if data looks like float32 or int16 by examining
    the first samples for NaN/Inf and reasonable value ranges.
    """
    logger = logging.getLogger("gnss_receiver.config")
    
    raw = np.fromfile(bin_path, dtype=np.uint8, count=4096)
    if len(raw) == 0:
        logger.warning("Empty data file; defaulting to int16")
        return "int16"

    # Little-endian float IQ samples are finite and have a nontrivial amplitude.
    # Viewing packed integer IQ as float32 often produces tiny subnormals or
    # enormous values, so reject those instead of misclassifying int16 captures.
    float_bytes = raw[:len(raw) - (len(raw) % 4)]
    data_f32 = np.frombuffer(float_bytes.tobytes(), dtype="<f4")
    if len(data_f32):
        finite = data_f32[np.isfinite(data_f32)]
        if len(finite) >= 0.99 * len(data_f32):
            amplitude = np.abs(finite)
            max_amplitude = float(amplitude.max()) if len(amplitude) else 0.0
            rms = float(np.sqrt(np.mean(finite.astype(np.float64) ** 2))) if len(finite) else 0.0
            if 1e-8 < rms < 1e3 and max_amplitude < 1e3:
                logger.info("Detected data format: float32")
                return "float32"

    # The receiver's default integer format is little-endian int16 IQ.
    int16_bytes = raw[:len(raw) - (len(raw) % 2)]
    data_i16 = np.frombuffer(int16_bytes.tobytes(), dtype="<i2")
    if np.any(data_i16):
        logger.info("Detected data format: int16")
        return "int16"

    # Fall back to int8 for low-amplitude integer captures.
    if np.any(raw):
        logger.info("Detected data format: int8")
        return "int8"

    logger.warning("Could not auto-detect data format, defaulting to int16")
    return "int16"


def build_settings_from_args(args: argparse.Namespace) -> Settings:
    """Build Settings object from parsed CLI arguments."""
    logger = logging.getLogger("gnss_receiver.config")
    
    settings = Settings()
    
    # Parse JSON metadata if provided
    if args.json:
        meta = parse_zenodo_json(args.json, args.data)
        if meta["fs"]:
            settings.fs = meta["fs"]
    
    # Apply CLI overrides
    if args.data:
        settings.file_name = args.data
    
    if args.fs:
        settings.fs = float(args.fs)
    
    if args.doppler_range:
        settings.acq_doppler_range = float(args.doppler_range)
    
    if args.doppler_step:
        settings.acq_doppler_step = float(args.doppler_step)
    
    if args.coherent_ms:
        settings.acq_coherent_ms = int(args.coherent_ms)
    
    if args.noncoherent:
        settings.acq_noncoherent = int(args.noncoherent)
    
    if args.threshold:
        settings.acq_threshold = float(args.threshold)
    
    if args.prn:
        settings.acq_satellite_list = [int(x) for x in args.prn.split(",")]
    
    if args.offset:
        settings.offset = int(args.offset)
    
    if args.blocks:
        settings.blocks = int(args.blocks)

    settings.ms_to_process = int(args.ms_to_process)
    settings.nav_sol_period = int(args.nav_sol_period_ms)
    settings.number_of_channels = int(args.channels)
    
    if args.output:
        settings.output_path = args.output
    
    if args.log_level:
        settings.log_level = args.log_level
    
    if args.if_freq:
        settings.f_if = float(args.if_freq)
    
    if args.save_correlation:
        settings.save_correlation = True
    
    if args.plot:
        settings.plot_tracking = True
    
    # Auto-detect data format if not specified
    if args.data_format:
        settings.data_type = args.data_format
    elif settings.file_name and Path(settings.file_name).exists():
        settings.data_type = detect_data_format(settings.file_name)
    
    # Validate
    if settings.fs <= 0:
        logger.error("Sampling frequency must be positive")
        sys.exit(1)

    if settings.ms_to_process <= 0:
        logger.error("--ms-to-process must be positive")
        sys.exit(1)

    if settings.nav_sol_period <= 0:
        logger.error("--nav-sol-period-ms must be positive")
        sys.exit(1)

    if not 1 <= settings.number_of_channels <= 32:
        logger.error("--channels must be between 1 and 32")
        sys.exit(1)

    if not settings.file_name:
        logger.error("No data file specified. Use --data <path>")
        sys.exit(1)
    
    if not Path(settings.file_name).exists():
        logger.error(f"Data file not found: {settings.file_name}")
        sys.exit(1)
    
    # Validate file size
    file_size = Path(settings.file_name).stat().st_size
    if file_size % settings.bytes_per_iq_pair != 0:
        n_pairs = file_size // settings.bytes_per_iq_pair
        trimmed = n_pairs * settings.bytes_per_iq_pair
        logger.warning(
            f"File size ({file_size} bytes) is not a multiple of "
            f"{settings.bytes_per_iq_pair} bytes. "
            f"Will read {n_pairs} I/Q pairs ({trimmed} bytes)."
        )
    
    total_samples = file_size // settings.bytes_per_iq_pair
    duration_ms = total_samples / settings.fs * 1000
    
    if duration_ms < settings.acq_coherent_ms:
        logger.error(
            f"Data duration ({duration_ms:.4f} ms) is less than "
            f"requested coherent integration time ({settings.acq_coherent_ms} ms). "
            f"Acquisition is not possible."
        )
        sys.exit(1)
    
    if duration_ms < 10:
        logger.warning(
            f"Only {duration_ms:.4f} ms of data available. "
            f"Recommend >= 10 ms for reliable acquisition."
        )
    
    return settings


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GNSS Software Receiver — GPS L1 C/A signal processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic acquisition with auto-detected parameters
  python -m gnss_receiver --data signal.bin --json metadata.json

  # Full receiver: acquisition + tracking + navigation solution
  python -m gnss_receiver --data signal.bin --json metadata.json --data-format float32 --mode full

  # Specify sampling rate, format, duration and threshold
  python -m gnss_receiver --data signal.bin --fs 50000000 --data-format int16 --ms-to-process 37000 --threshold 3.0
        """,
    )
    
    parser.add_argument("--data", required=True, help="Path to I/Q data file (.bin)")
    parser.add_argument("--json", help="Path to Zenodo metadata JSON file")
    parser.add_argument("--mode", choices=["acquisition", "full"], default="acquisition",
                       help="acquisition only, or full acquisition/tracking/navigation (default: acquisition)")
    parser.add_argument("--fs", type=float, help="Sampling frequency in Hz")
    parser.add_argument("--if-freq", type=float, help="Intermediate frequency in Hz (default: 0)")
    parser.add_argument("--data-format", choices=["int8", "int16", "float32"],
                       help="Data format (auto-detected if not specified)")
    parser.add_argument("--doppler-range", type=float, default=10000,
                       help="Doppler search range in Hz (default: 10000)")
    parser.add_argument("--doppler-step", type=float, default=500,
                       help="Doppler search step in Hz (default: 500)")
    parser.add_argument("--coherent-ms", type=int, default=1,
                       help="Coherent integration time in ms (default: 1)")
    parser.add_argument("--noncoherent", type=int, default=1,
                       help="Number of non-coherent integrations (default: 1)")
    parser.add_argument("--threshold", type=float, default=2.5,
                       help="SNR detection threshold (default: 2.5)")
    parser.add_argument("--prn", help="Comma-separated list of PRN numbers (default: 1-32)")
    parser.add_argument("--offset", type=int, default=0,
                       help="Starting sample offset in file (default: 0)")
    parser.add_argument("--blocks", type=int, default=1,
                       help="Number of 1ms blocks to process (default: 1)")
    parser.add_argument("--ms-to-process", type=int, default=37000,
                       help="Maximum tracking duration in ms (default: 37000; about 36 s or more is needed for navigation)")
    parser.add_argument("--nav-sol-period-ms", type=int, default=500,
                       help="Interval between navigation solutions in ms (default: 500)")
    parser.add_argument("--channels", type=int, default=8,
                       help="Maximum number of satellites to track in full mode (default: 8)")
    parser.add_argument("--output", help="Path to output JSON file")
    parser.add_argument("--save-correlation", action="store_true",
                       help="Save correlation matrices to .npy files")
    parser.add_argument("--plot", action="store_true",
                       help="Generate and save plots")
    parser.add_argument("--log-level", choices=["ERROR", "WARNING", "INFO", "DEBUG"],
                       default="INFO", help="Logging level (default: INFO)")
    
    return parser.parse_args()
