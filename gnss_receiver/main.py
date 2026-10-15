"""
Main entry point for the GNSS Software Receiver.

Orchestrates the complete signal processing pipeline:
1. Parse configuration and metadata
2. Load I/Q data
3. Acquisition (satellite detection)
4. Tracking (optional)
5. Navigation solution (optional)
6. Output results
"""

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .config import parse_args, build_settings_from_args, Settings
from .data_loader import load_signal_for_acquisition, get_file_info
from .acquisition import acquire_all_prns, prepare_for_tracking
from .output import (
    print_acquisition_table,
    format_acquisition_json,
    save_json,
    save_csv,
    plot_acquisition_results,
    plot_correlation_2d,
    plot_tracking_results,
)
from .tracking import TrackingResult, track_all_channels


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the receiver."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def run_acquisition(settings: Settings) -> 'AcquisitionResult':
    """
    Run the acquisition stage.
    
    Parameters
    ----------
    settings : Settings
        Receiver settings.
    
    Returns
    -------
    AcquisitionResult
        Acquisition results.
    """
    logger = logging.getLogger("gnss_receiver.main")
    
    # Print file info
    info = get_file_info(settings)
    logger.info(f"Data file: {info['file_name']}")
    logger.info(f"File size: {info['file_size_bytes']:,} bytes")
    logger.info(f"Data type: {info['data_type']}")
    logger.info(f"Total samples: {info['total_samples']:,}")
    logger.info(f"Duration: {info['duration_ms']:.2f} ms")
    logger.info(f"Sampling freq: {info['sampling_freq_mhz']:.1f} MSps")
    logger.info(f"Samples/code: {info['samples_per_code_period']}")
    
    # Load signal data for acquisition (need at least 2 ms)
    signal = load_signal_for_acquisition(
        settings,
        num_ms=max(2, settings.acq_coherent_ms + 1),
        sample_offset=settings.offset,
    )
    
    # Run acquisition
    result = acquire_all_prns(settings, signal)
    
    return result


def _tracking_cache_path(settings: Settings) -> Path:
    """Return a cache name tied to the input and tracking configuration."""
    source = Path(settings.file_name)
    stat = source.stat()
    signature = {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "fs": settings.fs,
        "if": settings.f_if,
        "format": settings.data_type,
        "offset": settings.offset,
        "skip_bytes": settings.skip_number_of_bytes,
        "duration_ms": settings.ms_to_process,
        "channels": settings.number_of_channels,
        "acq_range": settings.acq_doppler_range,
        "acq_step": settings.acq_doppler_step,
        "acq_threshold": settings.acq_threshold,
        "acq_prns": settings.acq_satellite_list,
        "coherent_ms": settings.acq_coherent_ms,
        "noncoherent": settings.acq_noncoherent,
        "dll": [settings.dll_noise_bandwidth, settings.dll_damping_ratio,
                settings.dll_correlator_spacing],
        "pll": [settings.pll_noise_bandwidth, settings.pll_damping_ratio],
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return Path(f"{source.stem}_{digest}_tracking_v2.npy")


def _navigation_unavailable(message: str) -> dict:
    return {"status": "unavailable", "message": message, "valid_ephemerides": [], "solutions": []}


def run_full_receiver(settings: Settings) -> dict:
    """Run acquisition, tracking, LNAV decoding, and PVT position solution."""
    logger = logging.getLogger("gnss_receiver.main")

    logger.info("=" * 60)
    logger.info("STEP 1: ACQUISITION")
    logger.info("=" * 60)
    acq_result = run_acquisition(settings)
    if not acq_result.detected_prns:
        message = "No GPS satellites were acquired; no navigation solution is possible."
        logger.error(message)
        return {
            "acquisition": format_acquisition_json(acq_result),
            "tracking_channels": 0,
            "navigation": _navigation_unavailable(message),
        }

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: PREPARING TRACKING CHANNELS")
    logger.info("=" * 60)
    channels = prepare_for_tracking(acq_result, max_channels=settings.number_of_channels)
    if not channels:
        message = "No channels were prepared for tracking; no navigation solution is possible."
        logger.error(message)
        return {
            "acquisition": format_acquisition_json(acq_result),
            "tracking_channels": 0,
            "navigation": _navigation_unavailable(message),
        }

    # Do not allocate/traverse more tracking epochs than the file actually has.
    file_info = get_file_info(settings)
    first_sample_byte = int(settings.skip_number_of_bytes) + int(settings.offset) * settings.bytes_per_iq_pair
    available_samples = max(
        0,
        (int(file_info["file_size_bytes"]) - first_sample_byte) // settings.bytes_per_iq_pair,
    )
    available_ms = int(available_samples / settings.fs * 1000.0)
    requested_ms = int(settings.ms_to_process)
    settings.ms_to_process = min(requested_ms, available_ms)
    logger.info(
        "Tracking up to %d ms (requested %d ms; file has %.1f ms after the selected offset)",
        int(settings.ms_to_process), requested_ms, available_samples / settings.fs * 1000.0,
    )
    if settings.ms_to_process < 30_000:
        logger.warning(
            "This recording is shorter than the 30-second GPS ephemeris block; "
            "navigation may not produce coordinates."
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: TRACKING")
    logger.info("=" * 60)
    track_result = None
    cache_path = _tracking_cache_path(settings)
    if cache_path.exists():
        try:
            logger.info("Loading cached tracking results from %s", cache_path)
            cached = np.load(cache_path, allow_pickle=True)
            cached_results = cached.tolist() if hasattr(cached, "tolist") else list(cached)
            if all("absolute_sample" in item and "I_P" in item for item in cached_results):
                track_result = TrackingResult(settings)
                track_result.channels = channels
                track_result.results = cached_results
            else:
                logger.warning("Ignoring tracking cache with an incompatible result format")
        except Exception as error:
            logger.warning("Could not load tracking cache: %s", error)

    if track_result is None:
        track_result = track_all_channels(channels, settings, settings.file_name)
        try:
            np.save(cache_path, np.array(track_result.results, dtype=object))
            logger.info("Tracking results cached to %s", cache_path)
        except Exception as error:
            logger.warning("Could not cache tracking results: %s", error)

    if settings.plot_tracking and track_result.results:
        plot_dir = Path(settings.file_name).stem + "_plots"
        Path(plot_dir).mkdir(exist_ok=True)
        plot_tracking_results(track_result.results, settings, save_dir=str(plot_dir))

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: NAVIGATION MESSAGE AND POSITION")
    logger.info("=" * 60)
    from .pvt import solve_navigation

    navigation = solve_navigation(track_result, settings)
    logger.info("%s", navigation["message"])
    if navigation["status"] == "solved":
        latest = navigation["solutions"][-1]
        logger.info(
            "Latest fix: lat %.8f°, lon %.8f°, h %.2f m; %d satellites; PDOP %.2f",
            latest["latitude_deg"], latest["longitude_deg"], latest["height_m"],
            latest["satellite_count"], latest["dop"]["pdop"],
        )

    return {
        "acquisition": format_acquisition_json(acq_result),
        "tracking_channels": len(track_result.results),
        "tracking_duration_ms": int(settings.ms_to_process),
        "navigation": navigation,
    }


def main() -> int:
    """
    Main entry point.
    
    Returns
    -------
    int
        Exit code (0 = success).
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger("gnss_receiver.main")
    
    logger.info("=" * 60)
    logger.info("GNSS Software Receiver v1.0")
    logger.info("GPS L1 C/A Signal Processing")
    logger.info("=" * 60)
    
    # Build settings
    settings = build_settings_from_args(args)
    
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Data file: {settings.file_name}")
    logger.info(f"Data format: {settings.data_type}")
    logger.info(f"Sampling freq: {settings.fs / 1e6:.1f} MSps")
    logger.info(f"IF frequency: {settings.f_if:.0f} Hz")
    logger.info(f"Threshold: {settings.acq_threshold}")
    
    # Set default output path
    if not settings.output_path:
        stem = Path(settings.file_name).stem
        suffix = "navigation_results" if args.mode == "full" else "acquisition_results"
        settings.output_path = f"{stem}_{suffix}.json"
    
    t_start = time.time()
    
    try:
        if args.mode == "acquisition":
            # Acquisition only
            acq_result = run_acquisition(settings)
            
            # Print results
            print_acquisition_table(acq_result)
            
            # Save JSON output
            output_data = format_acquisition_json(acq_result)
            save_json(output_data, settings.output_path)

            # Save CSV table
            csv_path = str(Path(settings.output_path).with_suffix(".csv"))
            save_csv(acq_result, csv_path)
            
            # Plot if requested
            if args.plot:
                plot_dir = Path(settings.file_name).stem + "_plots"
                Path(plot_dir).mkdir(exist_ok=True)
                
                plot_acquisition_results(
                    acq_result,
                    save_path=str(Path(plot_dir) / "acquisition_results.png"),
                )
                
                # Plot correlation surfaces for detected PRNs
                if settings.save_correlation and acq_result.correlation_surfaces:
                    for prn in acq_result.detected_prns:
                        if prn in acq_result.correlation_surfaces:
                            plot_correlation_2d(
                                acq_result.correlation_surfaces[prn],
                                prn,
                                settings.acq_doppler_range,
                                settings.acq_doppler_step,
                                settings.fs,
                                save_path=str(Path(plot_dir) / f"correlation_prn{prn}.png"),
                            )
            
            exit_code = 0 if len(acq_result.detected_prns) > 0 else 1
        
        elif args.mode == "full":
            # Acquisition + tracking + broadcast navigation/PVT.
            results = run_full_receiver(settings)
            save_json(results, settings.output_path)
            navigation = results.get("navigation", {})
            exit_code = 0 if navigation.get("status") == "solved" else 1
        
        else:
            logger.error(f"Unknown mode: {args.mode}")
            exit_code = 1
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user.")
        exit_code = 130
    
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        exit_code = 1
    
    elapsed = time.time() - t_start
    logger.info(f"Total execution time: {elapsed:.1f} s")
    
    return exit_code


def run():
    """Entry point for console_scripts."""
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
