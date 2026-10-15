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
from .tracking import track_all_channels


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


def run_full_receiver(settings: Settings) -> dict:
    """
    Run the full receiver pipeline: acquisition + tracking + navigation.
    
    Parameters
    ----------
    settings : Settings
        Receiver settings.
    
    Returns
    -------
    dict
        Complete results.
    """
    logger = logging.getLogger("gnss_receiver.main")
    
    # Step 1: Acquisition
    logger.info("=" * 60)
    logger.info("STEP 1: ACQUISITION")
    logger.info("=" * 60)
    
    acq_result = run_acquisition(settings)
    
    if len(acq_result.detected_prns) == 0:
        logger.error("No satellites detected. Cannot proceed to tracking.")
        return {"acquisition": format_acquisition_json(acq_result)}
    
    # Step 2: Prepare channels for tracking
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: PREPARING TRACKING CHANNELS")
    logger.info("=" * 60)
    
    channels = prepare_for_tracking(
        acq_result,
        max_channels=settings.number_of_channels,
    )
    
    if len(channels) == 0:
        logger.error("No channels prepared for tracking.")
        return {"acquisition": format_acquisition_json(acq_result)}
    
    # Step 3: Tracking
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: TRACKING")
    logger.info("=" * 60)
    
    # Try to load cached tracking results
    cache_path = Path(settings.file_name).stem + "_tracking_cache.npy"
    
    track_result = None
    if Path(cache_path).exists():
        try:
            logger.info(f"Loading cached tracking results from {cache_path}")
            cached = np.load(cache_path, allow_pickle=True)
            track_result = type('TrackingResult', (), {
                'results': cached.tolist() if hasattr(cached, 'tolist') else list(cached),
                'channels': channels,
                'settings': settings,
            })()
        except Exception as e:
            logger.warning(f"Could not load cached results: {e}")
    
    if track_result is None:
        track_result = track_all_channels(channels, settings, settings.file_name)
        
        # Save tracking results to cache
        try:
            np.save(cache_path, np.array(track_result.results, dtype=object))
            logger.info(f"Tracking results cached to {cache_path}")
        except Exception as e:
            logger.warning(f"Could not cache tracking results: {e}")
    
    # Step 4: Plot tracking results
    if settings.plot_tracking and track_result.results:
        plot_dir = Path(settings.file_name).stem + "_plots"
        Path(plot_dir).mkdir(exist_ok=True)
        plot_tracking_results(track_result.results, settings, save_dir=str(plot_dir))
    
    return {
        "acquisition": format_acquisition_json(acq_result),
        "tracking_channels": len(track_result.results) if track_result else 0,
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
        settings.output_path = f"{stem}_acquisition_results.json"
    
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
            # Full receiver pipeline
            results = run_full_receiver(settings)
            
            # Save results
            save_json(results, settings.output_path)
            
            exit_code = 0
        
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
