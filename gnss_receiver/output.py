"""
Output formatting module for the GNSS receiver.

Handles printing result tables, saving JSON output,
and optionally generating plots.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import Settings
from .acquisition import AcquisitionResult

logger = logging.getLogger("gnss_receiver.output")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def print_acquisition_table(result: AcquisitionResult) -> None:
    """
    Print a formatted table of acquisition results.
    
    Parameters
    ----------
    result : AcquisitionResult
        Acquisition results.
    """
    header = (
        f"{'PRN':>4} {'Doppler (Hz)':>12} {'Code Phase':>11} {'Code Phase':>12} "
        f"{'SNR':>7} {'Flag':>5} {'Peak Ratio':>11}"
    )
    subheader = (
        f"{'':>4} {'':>12} {'(samples)':>11} {'(chips)':>12} "
        f"{'':>7} {'':>5} {'(1st/2nd)':>11}"
    )
    separator = "=" * 68
    
    print()
    print(separator)
    print("ACQUISITION RESULTS")
    print(separator)
    print(header)
    print(subheader)
    print("-" * 68)
    
    for prn in sorted(result.results.keys()):
        res = result.results[prn]
        if res.detected:
            print(
                f"{prn:>4} {res.doppler_hz:>12.1f} "
                f"{res.code_phase_samples:>11} "
                f"{res.code_phase_chips:>12.2f} "
                f"{res.snr:>7.2f} {res.flag:>5} "
                f"{res.peak_to_second_ratio:>11.2f}"
            )
    
    print("-" * 68)
    
    summary = result.summary()
    print()
    print(f"Total detected satellites: {summary['total_detected']}")
    print(f"  Strong detections (ok): {summary['strong']}")
    print(f"  Weak detections:        {summary['weak']}")
    print(f"  Execution time:         {summary['execution_time_sec']:.1f} s")
    print()
    
    # Print parameters used
    s = result.settings
    print("Parameters used:")
    print(f"  Sampling frequency:   {s.fs / 1e6:.1f} MSps")
    print(f"  IF frequency:         {s.f_if:.0f} Hz")
    print(f"  Doppler range:        ±{s.acq_doppler_range:.0f} Hz")
    print(f"  Doppler step:         {s.acq_doppler_step:.0f} Hz")
    print(f"  Coherent integration: {s.acq_coherent_ms} ms")
    print(f"  Non-coherent sum.:    {s.acq_noncoherent}")
    print(f"  Detection threshold:  {s.acq_threshold:.1f}")
    print(separator)
    print()


def format_acquisition_json(result: AcquisitionResult) -> dict:
    """
    Format acquisition results as a JSON-serializable dictionary.
    
    Parameters
    ----------
    result : AcquisitionResult
        Acquisition results.
    
    Returns
    -------
    dict
        Results in JSON format.
    """
    s = result.settings
    
    satellites = []
    for prn in sorted(result.results.keys()):
        res = result.results[prn]
        if res.detected:
            satellites.append({
                "prn": prn,
                "doppler_hz": round(res.doppler_hz, 2),
                "code_phase_samples": int(res.code_phase_samples),
                "code_phase_chips": round(res.code_phase_chips, 4),
                "snr": round(res.snr, 2),
                "peak_to_second_ratio": round(res.peak_to_second_ratio, 2),
                "flag": res.flag,
            })
    
    summary = result.summary()
    
    output = {
        "input_file": s.file_name,
        "mode": "acquisition",
        "parameters": {
            "fs": s.fs,
            "f_if": s.f_if,
            "doppler_range": [-s.acq_doppler_range, s.acq_doppler_range],
            "doppler_step": s.acq_doppler_step,
            "coherent_time_ms": s.acq_coherent_ms,
            "noncoherent_integrations": s.acq_noncoherent,
            "threshold": s.acq_threshold,
            "prn_list": [int(x) for x in s.acq_satellite_list],
            "data_type": str(s.data_type),
        },
        "satellites": satellites,
        "summary": {
            "total_detected": summary["total_detected"],
            "strong": summary["strong"],
            "weak": summary["weak"],
            "execution_time_sec": round(summary["execution_time_sec"], 2),
        },
    }
    
    return output


def save_json(data: dict, path: str) -> None:
    """
    Save results to a JSON file.
    
    Parameters
    ----------
    data : dict
        Data to save.
    path : str
        Output file path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    logger.info(f"Results saved to: {output_path}")


def save_csv(result: AcquisitionResult, path: str) -> None:
    """
    Save acquisition results to a CSV file.

    One row per detected satellite with columns:
      PRN, Doppler_Hz, CodePhase_samples, CodePhase_chips,
      CodePhase_meters, SNR, PeakRatio, Flag

    Parameters
    ----------
    result : AcquisitionResult
        Acquisition results.
    path : str
        Output file path (should end with .csv).
    """
    import csv

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "PRN",
        "Doppler (Hz)",
        "Code Phase (samples)",
        "Code Phase (chips)",
        "SNR",
        "Flag",
        "Peak Ratio (1st/2nd)",
    ]

    rows = []
    for prn in sorted(result.results.keys()):
        res = result.results[prn]
        if res.detected:
            rows.append([
                prn,
                _num(res.doppler_hz, 1),
                int(res.code_phase_samples),
                _num(res.code_phase_chips, 2),
                _num(res.snr, 2),
                res.flag,
                _num(res.peak_to_second_ratio, 2),
            ])

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)

    logger.info(f"CSV saved to: {output_path} ({len(rows)} satellites)")


def _num(value, decimals):
    """Format a float with comma as decimal separator (EU locale for Excel)."""
    return f"{value:.{decimals}f}".replace(".", ",")


def plot_acquisition_results(
    result: AcquisitionResult,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot acquisition results.
    
    Parameters
    ----------
    result : AcquisitionResult
        Acquisition results.
    save_path : str, optional
        Path to save the plot. If None, displays interactively.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    # Bar chart of acquisition metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    
    prns = list(range(1, 33))
    snrs = []
    colors = []
    
    for prn in prns:
        if prn in result.results and result.results[prn].detected:
            snrs.append(result.results[prn].snr)
            if result.results[prn].flag == "ok":
                colors.append("green")
            else:
                colors.append("orange")
        else:
            snrs.append(0)
            colors.append("lightgray")
    
    bars = ax.bar(prns, snrs, color=colors, edgecolor="black", linewidth=0.5)
    
    ax.axhline(y=result.settings.acq_threshold, color="red", linestyle="--",
               label=f"Threshold ({result.settings.acq_threshold})")
    
    ax.set_xlabel("PRN Number")
    ax.set_ylabel("SNR (Acquisition Metric)")
    ax.set_title("GPS L1 C/A Acquisition Results")
    ax.set_xticks(prns)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add PRN labels on detected bars
    for prn in prns:
        if prn in result.results and result.results[prn].detected:
            ax.text(prn, snrs[prn - 1] + 0.3, str(prn),
                   ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Acquisition plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_correlation_2d(
    correlation: np.ndarray,
    prn: int,
    doppler_range: float,
    doppler_step: float,
    fs: float,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot 2D correlation surface for a single PRN.
    
    Parameters
    ----------
    correlation : np.ndarray
        2D correlation array (n_doppler x n_code_phases).
    prn : int
        PRN number.
    doppler_range : float
        Doppler range.
    doppler_step : float
        Doppler step.
    fs : float
        Sampling frequency.
    save_path : str, optional
        Path to save the plot.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    n_doppler, n_samples = correlation.shape
    doppler_axis = np.linspace(-doppler_range, doppler_range, n_doppler)
    code_axis = np.arange(n_samples)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 2D surface
    ax = axes[0]
    im = ax.pcolormesh(code_axis, doppler_axis, correlation, shading='auto', cmap='jet')
    plt.colorbar(im, ax=ax, label='Correlation Power')
    ax.set_xlabel('Code Phase (samples)')
    ax.set_ylabel('Doppler (Hz)')
    ax.set_title(f'PRN {prn}: 2D Correlation Surface')
    
    # 1D slice at best Doppler
    peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    ax = axes[1]
    ax.plot(code_axis, correlation[peak_idx[0], :], 'b-')
    ax.set_xlabel('Code Phase (samples)')
    ax.set_ylabel('Correlation Power')
    ax.set_title(f'PRN {prn}: Code Phase Slice (Doppler={doppler_axis[peak_idx[0]]:.0f} Hz)')
    ax.grid(True, alpha=0.3)
    
    # 1D slice at best code phase
    ax = axes[2]
    ax.plot(doppler_axis, correlation[:, peak_idx[1]], 'r-')
    ax.set_xlabel('Doppler (Hz)')
    ax.set_ylabel('Correlation Power')
    ax.set_title(f'PRN {prn}: Doppler Slice (Code Phase={peak_idx[1]})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Correlation plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_tracking_results(
    track_results: List[dict],
    settings: Settings,
    save_dir: Optional[str] = None,
) -> None:
    """
    Plot tracking results for all channels.
    
    Parameters
    ----------
    track_results : list
        List of tracking result dicts.
    settings : Settings
        Receiver settings.
    save_dir : str, optional
        Directory to save plots.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    for ch_nr, tr in enumerate(track_results):
        time_axis = np.arange(len(tr["I_P"])) / 1000.0
        if len(time_axis) == 0:
            continue
        prn = tr["PRN"]
        
        fig = plt.figure(figsize=(14, 10))
        spec = gridspec.GridSpec(3, 3)
        
        # Scatter plot
        ax = fig.add_subplot(spec[0, 0])
        ax.plot(tr["I_P"], tr["Q_P"], '.', markersize=0.5)
        ax.set_aspect('equal')
        ax.set_title('I/Q Scatter Plot')
        ax.set_xlabel('I prompt')
        ax.set_ylabel('Q prompt')
        ax.grid(True)
        
        # Navigation bits
        ax = fig.add_subplot(spec[0, 1:])
        ax.plot(time_axis, tr["I_P"])
        ax.set_title('Navigation Message Bits')
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        ax.set_xlim([0, time_axis[-1]])
        
        # PLL discriminator
        ax = fig.add_subplot(spec[1, 0])
        ax.plot(time_axis, tr["pll_discr"], 'r', linewidth=0.5)
        ax.set_title('Raw PLL Discriminator')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.set_xlim([0, time_axis[-1]])
        
        # Correlation
        ax = fig.add_subplot(spec[1, 1:])
        ax.plot(time_axis, np.sqrt(tr["I_E"]**2 + tr["Q_E"]**2),
                time_axis, np.sqrt(tr["I_P"]**2 + tr["Q_P"]**2),
                time_axis, np.sqrt(tr["I_L"]**2 + tr["Q_L"]**2))
        ax.set_title('Correlation Results')
        ax.set_xlabel('Time (s)')
        ax.legend(['Early', 'Prompt', 'Late'])
        ax.grid(True)
        ax.set_xlim([0, time_axis[-1]])
        
        # PLL filtered
        ax = fig.add_subplot(spec[2, 0])
        ax.plot(time_axis, tr["pll_discr_filt"], 'b', linewidth=0.5)
        ax.set_title('Filtered PLL')
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        ax.set_xlim([0, time_axis[-1]])
        
        # DLL discriminator
        ax = fig.add_subplot(spec[2, 1])
        ax.plot(time_axis, tr["dll_discr"], 'r', linewidth=0.5)
        ax.set_title('Raw DLL Discriminator')
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        ax.set_xlim([0, time_axis[-1]])
        
        # DLL filtered
        ax = fig.add_subplot(spec[2, 2])
        ax.plot(time_axis, tr["dll_discr_filt"], 'b', linewidth=0.5)
        ax.set_title('Filtered DLL')
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        ax.set_xlim([0, time_axis[-1]])
        
        fig.suptitle(f'Channel {ch_nr} — PRN {prn}', fontsize=14)
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f"tracking_ch{ch_nr}_prn{prn}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Tracking plot saved to: {save_path}")
        
        plt.close()
