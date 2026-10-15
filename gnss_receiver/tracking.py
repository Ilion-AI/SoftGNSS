"""
GPS L1 C/A signal tracking with DLL and PLL.

Implements code tracking (Delay Lock Loop) and carrier tracking
(Phase Lock Loop) for each acquired satellite signal.
"""

import logging
from typing import List, Optional

import numpy as np

from .config import Settings, CA_CODE_LENGTH, CA_CHIPPING_RATE
from .prn_generator import generate_ca_code

logger = logging.getLogger("gnss_receiver.tracking")


class TrackingResult:
    """Container for tracking results for all channels."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.channels: List[dict] = []
        self.results: List[dict] = []
    
    def get_tracked_data(self, channel_nr: int) -> dict:
        """Get tracking data for a specific channel."""
        if channel_nr < len(self.results):
            return self.results[channel_nr]
        return {}


def track_channel(
    fid,
    channel: dict,
    settings: Settings,
    channel_nr: int,
) -> Optional[dict]:
    """
    Track a single satellite channel.
    
    Parameters
    ----------
    fid : file
        Open file object positioned at the start of data.
    channel : dict
        Channel initialization data from acquisition.
    settings : Settings
        Receiver settings.
    channel_nr : int
        Channel number (for logging).
    
    Returns
    -------
    dict or None
        Tracking results for this channel, or None if tracking fails.
    """
    requested_ms = max(0, int(settings.ms_to_process))

    # Acquisition's code phase is relative to the user-selected sample offset.
    # Account for both that offset and the code phase when seeking this channel.
    byte_offset = int(settings.skip_number_of_bytes)
    byte_offset += (int(settings.offset) + int(channel["code_phase"])) * settings.bytes_per_iq_pair
    current_pos = fid.tell()
    fid.seek(0, 2)
    file_size = fid.tell()
    fid.seek(current_pos, 0)
    available_bytes = max(0, file_size - byte_offset)
    available_samples = available_bytes // settings.bytes_per_iq_pair
    samples_per_ms = settings.fs * 1e-3
    available_ms = int(available_samples / samples_per_ms) if samples_per_ms > 0 else 0
    ms_to_process = min(requested_ms, available_ms)
    if ms_to_process <= 0:
        logger.warning("Ch %d: no samples available for tracking", channel_nr)
        return None
    
    # DLL parameters
    early_late_spacing = settings.dll_correlator_spacing
    pdi_code = 0.001  # 1 ms integration
    tau1_code, tau2_code = settings.calc_loop_coef(
        settings.dll_noise_bandwidth, settings.dll_damping_ratio, 1.0
    )
    
    # PLL parameters
    pdi_carr = 0.001  # 1 ms integration
    tau1_carr, tau2_carr = settings.calc_loop_coef(
        settings.pll_noise_bandwidth, settings.pll_damping_ratio, 0.25
    )
    
    # Output arrays
    absolute_sample = np.zeros(ms_to_process)
    code_freq = np.full(ms_to_process, np.inf)
    carr_freq = np.full(ms_to_process, np.inf)
    I_P = np.zeros(ms_to_process)
    I_E = np.zeros(ms_to_process)
    I_L = np.zeros(ms_to_process)
    Q_E = np.zeros(ms_to_process)
    Q_P = np.zeros(ms_to_process)
    Q_L = np.zeros(ms_to_process)
    dll_discr = np.full(ms_to_process, np.inf)
    dll_discr_filt = np.full(ms_to_process, np.inf)
    pll_discr = np.full(ms_to_process, np.inf)
    pll_discr_filt = np.full(ms_to_process, np.inf)
    
    prn = channel["prn"]
    
    # Seek to the same recording offset used by acquisition, at the acquired
    # code phase for this PRN.
    fid.seek(byte_offset, 0)
    
    # Generate C/A code for this PRN
    ca_code = generate_ca_code(prn - 1)  # 0-based
    # Add wrap-around chips for early/late correlators
    ca_code_ext = np.r_[ca_code[-1], ca_code, ca_code[0]]
    
    # Initial tracking parameters
    code_freq_nco = CA_CHIPPING_RATE
    rem_code_phase = 0.0
    carr_freq_val = channel["acquired_freq"]
    carr_freq_basis = channel["acquired_freq"]
    rem_carr_phase = 0.0
    old_code_nco = 0.0
    old_code_error = 0.0
    old_carr_nco = 0.0
    old_carr_error = 0.0
    processed_ms = 0
    
    for loop_cnt in range(ms_to_process):
        # Progress logging
        if loop_cnt % 1000 == 0:
            logger.info(
                f"Tracking: Ch {channel_nr + 1}/{settings.number_of_channels} "
                f"PRN#{prn:02d} - {loop_cnt}/{ms_to_process} ms"
            )
        
        # Calculate block size
        code_phase_step = code_freq_nco / settings.fs
        blksize = int(np.ceil((CA_CODE_LENGTH - rem_code_phase) / code_phase_step))
        
        # Read raw signal
        raw_iq = np.fromfile(fid, settings.dtype, blksize * 2)
        
        if len(raw_iq) != blksize * 2:
            logger.warning(
                f"Ch {channel_nr}: Could not read enough samples at ms {loop_cnt}. "
                f"Stopping tracking."
            )
            break
        
        # Convert to complex
        raw_signal = (
            raw_iq[0::2].astype(np.float64) +
            1j * raw_iq[1::2].astype(np.float64)
        )
        
        # Generate early, prompt, and late code replicas
        tcode_e = np.linspace(
            rem_code_phase - early_late_spacing,
            blksize * code_phase_step + rem_code_phase - early_late_spacing,
            blksize, endpoint=False,
        )
        early_code = ca_code_ext[np.ceil(tcode_e).astype(np.int64)]
        
        tcode_l = np.linspace(
            rem_code_phase + early_late_spacing,
            blksize * code_phase_step + rem_code_phase + early_late_spacing,
            blksize, endpoint=False,
        )
        late_code = ca_code_ext[np.ceil(tcode_l).astype(np.int64)]
        
        tcode_p = np.linspace(
            rem_code_phase,
            blksize * code_phase_step + rem_code_phase,
            blksize, endpoint=False,
        )
        prompt_code = ca_code_ext[np.ceil(tcode_p).astype(np.int64)]
        
        rem_code_phase = tcode_p[-1] + code_phase_step - CA_CODE_LENGTH
        
        # Generate carrier
        time_vec = np.arange(0, blksize + 1) / settings.fs
        trig_arg = carr_freq_val * 2.0 * np.pi * time_vec + rem_carr_phase
        rem_carr_phase = trig_arg[blksize] % (2 * np.pi)
        
        # Remove the residual carrier from the complex signal.
        carrier = np.exp(-1j * trig_arg[:blksize])
        baseband = raw_signal * carrier
        
        # Correlate with early, prompt, late codes
        i_e = np.sum(early_code * baseband.real)
        q_e = np.sum(early_code * baseband.imag)
        i_p = np.sum(prompt_code * baseband.real)
        q_p = np.sum(prompt_code * baseband.imag)
        i_l = np.sum(late_code * baseband.real)
        q_l = np.sum(late_code * baseband.imag)
        
        # Costas discriminator is insensitive to 180-degree GPS data-bit flips.
        # Its output is the phase error in carrier cycles, in [-0.25, 0.25].
        carr_error = np.arctan2(2.0 * i_p * q_p, i_p * i_p - q_p * q_p) / (4.0 * np.pi)
        
        # PLL loop filter
        carr_nco = (old_carr_nco +
                    tau2_carr / tau1_carr * (carr_error - old_carr_error) +
                    carr_error * (pdi_carr / tau1_carr))
        old_carr_nco = carr_nco
        old_carr_error = carr_error
        carr_freq_val = carr_freq_basis + carr_nco
        
        # DLL discriminator (normalized early-minus-late)
        e_mag = np.sqrt(i_e ** 2 + q_e ** 2)
        l_mag = np.sqrt(i_l ** 2 + q_l ** 2)
        
        if (e_mag + l_mag) > 0:
            code_error = (e_mag - l_mag) / (e_mag + l_mag)
        else:
            code_error = 0.0
        
        # DLL loop filter
        code_nco = (old_code_nco +
                    tau2_code / tau1_code * (code_error - old_code_error) +
                    code_error * (pdi_code / tau1_code))
        old_code_nco = code_nco
        old_code_error = code_error
        code_freq_nco = CA_CHIPPING_RATE - code_nco
        
        # Store results
        # Keep positions in complex I/Q samples (not bytes) so that code
        # pseudoranges use the correct time scale for int8/int16/float32 data.
        absolute_sample[loop_cnt] = fid.tell() / settings.bytes_per_iq_pair
        code_freq[loop_cnt] = code_freq_nco
        carr_freq[loop_cnt] = carr_freq_val
        I_E[loop_cnt] = i_e
        I_P[loop_cnt] = i_p
        I_L[loop_cnt] = i_l
        Q_E[loop_cnt] = q_e
        Q_P[loop_cnt] = q_p
        Q_L[loop_cnt] = q_l
        dll_discr[loop_cnt] = code_error
        dll_discr_filt[loop_cnt] = code_nco
        pll_discr[loop_cnt] = carr_error
        pll_discr_filt[loop_cnt] = carr_nco
        processed_ms = loop_cnt + 1

    if processed_ms == 0:
        return None

    valid = slice(0, processed_ms)
    return {
        "status": channel["status"],
        "absolute_sample": absolute_sample[valid],
        "code_freq": code_freq[valid],
        "carr_freq": carr_freq[valid],
        "I_P": I_P[valid], "I_E": I_E[valid], "I_L": I_L[valid],
        "Q_P": Q_P[valid], "Q_E": Q_E[valid], "Q_L": Q_L[valid],
        "dll_discr": dll_discr[valid],
        "dll_discr_filt": dll_discr_filt[valid],
        "pll_discr": pll_discr[valid],
        "pll_discr_filt": pll_discr_filt[valid],
        "PRN": prn,
        "processed_ms": processed_ms,
    }


def track_all_channels(
    channels: List[dict],
    settings: Settings,
    file_path: str,
) -> TrackingResult:
    """
    Run tracking for all channels.
    
    Parameters
    ----------
    channels : list
        Channel initialization data from acquisition.
    settings : Settings
        Receiver settings.
    file_path : str
        Path to the data file.
    
    Returns
    -------
    TrackingResult
        Tracking results for all channels.
    """
    result = TrackingResult(settings)
    result.channels = channels
    
    logger.info(
        f"Starting tracking: {len(channels)} channels, "
        f"{settings.ms_to_process} ms"
    )
    
    with open(file_path, "rb") as fid:
        for ch_nr, channel in enumerate(channels):
            track_data = track_channel(fid, channel, settings, ch_nr)
            if track_data is not None:
                result.results.append(track_data)
            else:
                logger.warning(f"Tracking failed for channel {ch_nr} (PRN {channel['prn']})")
    
    logger.info(
        f"Tracking completed. Successfully tracked "
        f"{len(result.results)}/{len(channels)} channels."
    )
    
    return result
