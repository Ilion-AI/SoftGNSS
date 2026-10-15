"""
Navigation message decoder for GPS L1 C/A.

Decodes subframes from the tracking output, extracts ephemeris data
and Time of Week (TOW) from the GPS navigation message.
"""

import logging
from typing import Tuple, Optional, List

import numpy as np

logger = logging.getLogger("gnss_receiver.nav_decoder")

# Preamble pattern (TLM word)
PREAMBLE_BITS = np.array([1, -1, -1, -1, 1, -1, 1, 1])


def bin2dec(binary_str: str) -> int:
    """Convert binary string to decimal integer."""
    return int(binary_str, 2)


def twos_comp2dec(binary_str: str) -> int:
    """Convert two's complement binary string to signed integer."""
    val = int(binary_str, 2)
    if binary_str[0] == '1':
        val -= 2 ** len(binary_str)
    return val


def check_phase(word: list, d30star: str) -> list:
    """
    Check and correct data bit polarity using D30* from previous word.
    
    Parameters
    ----------
    word : list
        List of 30 bit characters ('0' or '1').
    d30star : str
        Last bit of previous word ('0' or '1').
    
    Returns
    -------
    list
        Word with corrected polarity.
    """
    if d30star == '1':
        for i in range(24):
            word[i] = '0' if word[i] == '1' else '1'
    return word


def nav_party_check(ndat: np.ndarray) -> int:
    """
    Compute and check parity of a GPS navigation word.
    
    Based on Figure 2-10 in the GPS SPS Signal Spec.
    
    Parameters
    ----------
    ndat : np.ndarray
        Array of 32 values (+1/-1) representing 30 bits plus
        two previous parity bits.
    
    Returns
    -------
    int
        +1 if parity OK (normal polarity), -1 if OK (inverted),
        0 if parity fails.
    """
    # Invert data bits if D30* = -1
    if ndat[1] != 1:
        ndat = ndat.copy()
        ndat[2:26] *= -1
    
    # Calculate 6 parity bits
    parity = np.zeros(6)
    parity[0] = ndat[0] * ndat[2] * ndat[3] * ndat[4] * ndat[6] * \
                ndat[7] * ndat[11] * ndat[12] * ndat[13] * ndat[14] * \
                ndat[15] * ndat[18] * ndat[19] * ndat[21] * ndat[24]
    
    parity[1] = ndat[1] * ndat[3] * ndat[4] * ndat[5] * ndat[7] * \
                ndat[8] * ndat[12] * ndat[13] * ndat[14] * ndat[15] * \
                ndat[16] * ndat[19] * ndat[20] * ndat[22] * ndat[25]
    
    parity[2] = ndat[0] * ndat[2] * ndat[4] * ndat[5] * ndat[6] * \
                ndat[8] * ndat[9] * ndat[13] * ndat[14] * ndat[15] * \
                ndat[16] * ndat[17] * ndat[20] * ndat[21] * ndat[23]
    
    parity[3] = ndat[1] * ndat[3] * ndat[5] * ndat[6] * ndat[7] * \
                ndat[9] * ndat[10] * ndat[14] * ndat[15] * ndat[16] * \
                ndat[17] * ndat[18] * ndat[21] * ndat[22] * ndat[24]
    
    parity[4] = ndat[1] * ndat[2] * ndat[4] * ndat[6] * ndat[7] * \
                ndat[8] * ndat[10] * ndat[11] * ndat[15] * ndat[16] * \
                ndat[17] * ndat[18] * ndat[19] * ndat[22] * ndat[23] * \
                ndat[25]
    
    parity[5] = ndat[0] * ndat[4] * ndat[6] * ndat[7] * ndat[9] * \
                ndat[10] * ndat[11] * ndat[12] * ndat[14] * ndat[16] * \
                ndat[20] * ndat[23] * ndat[24] * ndat[25]
    
    # Check if received parity matches computed
    if np.sum(parity == ndat[26:32]) == 6:
        status = -1 * ndat[1]
    else:
        status = 0
    
    return int(status)


def find_preambles(track_data: dict, search_offset: int = 0) -> Tuple[int, bool]:
    """
    Find the first valid preamble in tracking output.
    
    Parameters
    ----------
    track_data : dict
        Tracking results for one channel (must have 'I_P' key).
    search_offset : int
        Starting sample offset for preamble search.
    
    Returns
    -------
    tuple
        (first_subframe_ms, found)
        first_subframe_ms: ms index of first preamble, or 0 if not found.
        found: True if valid preamble was found.
    """
    i_p = track_data["I_P"]
    
    # Convert to bipolar (+1/-1)
    bits = i_p[search_offset:].copy()
    bits[bits > 0] = 1
    bits[bits <= 0] = -1
    
    # Upsample preamble to 20 samples per bit
    preamble_ms = np.kron(PREAMBLE_BITS, np.ones(20))
    
    # Cross-correlate
    padded_preamble = np.pad(preamble_ms, (0, max(0, len(bits) - len(preamble_ms))), 'constant')
    xcorr = np.correlate(bits, padded_preamble, mode='full')
    
    # Find high correlation peaks
    xcorr_mid = (len(xcorr) + 1) // 2
    threshold = 153
    indices = np.where(np.abs(xcorr[xcorr_mid:]) > threshold)[0] + search_offset
    
    # Check for 6-second spacing (6000 ms)
    for i, idx in enumerate(indices):
        for j in range(i + 1, len(indices)):
            if indices[j] - idx == 6000:
                # Verify parity of first two words
                start = idx - 40
                end = start + 20 * 62  # 62 bits = 2 words + 2 extra
                
                if end > len(i_p):
                    break
                
                word_bits = i_p[start:end].copy()
                word_bits = word_bits.reshape(20, -1, order='F').sum(axis=0)
                word_bits[word_bits > 0] = 1
                word_bits[word_bits <= 0] = -1
                
                if (nav_party_check(word_bits[:32]) != 0 and
                    nav_party_check(word_bits[30:62]) != 0):
                    return int(idx), True
    
    return 0, False


def decode_ephemeris(bits: list, d30star: str) -> Tuple[dict, float]:
    """
    Decode ephemeris data and TOW from navigation message bits.
    
    Parameters
    ----------
    bits : list
        List of 1500 bit characters ('0' or '1').
    d30star : str
        Last bit of the previous word.
    
    Returns
    -------
    tuple
        (ephemeris_dict, TOW_seconds)
    """
    if len(bits) < 1500:
        raise ValueError("Need at least 1500 bits for ephemeris decoding")
    
    gps_pi = 3.1415926535898
    
    eph = {}
    
    for i in range(5):
        subframe = bits[300 * i:300 * (i + 1)]
        
        for j in range(10):
            subframe[30 * j:30 * (j + 1)] = check_phase(
                subframe[30 * j:30 * (j + 1)], d30star
            )
            d30star = subframe[30 * (j + 1) - 1]
        
        subframe_str = ''.join(subframe)
        subframe_id = bin2dec(subframe_str[49:52])
        
        if subframe_id == 1:
            eph["weekNumber"] = bin2dec(subframe_str[60:70]) + 1024
            eph["accuracy"] = bin2dec(subframe_str[72:76])
            eph["health"] = bin2dec(subframe_str[76:82])
            eph["T_GD"] = twos_comp2dec(subframe_str[195:204]) * 2**(-31)
            eph["IODC"] = bin2dec(subframe_str[82:84] + subframe_str[196:204])
            eph["t_oc"] = bin2dec(subframe_str[218:234]) * 2**4
            eph["a_f2"] = twos_comp2dec(subframe_str[240:248]) * 2**(-55)
            eph["a_f1"] = twos_comp2dec(subframe_str[248:264]) * 2**(-43)
            eph["a_f0"] = twos_comp2dec(subframe_str[270:292]) * 2**(-31)
        
        elif subframe_id == 2:
            eph["IODE_sf2"] = bin2dec(subframe_str[60:68])
            eph["C_rs"] = twos_comp2dec(subframe_str[68:84]) * 2**(-5)
            eph["deltan"] = twos_comp2dec(subframe_str[90:106]) * 2**(-43) * gps_pi
            eph["M_0"] = twos_comp2dec(subframe_str[106:114] + subframe_str[120:144]) * 2**(-31) * gps_pi
            eph["C_uc"] = twos_comp2dec(subframe_str[150:166]) * 2**(-29)
            eph["e"] = bin2dec(subframe_str[166:174] + subframe_str[180:204]) * 2**(-33)
            eph["C_us"] = twos_comp2dec(subframe_str[210:226]) * 2**(-29)
            eph["sqrtA"] = bin2dec(subframe_str[226:234] + subframe_str[240:264]) * 2**(-19)
            eph["t_oe"] = bin2dec(subframe_str[270:286]) * 2**4
        
        elif subframe_id == 3:
            eph["C_ic"] = twos_comp2dec(subframe_str[60:76]) * 2**(-29)
            eph["omega_0"] = twos_comp2dec(subframe_str[76:84] + subframe_str[90:114]) * 2**(-31) * gps_pi
            eph["C_is"] = twos_comp2dec(subframe_str[120:136]) * 2**(-29)
            eph["i_0"] = twos_comp2dec(subframe_str[136:144] + subframe_str[150:174]) * 2**(-31) * gps_pi
            eph["C_rc"] = twos_comp2dec(subframe_str[180:196]) * 2**(-5)
            eph["omega"] = twos_comp2dec(subframe_str[196:204] + subframe_str[210:234]) * 2**(-31) * gps_pi
            eph["omegaDot"] = twos_comp2dec(subframe_str[240:264]) * 2**(-43) * gps_pi
            eph["IODE_sf3"] = bin2dec(subframe_str[270:278])
            eph["iDot"] = twos_comp2dec(subframe_str[278:292]) * 2**(-43) * gps_pi
    
    # Extract TOW
    tow = bin2dec(subframe_str[30:47]) * 6 - 30
    
    return eph, tow
