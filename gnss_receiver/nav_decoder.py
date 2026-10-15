"""GPS L1 C/A navigation-message synchronization and decoding.

Tracking produces one prompt correlator sample per millisecond.  GPS LNAV
symbols last 20 ms, and a subframe contains 300 symbols (6 seconds).  This
module searches all possible 1-ms symbol phases, verifies the repeated
subframe preambles, and decodes the broadcast ephemeris.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from .config import GPS_PI

logger = logging.getLogger("gnss_receiver.nav_decoder")

# TLM preamble 10001011, represented as bipolar values.
PREAMBLE_BITS = np.array([1, -1, -1, -1, 1, -1, 1, 1], dtype=np.int8)
NAV_SYMBOL_MS = 20
SUBFRAME_BITS = 300
SUBFRAME_MS = SUBFRAME_BITS * NAV_SYMBOL_MS
EPHEMERIS_BITS = 5 * SUBFRAME_BITS
GPS_WEEK_SECONDS = 604800.0


class NavigationDecodeError(ValueError):
    """Raised when a complete, consistent LNAV ephemeris cannot be decoded."""


def bin2dec(binary_str: str) -> int:
    """Convert a binary string to an unsigned integer."""
    return int(binary_str, 2)


def twos_comp2dec(binary_str: str) -> int:
    """Convert a two's-complement binary string to a signed integer."""
    val = int(binary_str, 2)
    if binary_str[0] == "1":
        val -= 2 ** len(binary_str)
    return val


def check_phase(word: list, d30star: str) -> list:
    """Remove the GPS LNAV data-bit inversion controlled by D30*.

    The six parity bits are not inverted.  ``word`` is a list of 30 binary
    characters and ``d30star`` is the final transmitted bit of the previous
    word.
    """
    if d30star == "1":
        for i in range(24):
            word[i] = "0" if word[i] == "1" else "1"
    return word


def nav_party_check(ndat: np.ndarray) -> int:
    """Check parity for one GPS LNAV word in bipolar form.

    ``ndat`` contains D29*, D30*, and the 30 bits of the current word,
    represented by +1/-1.  Returns zero when parity fails.
    """
    ndat = np.asarray(ndat).copy()
    if ndat.shape != (32,):
        raise ValueError("A GPS navigation word parity check needs 32 bits")

    # The ICD parity equations are expressed for data bits with the preceding
    # D30* correction removed.
    if ndat[1] != 1:
        ndat[2:26] *= -1

    parity = np.zeros(6, dtype=np.int8)
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

    if np.array_equal(parity, ndat[26:32]):
        return int(-ndat[1])
    return 0


def _symbol_candidates(i_p: np.ndarray, search_offset: int = 0) -> List[dict]:
    """Integrate prompt samples to 20-ms symbols and locate possible TLM words."""
    prompt = np.asarray(i_p, dtype=np.float64).reshape(-1)
    candidates: List[dict] = []

    # The symbol boundary can occur at any 1-ms tracking epoch.
    for phase_ms in range(NAV_SYMBOL_MS):
        usable = prompt[phase_ms:]
        symbol_count = len(usable) // NAV_SYMBOL_MS
        if symbol_count < 8:
            continue

        soft_symbols = usable[:symbol_count * NAV_SYMBOL_MS].reshape(
            symbol_count, NAV_SYMBOL_MS
        ).sum(axis=1)
        # Correct symbol boundaries maximize the magnitude of 20-ms prompt
        # integrations across the complete bit stream.
        phase_score = float(np.mean(np.abs(soft_symbols)))
        symbols = np.where(soft_symbols >= 0, 1, -1).astype(np.int8)
        correlations = np.correlate(symbols, PREAMBLE_BITS, mode="valid")
        indices = np.flatnonzero(np.abs(correlations) >= 6)

        for bit_index in indices:
            start_ms = phase_ms + int(bit_index) * NAV_SYMBOL_MS
            if start_ms >= search_offset:
                candidates.append({
                    "start_ms": start_ms,
                    "phase_ms": phase_ms,
                    "bit_index": int(bit_index),
                    "correlation": int(correlations[bit_index]),
                    "phase_score": phase_score,
                    "symbols": symbols,
                })

    # A candidate can occasionally be reported by neighbouring phases.  Keep
    # the strongest one within a 1-ms neighbourhood.
    candidates.sort(key=lambda item: (item["start_ms"], -item["phase_score"], -abs(item["correlation"])))
    deduplicated: List[dict] = []
    for candidate in candidates:
        if deduplicated and candidate["start_ms"] - deduplicated[-1]["start_ms"] < NAV_SYMBOL_MS:
            if (candidate["phase_score"], abs(candidate["correlation"])) > (
                deduplicated[-1]["phase_score"], abs(deduplicated[-1]["correlation"])
            ):
                deduplicated[-1] = candidate
        else:
            deduplicated.append(candidate)
    return deduplicated


def find_nav_preambles(track_data: dict, search_offset: int = 0) -> List[dict]:
    """Return TLM preamble candidates in a tracking result.

    Each candidate contains the prompt-bit phase and the index of the first
    symbol of the subframe.  A candidate is only a synchronization hint; use
    :func:`decode_ephemeris_from_tracking` to validate the following frames.
    """
    if "I_P" not in track_data:
        raise KeyError("Tracking data must contain the 'I_P' prompt correlator")
    return _symbol_candidates(track_data["I_P"], search_offset=search_offset)


def find_preambles(track_data: dict, search_offset: int = 0) -> Tuple[int, bool]:
    """Compatibility helper returning the first possible preamble position."""
    candidates = find_nav_preambles(track_data, search_offset=search_offset)
    if not candidates:
        return 0, False
    return int(candidates[0]["start_ms"]), True


def _decode_subframes(bits: List[str], d30star: str) -> Tuple[dict, float, List[int], List[int]]:
    """Decode five consecutive subframes and retain their IDs/HOW counters."""
    if len(bits) < EPHEMERIS_BITS:
        raise NavigationDecodeError("At least 1500 navigation bits are required")

    eph: Dict[str, float] = {}
    subframe_ids: List[int] = []
    z_counts: List[int] = []

    for frame_index in range(5):
        subframe = list(bits[SUBFRAME_BITS * frame_index:SUBFRAME_BITS * (frame_index + 1)])
        for word_index in range(10):
            word_start = 30 * word_index
            word = check_phase(subframe[word_start:word_start + 30], d30star)
            subframe[word_start:word_start + 30] = word
            d30star = word[-1]

        subframe_str = "".join(subframe)
        subframe_id = bin2dec(subframe_str[49:52])
        z_count = bin2dec(subframe_str[30:47])
        subframe_ids.append(subframe_id)
        z_counts.append(z_count)

        if subframe_id == 1:
            eph["weekNumber"] = bin2dec(subframe_str[60:70]) + 1024
            eph["accuracy"] = bin2dec(subframe_str[72:76])
            eph["health"] = bin2dec(subframe_str[76:82])
            eph["T_GD"] = twos_comp2dec(subframe_str[195:204]) * 2 ** (-31)
            eph["IODC"] = bin2dec(subframe_str[82:84] + subframe_str[196:204])
            eph["t_oc"] = bin2dec(subframe_str[218:234]) * 2 ** 4
            eph["a_f2"] = twos_comp2dec(subframe_str[240:248]) * 2 ** (-55)
            eph["a_f1"] = twos_comp2dec(subframe_str[248:264]) * 2 ** (-43)
            eph["a_f0"] = twos_comp2dec(subframe_str[270:292]) * 2 ** (-31)

        elif subframe_id == 2:
            eph["IODE_sf2"] = bin2dec(subframe_str[60:68])
            eph["C_rs"] = twos_comp2dec(subframe_str[68:84]) * 2 ** (-5)
            eph["deltan"] = twos_comp2dec(subframe_str[90:106]) * 2 ** (-43) * GPS_PI
            eph["M_0"] = twos_comp2dec(
                subframe_str[106:114] + subframe_str[120:144]
            ) * 2 ** (-31) * GPS_PI
            eph["C_uc"] = twos_comp2dec(subframe_str[150:166]) * 2 ** (-29)
            eph["e"] = bin2dec(subframe_str[166:174] + subframe_str[180:204]) * 2 ** (-33)
            eph["C_us"] = twos_comp2dec(subframe_str[210:226]) * 2 ** (-29)
            eph["sqrtA"] = bin2dec(
                subframe_str[226:234] + subframe_str[240:264]
            ) * 2 ** (-19)
            eph["t_oe"] = bin2dec(subframe_str[270:286]) * 2 ** 4

        elif subframe_id == 3:
            eph["C_ic"] = twos_comp2dec(subframe_str[60:76]) * 2 ** (-29)
            eph["omega_0"] = twos_comp2dec(
                subframe_str[76:84] + subframe_str[90:114]
            ) * 2 ** (-31) * GPS_PI
            eph["C_is"] = twos_comp2dec(subframe_str[120:136]) * 2 ** (-29)
            eph["i_0"] = twos_comp2dec(
                subframe_str[136:144] + subframe_str[150:174]
            ) * 2 ** (-31) * GPS_PI
            eph["C_rc"] = twos_comp2dec(subframe_str[180:196]) * 2 ** (-5)
            eph["omega"] = twos_comp2dec(
                subframe_str[196:204] + subframe_str[210:234]
            ) * 2 ** (-31) * GPS_PI
            eph["omegaDot"] = twos_comp2dec(
                subframe_str[240:264]
            ) * 2 ** (-43) * GPS_PI
            eph["IODE_sf3"] = bin2dec(subframe_str[270:278])
            eph["iDot"] = twos_comp2dec(subframe_str[278:292]) * 2 ** (-43) * GPS_PI

    # The HOW Z-count is the start time of the following subframe.  Subtract
    # the five 6-second intervals to obtain the start time of this data block.
    tow = (z_counts[-1] * 6 - 30) % GPS_WEEK_SECONDS
    return eph, tow, subframe_ids, z_counts


def decode_ephemeris(bits: list, d30star: str) -> Tuple[dict, float]:
    """Decode broadcast ephemeris and TOW from 1500 consecutive LNAV bits."""
    eph, tow, _subframe_ids, _z_counts = _decode_subframes(list(bits), d30star)
    return eph, tow


def _has_five_subframe_preambles(symbols: np.ndarray, start: int) -> bool:
    """Require the TLM preamble at all five 6-second subframe boundaries."""
    if start < 0 or start + EPHEMERIS_BITS > len(symbols):
        return False
    for frame_index in range(5):
        frame_start = start + frame_index * SUBFRAME_BITS
        observed = symbols[frame_start:frame_start + len(PREAMBLE_BITS)]
        if len(observed) != len(PREAMBLE_BITS):
            return False
        if abs(int(np.dot(observed, PREAMBLE_BITS))) < 6:
            return False
    return True


def _ephemeris_is_complete_and_consistent(
    eph: dict, subframe_ids: List[int], z_counts: List[int]
) -> bool:
    required = {
        "weekNumber", "health", "T_GD", "IODC", "t_oc", "a_f2", "a_f1", "a_f0",
        "IODE_sf2", "C_rs", "deltan", "M_0", "C_uc", "e", "C_us", "sqrtA",
        "t_oe", "C_ic", "omega_0", "C_is", "i_0", "C_rc", "omega", "omegaDot",
        "IODE_sf3", "iDot",
    }
    if not required.issubset(eph):
        return False

    expected_ids = [((subframe_ids[0] - 1 + i) % 5) + 1 for i in range(5)]
    if subframe_ids != expected_ids or set(subframe_ids) != {1, 2, 3, 4, 5}:
        return False
    if any((z_counts[i + 1] - z_counts[i]) % 100800 != 1 for i in range(4)):
        return False
    if eph["IODE_sf2"] != eph["IODE_sf3"]:
        return False
    if (eph["IODC"] & 0xFF) != eph["IODE_sf2"]:
        return False
    if not (0.0 <= eph["e"] < 1.0 and 4000.0 < eph["sqrtA"] < 7000.0):
        return False
    if not (0 <= eph["t_oe"] < GPS_WEEK_SECONDS and 0 <= eph["t_oc"] < GPS_WEEK_SECONDS):
        return False
    return True


def decode_ephemeris_from_tracking(track_data: dict, prn: int = None) -> dict:
    """Synchronize and decode one satellite's broadcast ephemeris.

    Args:
        track_data: Tracking result with ``I_P`` and, optionally, ``PRN``.
        prn: Satellite PRN, used for diagnostics.

    Returns:
        A dictionary containing the ephemeris, TOW of the first decoded
        subframe, its tracking index, and the detected symbol phase.

    Raises:
        NavigationDecodeError: if no complete and consistent 30-second
            ephemeris block is present.
    """
    if prn is None:
        prn = int(track_data.get("PRN", 0))

    candidates = find_nav_preambles(track_data)
    for candidate in candidates:
        symbols = candidate["symbols"]
        bit_index = candidate["bit_index"]
        if not _has_five_subframe_preambles(symbols, bit_index):
            continue

        raw_symbols = symbols[bit_index:bit_index + EPHEMERIS_BITS]
        previous_symbol = symbols[bit_index - 1] if bit_index > 0 else None

        # A Costas-loop receiver has a 180-degree polarity ambiguity.  Try both
        # signs; the D30* correction and frame checks disambiguate it.
        for polarity in (1, -1):
            signed_symbols = raw_symbols * polarity
            if previous_symbol is None:
                previous_options = ("0", "1")
            else:
                previous_options = ("1" if previous_symbol * polarity > 0 else "0",)

            nav_bits = ["1" if value > 0 else "0" for value in signed_symbols]
            for d30star in previous_options:
                try:
                    eph, tow, subframe_ids, z_counts = _decode_subframes(nav_bits, d30star)
                except (ValueError, IndexError, TypeError):
                    continue
                if not _ephemeris_is_complete_and_consistent(eph, subframe_ids, z_counts):
                    continue

                return {
                    "prn": int(prn),
                    "ephemeris": eph,
                    "tow_seconds": float(tow),
                    "subframe_start_ms": int(candidate["start_ms"]),
                    "symbol_phase_ms": int(candidate["phase_ms"]),
                    "subframe_ids": subframe_ids,
                    "health": int(eph["health"]),
                }

    duration_ms = len(np.asarray(track_data.get("I_P", [])))
    raise NavigationDecodeError(
        f"PRN {prn}: no complete, consistent 30-second GPS ephemeris found "
        f"in {duration_ms} ms of tracking data"
    )
