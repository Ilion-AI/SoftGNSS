"""GPS broadcast-navigation decoding and position/velocity/time solution.

The module consumes millisecond prompt-correlator outputs from ``tracking``.
It decodes a 30-second GPS LNAV ephemeris block, forms code pseudoranges at
one or more epochs, computes GPS satellite ECEF positions, and solves the
receiver's ECEF position and clock bias by least squares.
"""

import logging
from types import SimpleNamespace
from typing import List, Sequence, Tuple

import numpy as np

from geoFunctions import e_r_corr, satpos, togeod, topocent, tropo

from .config import C, Settings
from .nav_decoder import GPS_WEEK_SECONDS, NavigationDecodeError, decode_ephemeris_from_tracking

logger = logging.getLogger("gnss_receiver.pvt")


class NavigationSolutionError(ValueError):
    """Raised when there is not enough valid data for a position solution."""


def ecef_to_geodetic(position_ecef_m: Sequence[float]) -> Tuple[float, float, float]:
    """Convert WGS-84 ECEF coordinates to decimal-degree geodetic coordinates."""
    position = np.asarray(position_ecef_m, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("ECEF position must contain three finite coordinates")

    latitude_deg, longitude_deg, height_m = togeod(
        6378137.0, 298.257223563, position[0], position[1], position[2]
    )
    longitude_deg = (float(longitude_deg) + 180.0) % 360.0 - 180.0
    return float(latitude_deg), longitude_deg, float(height_m)


def _tropospheric_delay(receiver_ecef: np.ndarray, satellite_ecef: np.ndarray) -> Tuple[float, float, float]:
    """Return standard-atmosphere tropospheric delay, azimuth and elevation."""
    azimuth_deg, elevation_deg, _distance = topocent(
        receiver_ecef, satellite_ecef - receiver_ecef
    )
    if elevation_deg <= 0.0:
        return 0.0, float(azimuth_deg), float(elevation_deg)

    _lat, _lon, height_m = ecef_to_geodetic(receiver_ecef)
    delay_m = tropo(
        np.sin(np.deg2rad(elevation_deg)),
        max(float(height_m) / 1000.0, 0.0),
        1013.0,
        293.0,
        50.0,
        0.0,
        0.0,
        0.0,
    )
    return float(delay_m), float(azimuth_deg), float(elevation_deg)


def least_squares_position(
    satellite_positions_ecef_m: Sequence[Sequence[float]],
    pseudoranges_m: Sequence[float],
    use_trop_corr: bool = True,
    initial_position_ecef_m: Sequence[float] = None,
    max_iterations: int = 12,
) -> dict:
    """Solve receiver ECEF position and clock bias from pseudoranges.

    Satellite coordinates are rotated for Earth rotation during signal travel.
    At least four satellites with non-degenerate geometry are required.
    ``pseudoranges_m`` should already include the satellite clock correction.
    """
    satellite_positions = np.asarray(satellite_positions_ecef_m, dtype=np.float64)
    observations = np.asarray(pseudoranges_m, dtype=np.float64).reshape(-1)
    if satellite_positions.ndim != 2 or satellite_positions.shape[1] != 3:
        raise ValueError("Satellite ECEF positions must have shape (n, 3)")
    if len(satellite_positions) != len(observations) or len(observations) < 4:
        raise NavigationSolutionError("At least four matching satellite observations are required")
    if not np.all(np.isfinite(satellite_positions)) or not np.all(np.isfinite(observations)):
        raise ValueError("Satellite positions and pseudoranges must be finite")

    receiver = (
        np.zeros(3, dtype=np.float64)
        if initial_position_ecef_m is None
        else np.asarray(initial_position_ecef_m, dtype=np.float64).copy()
    )
    if receiver.shape != (3,):
        raise ValueError("Initial receiver position must contain three coordinates")
    receiver_clock_bias_m = 0.0

    azimuths = np.zeros(len(observations), dtype=np.float64)
    elevations = np.zeros(len(observations), dtype=np.float64)
    geometric_ranges = np.zeros(len(observations), dtype=np.float64)

    for _iteration in range(max_iterations):
        design = np.zeros((len(observations), 4), dtype=np.float64)
        residuals = np.zeros(len(observations), dtype=np.float64)

        for index, satellite in enumerate(satellite_positions):
            initial_range = max(float(np.linalg.norm(satellite - receiver)), 1.0)
            rotated = e_r_corr(initial_range / C, satellite)
            difference = rotated - receiver
            distance = float(np.linalg.norm(difference))
            if not np.isfinite(distance) or distance <= 0.0:
                raise NavigationSolutionError("Invalid satellite-to-receiver range")

            correction = 0.0
            if use_trop_corr and np.linalg.norm(receiver) > 1_000_000.0:
                correction, azimuths[index], elevations[index] = _tropospheric_delay(
                    receiver, rotated
                )
            else:
                azimuths[index] = np.nan
                elevations[index] = np.nan

            geometric_ranges[index] = distance
            design[index, :3] = (receiver - rotated) / distance
            design[index, 3] = 1.0
            residuals[index] = observations[index] - (
                distance + receiver_clock_bias_m + correction
            )

        if np.linalg.matrix_rank(design) < 4:
            raise NavigationSolutionError("Satellite geometry is insufficient for a 3-D fix")
        update, _residuals, _rank, _singular = np.linalg.lstsq(
            design, residuals, rcond=None
        )
        if not np.all(np.isfinite(update)):
            raise NavigationSolutionError("Least-squares position update is not finite")

        receiver += update[:3]
        receiver_clock_bias_m += float(update[3])
        if np.linalg.norm(update[:3]) < 1e-4 and abs(update[3]) < 1e-4:
            break

    # Re-evaluate the final geometry and corrections at the converged position.
    design = np.zeros((len(observations), 4), dtype=np.float64)
    for index, satellite in enumerate(satellite_positions):
        travel_time = float(np.linalg.norm(satellite - receiver)) / C
        rotated = e_r_corr(travel_time, satellite)
        difference = rotated - receiver
        distance = float(np.linalg.norm(difference))
        geometric_ranges[index] = distance
        design[index, :3] = (receiver - rotated) / distance
        design[index, 3] = 1.0
        if use_trop_corr:
            _delay, azimuths[index], elevations[index] = _tropospheric_delay(receiver, rotated)
        else:
            azimuths[index], elevations[index] = topocent(receiver, difference)[:2]

    if np.linalg.matrix_rank(design) < 4:
        raise NavigationSolutionError("Satellite geometry is insufficient for a 3-D fix")
    covariance = np.linalg.pinv(design.T @ design)
    latitude_deg, longitude_deg, height_m = ecef_to_geodetic(receiver)

    lat = np.deg2rad(latitude_deg)
    lon = np.deg2rad(longitude_deg)
    ecef_to_enu = np.array([
        [-np.sin(lon), np.cos(lon), 0.0],
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
    ])
    covariance_enu = ecef_to_enu @ covariance[:3, :3] @ ecef_to_enu.T
    pdop = float(np.sqrt(max(np.trace(covariance[:3, :3]), 0.0)))
    hdop = float(np.sqrt(max(covariance_enu[0, 0] + covariance_enu[1, 1], 0.0)))
    vdop = float(np.sqrt(max(covariance_enu[2, 2], 0.0)))
    tdop = float(np.sqrt(max(covariance[3, 3], 0.0)))

    return {
        "ecef_m": [float(value) for value in receiver],
        "latitude_deg": latitude_deg,
        "longitude_deg": longitude_deg,
        "height_m": height_m,
        "receiver_clock_bias_m": float(receiver_clock_bias_m),
        "dop": {
            "gdop": float(np.sqrt(pdop * pdop + tdop * tdop)),
            "pdop": pdop,
            "hdop": hdop,
            "vdop": vdop,
            "tdop": tdop,
        },
        "azimuths_deg": [float(value) for value in azimuths],
        "elevations_deg": [float(value) for value in elevations],
        "geometric_ranges_m": [float(value) for value in geometric_ranges],
    }


def _satellite_ephemeris_table(decoded: Sequence[dict]) -> np.ndarray:
    """Create the 1-based-PRN-indexed object table expected by ``satpos``."""
    table = np.empty(32, dtype=object)
    table[:] = None
    for item in decoded:
        table[int(item["prn"]) - 1] = SimpleNamespace(**item["ephemeris"])
    return table


def _pseudorange_epoch(
    navigation_data: Sequence[dict],
    epoch_offset_ms: int,
    settings: Settings,
) -> Tuple[List[dict], np.ndarray, np.ndarray, np.ndarray]:
    """Build code pseudoranges and satellite states for one tracking epoch."""
    usable = []
    transmit_tow = []
    sample_times = []

    for item in navigation_data:
        track_data = item["track_data"]
        sample_index = int(item["subframe_start_ms"]) + int(epoch_offset_ms)
        prompt = np.asarray(track_data["I_P"])
        samples = np.asarray(track_data["absolute_sample"])
        if sample_index >= len(prompt) or sample_index >= len(samples):
            continue
        sample_number = float(samples[sample_index])
        if not np.isfinite(sample_number) or sample_number <= 0.0:
            continue

        tx_tow = (float(item["tow_seconds"]) + epoch_offset_ms / 1000.0) % GPS_WEEK_SECONDS
        usable.append(item)
        transmit_tow.append(tx_tow)
        sample_times.append(sample_number / settings.fs)

    if len(usable) < 4:
        raise NavigationSolutionError(
            f"Only {len(usable)} satellites have measurements at this epoch; four are required"
        )

    transmit_tow = np.asarray(transmit_tow, dtype=np.float64)
    sample_times = np.asarray(sample_times, dtype=np.float64)
    reference_tow = transmit_tow[0]
    week_differences = (transmit_tow - reference_tow + GPS_WEEK_SECONDS / 2.0) % GPS_WEEK_SECONDS
    week_differences -= GPS_WEEK_SECONDS / 2.0
    unwrapped_tow = reference_tow + week_differences

    # The input capture has no absolute GPS timestamp.  Removing a common
    # origin leaves the measured inter-satellite code delays; the arbitrary
    # 68.802-ms seed is absorbed by the receiver clock state in the LS solve.
    raw_delays_s = sample_times - unwrapped_tow
    pseudoranges_m = (
        raw_delays_s - float(np.min(raw_delays_s)) + float(settings.start_offset) / 1000.0
    ) * C

    eph_table = _satellite_ephemeris_table(usable)
    satellite_positions = []
    satellite_clock_corrections = []
    for item, tx_tow in zip(usable, transmit_tow):
        prn = int(item["prn"])
        positions, clock_corrections = satpos(
            float(tx_tow), np.asarray([prn], dtype=np.int64), eph_table, settings
        )
        satellite_positions.append(positions[:, 0])
        satellite_clock_corrections.append(float(clock_corrections[0]))

    satellite_positions = np.asarray(satellite_positions, dtype=np.float64)
    satellite_clock_corrections = np.asarray(satellite_clock_corrections, dtype=np.float64)
    corrected_pseudoranges = pseudoranges_m + satellite_clock_corrections * C
    return usable, satellite_positions, corrected_pseudoranges, pseudoranges_m


def _solve_epoch(
    satellite_positions: np.ndarray,
    corrected_pseudoranges: np.ndarray,
    pseudoranges: np.ndarray,
    satellites: Sequence[dict],
    settings: Settings,
) -> dict:
    """Apply the elevation mask and return one JSON-ready PVT fix."""
    all_indices = np.arange(len(satellites))
    active_indices = all_indices
    solution = None

    # First obtain a coarse fix with all decoded satellites, then reject those
    # below the configured elevation mask and solve again.
    for _ in range(3):
        if len(active_indices) < 4:
            raise NavigationSolutionError("Fewer than four satellites remain above the elevation mask")
        solution = least_squares_position(
            satellite_positions[active_indices],
            corrected_pseudoranges[active_indices],
            use_trop_corr=bool(settings.use_trop_corr),
        )
        visible = np.flatnonzero(
            np.asarray(solution["elevations_deg"]) >= float(settings.elevation_mask)
        )
        next_indices = active_indices[visible]
        if len(next_indices) < 4:
            raise NavigationSolutionError(
                "Fewer than four satellites are above the configured elevation mask"
            )
        if np.array_equal(next_indices, active_indices):
            break
        active_indices = next_indices

    solution = least_squares_position(
        satellite_positions[active_indices],
        corrected_pseudoranges[active_indices],
        use_trop_corr=bool(settings.use_trop_corr),
    )
    radius = float(np.linalg.norm(solution["ecef_m"]))
    if not (5_000_000.0 < radius < 8_000_000.0):
        raise NavigationSolutionError(
            f"Position solution is not Earth-like (ECEF radius {radius:.0f} m); "
            "check the data format, sampling rate, and signal tracking"
        )

    used = []
    for local_index, source_index in enumerate(active_indices):
        nav = satellites[int(source_index)]
        used.append({
            "prn": int(nav["prn"]),
            "pseudorange_m": float(pseudoranges[int(source_index)]),
            "azimuth_deg": float(solution["azimuths_deg"][local_index]),
            "elevation_deg": float(solution["elevations_deg"][local_index]),
        })

    solution["satellites"] = used
    solution["satellite_count"] = len(used)
    return solution


def _unavailable_result(message: str, decoded: Sequence[dict] = ()) -> dict:
    return {
        "status": "unavailable",
        "message": str(message),
        "valid_ephemerides": [
            {
                "prn": int(item["prn"]),
                "tow_seconds": float(item["tow_seconds"]),
                "health": int(item["health"]),
            }
            for item in decoded
        ],
        "solutions": [],
    }


def solve_navigation(track_result, settings: Settings) -> dict:
    """Decode ephemerides and compute receiver coordinates from tracking data.

    A complete GPS LNAV ephemeris consists of five consecutive 6-second
    subframes.  At least four healthy satellites with valid broadcast
    ephemerides are required for a 3-D position solution.
    """
    decoded: List[dict] = []
    for track_data in getattr(track_result, "results", []):
        prn = int(track_data.get("PRN", 0))
        if not 1 <= prn <= 32:
            logger.warning("Skipping tracking channel with invalid PRN %s", prn)
            continue
        try:
            nav = decode_ephemeris_from_tracking(track_data, prn=prn)
        except NavigationDecodeError as error:
            logger.info("%s", error)
            continue
        if int(nav["health"]) != 0:
            logger.warning("Skipping unhealthy GPS satellite PRN %d (health=%d)", prn, nav["health"])
            continue
        nav["track_data"] = track_data
        decoded.append(nav)

    decoded.sort(key=lambda item: item["prn"])
    if len(decoded) < 4:
        return _unavailable_result(
            f"Decoded healthy ephemerides from {len(decoded)} satellites; at least four are "
            "needed for coordinates. A continuous recording of about 30 seconds after a "
            "valid subframe preamble is required (typically capture 36 seconds or more).",
            decoded,
        )

    max_offset_ms = min(
        min(
            len(np.asarray(item["track_data"]["I_P"])),
            len(np.asarray(item["track_data"]["absolute_sample"])),
        ) - int(item["subframe_start_ms"]) - 1
        for item in decoded
    )
    if max_offset_ms < 0:
        return _unavailable_result("Tracking data ends before the first valid navigation epoch", decoded)

    period_ms = max(1, int(round(float(settings.nav_sol_period))))
    offsets = list(range(0, max_offset_ms + 1, period_ms))
    if not offsets:
        offsets = [0]

    solutions = []
    failures = []
    for epoch_index, epoch_offset_ms in enumerate(offsets):
        try:
            satellites, positions, corrected_ranges, raw_ranges = _pseudorange_epoch(
                decoded, epoch_offset_ms, settings
            )
            solution = _solve_epoch(
                positions, corrected_ranges, raw_ranges, satellites, settings
            )
            solution.update({
                "epoch": int(epoch_index),
                "tracking_offset_ms": int(epoch_offset_ms),
                "gps_tow_seconds": float(
                    (satellites[0]["tow_seconds"] + epoch_offset_ms / 1000.0)
                    % GPS_WEEK_SECONDS
                ),
            })
            solutions.append(solution)
        except (NavigationSolutionError, ValueError, np.linalg.LinAlgError) as error:
            failures.append(f"epoch {epoch_offset_ms} ms: {error}")
            logger.warning("Navigation solution failed at %d ms: %s", epoch_offset_ms, error)

    if not solutions:
        detail = failures[0] if failures else "no valid navigation epochs were found"
        return _unavailable_result(
            f"Ephemerides were decoded for {len(decoded)} satellites, but no position "
            f"solution could be computed ({detail}). Check signal format, sampling rate, "
            "and tracking quality.",
            decoded,
        )

    return {
        "status": "solved",
        "message": f"Computed {len(solutions)} position epoch(s) using {len(decoded)} healthy satellite ephemerides.",
        "valid_ephemerides": [
            {
                "prn": int(item["prn"]),
                "tow_seconds": float(item["tow_seconds"]),
                "health": int(item["health"]),
                "week_number": int(item["ephemeris"]["weekNumber"]),
            }
            for item in decoded
        ],
        "solutions": solutions,
    }
