import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from geoFunctions import e_r_corr
from gnss_receiver.config import C, Settings, detect_data_format
from gnss_receiver.nav_decoder import (
    decode_ephemeris_from_tracking,
    find_nav_preambles,
    nav_party_check,
)
from gnss_receiver.pvt import ecef_to_geodetic, least_squares_position, solve_navigation


class NavigationTests(unittest.TestCase):
    def test_decode_complete_synthetic_ephemeris(self):
        bits = []
        sqrt_a_raw = round(5153.7954775 * (2 ** 19))
        eccentricity_raw = round(0.01 * (2 ** 33))

        def put_field(frame, start, end, value):
            frame[start:end] = list(format(value, f"0{end - start}b"))

        for frame_index, subframe_id in enumerate((1, 2, 3, 4, 5)):
            frame = list("0" * 300)
            frame[:8] = list("10001011")
            z_count = ((1002 + 6 * (frame_index + 1)) // 6) % 100800
            put_field(frame, 30, 47, z_count)
            put_field(frame, 49, 52, subframe_id)

            # A zero D30 in every word keeps the synthetic data bits un-inverted.
            for word_index in range(10):
                frame[word_index * 30 + 29] = "0"

            if subframe_id == 1:
                put_field(frame, 60, 70, 50)
                put_field(frame, 76, 82, 0)
                put_field(frame, 196, 204, 42)
                put_field(frame, 218, 234, 345600 // 16)
            elif subframe_id == 2:
                put_field(frame, 60, 68, 42)
                put_field(frame, 166, 174, eccentricity_raw >> 24)
                put_field(frame, 180, 204, eccentricity_raw & ((1 << 24) - 1))
                put_field(frame, 226, 234, sqrt_a_raw >> 24)
                put_field(frame, 240, 264, sqrt_a_raw & ((1 << 24) - 1))
                put_field(frame, 270, 286, 345600 // 16)
            elif subframe_id == 3:
                put_field(frame, 270, 278, 42)

            bits.extend(frame)

        nav_signs = np.where(np.asarray(bits) == "1", 1.0, -1.0)
        prompt = np.repeat(nav_signs, 20)
        decoded = decode_ephemeris_from_tracking({"I_P": prompt, "PRN": 4}, prn=4)

        self.assertEqual(decoded["prn"], 4)
        self.assertEqual(decoded["subframe_ids"], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(decoded["tow_seconds"], 1002.0)
        self.assertAlmostEqual(decoded["ephemeris"]["sqrtA"], 5153.7954775, places=4)
        self.assertAlmostEqual(decoded["ephemeris"]["e"], 0.01, places=8)

    def test_decode_with_d30star_transitions_noisy_preambles_and_offset(self):
        frames = []
        sqrt_a_raw = round(5153.7954775 * (2 ** 19))
        eccentricity_raw = round(0.01 * (2 ** 33))

        def put_field(frame, start, end, value):
            frame[start:end] = list(format(value, f"0{end - start}b"))

        for frame_index, subframe_id in enumerate((1, 2, 3, 4, 5)):
            frame = list("0" * 300)
            frame[:8] = list("10001011")
            z_count = ((1002 + 6 * (frame_index + 1)) // 6) % 100800
            put_field(frame, 30, 47, z_count)
            put_field(frame, 49, 52, subframe_id)
            if subframe_id == 1:
                put_field(frame, 60, 70, 50)
                put_field(frame, 76, 82, 0)
                put_field(frame, 196, 204, 42)
                put_field(frame, 218, 234, 345600 // 16)
            elif subframe_id == 2:
                put_field(frame, 60, 68, 42)
                put_field(frame, 166, 174, eccentricity_raw >> 24)
                put_field(frame, 180, 204, eccentricity_raw & ((1 << 24) - 1))
                put_field(frame, 226, 234, sqrt_a_raw >> 24)
                put_field(frame, 240, 264, sqrt_a_raw & ((1 << 24) - 1))
                put_field(frame, 270, 286, 345600 // 16)
            elif subframe_id == 3:
                put_field(frame, 270, 278, 42)
            frames.append(frame)

        # Construct parity-valid words and carry D29*/D30* between words.  This
        # exercises the LNAV data-bit inversion rather than assuming D30*=0.
        nav_bits = []
        previous_d29star, previous_d30star = "1", "1"
        d30star_values = []
        for frame in frames:
            for word_index in range(10):
                data = frame[word_index * 30:word_index * 30 + 24].copy()
                if previous_d30star == "1":
                    data = ["0" if bit == "1" else "1" for bit in data]
                for parity_value in range(64):
                    word = data + list(f"{parity_value:06b}")
                    bipolar = np.array([
                        1 if bit == "1" else -1
                        for bit in previous_d29star + previous_d30star + "".join(word)
                    ], dtype=np.int8)
                    if nav_party_check(bipolar):
                        break
                else:
                    self.fail("Could not construct a valid LNAV parity word")

                nav_bits.extend(word)
                previous_d29star, previous_d30star = word[-2:]
                d30star_values.append(previous_d30star)

        self.assertGreater(
            sum(left != right for left, right in zip(d30star_values, d30star_values[1:])),
            0,
            "the test stream should contain D30* transitions",
        )

        # Two errors in each TLM preamble test the relaxed correlation gate;
        # they do not affect the ephemeris fields. Add a 7-ms symbol offset.
        for frame_start in range(0, len(nav_bits), 300):
            for bit_index in (0, 1):
                absolute_index = frame_start + bit_index
                nav_bits[absolute_index] = "0" if nav_bits[absolute_index] == "1" else "1"
        nav_signs = np.where(np.asarray(nav_bits) == "1", 1.0, -1.0)
        prompt = np.concatenate((np.full(7, 0.1), np.repeat(nav_signs, 20)))

        candidates = find_nav_preambles({"I_P": prompt})
        self.assertTrue(any(candidate["phase_ms"] == 7 for candidate in candidates))
        decoded = decode_ephemeris_from_tracking({"I_P": prompt, "PRN": 4}, prn=4)
        self.assertEqual(decoded["subframe_ids"], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(decoded["tow_seconds"], 1002.0)
        self.assertAlmostEqual(decoded["ephemeris"]["sqrtA"], 5153.7954775, places=4)
        self.assertAlmostEqual(decoded["ephemeris"]["e"], 0.01, places=8)

    def test_short_tracking_record_does_not_claim_position(self):
        short_track = {
            "PRN": 1,
            "I_P": np.ones(125),
            "absolute_sample": np.arange(1, 126, dtype=np.float64) * 50000,
        }
        result = solve_navigation(SimpleNamespace(results=[short_track]), Settings())
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["solutions"], [])
        self.assertIn("at least four", result["message"])

    def test_least_squares_recovers_known_wgs84_position(self):
        latitude = np.deg2rad(61.45)
        longitude = np.deg2rad(23.86)
        height = 100.0
        semi_major_axis = 6378137.0
        flattening = 1 / 298.257223563
        eccentricity_sq = flattening * (2 - flattening)
        prime_vertical_radius = semi_major_axis / np.sqrt(
            1 - eccentricity_sq * np.sin(latitude) ** 2
        )
        receiver = np.array([
            (prime_vertical_radius + height) * np.cos(latitude) * np.cos(longitude),
            (prime_vertical_radius + height) * np.cos(latitude) * np.sin(longitude),
            (prime_vertical_radius * (1 - eccentricity_sq) + height) * np.sin(latitude),
        ])
        enu_to_ecef = np.array([
            [-np.sin(longitude), -np.sin(latitude) * np.cos(longitude), np.cos(latitude) * np.cos(longitude)],
            [np.cos(longitude), -np.sin(latitude) * np.sin(longitude), np.cos(latitude) * np.sin(longitude)],
            [0.0, np.cos(latitude), np.sin(latitude)],
        ])

        satellite_positions = []
        for azimuth_deg, elevation_deg in (
            (0, 25), (45, 55), (90, 30), (135, 60),
            (190, 35), (240, 70), (290, 40), (330, 20),
        ):
            azimuth, elevation = np.deg2rad([azimuth_deg, elevation_deg])
            line_of_sight_enu = np.array([
                np.cos(elevation) * np.sin(azimuth),
                np.cos(elevation) * np.cos(azimuth),
                np.sin(elevation),
            ])
            satellite_positions.append(receiver + enu_to_ecef @ line_of_sight_enu * 21_000_000.0)

        satellite_positions = np.asarray(satellite_positions)
        receiver_clock_bias_m = 123456.78
        pseudoranges = []
        for satellite in satellite_positions:
            travel_time = np.linalg.norm(satellite - receiver) / C
            rotated_satellite = e_r_corr(travel_time, satellite)
            pseudoranges.append(np.linalg.norm(rotated_satellite - receiver) + receiver_clock_bias_m)

        solution = least_squares_position(
            satellite_positions, pseudoranges, use_trop_corr=False
        )
        self.assertLess(np.linalg.norm(np.asarray(solution["ecef_m"]) - receiver), 1e-3)
        self.assertAlmostEqual(solution["receiver_clock_bias_m"], receiver_clock_bias_m, places=3)
        self.assertEqual(len(solution["dop"]), 5)

        lat, lon, alt = ecef_to_geodetic(solution["ecef_m"])
        self.assertAlmostEqual(lat, 61.45, places=7)
        self.assertAlmostEqual(lon, 23.86, places=7)
        self.assertAlmostEqual(alt, 100.0, places=3)

    def test_float32_and_int16_format_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            float_path = Path(directory) / "float_iq.bin"
            int16_path = Path(directory) / "int16_iq.bin"
            iq_float = np.linspace(-0.75, 0.75, 1000, dtype="<f4")
            iq_int16 = np.arange(-500, 500, dtype="<i2")
            iq_float.tofile(float_path)
            iq_int16.tofile(int16_path)
            self.assertEqual(detect_data_format(str(float_path)), "float32")
            self.assertEqual(detect_data_format(str(int16_path)), "int16")


if __name__ == "__main__":
    unittest.main()
