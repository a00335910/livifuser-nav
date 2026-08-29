import math
import unittest

from livifuser_nav.run_checks import (
    AngularFrame,
    compare_transform,
    count_timestamp_regressions,
    scan_geometry_report,
)

TWO_PI = 2.0 * math.pi

# The accepted base_scan -> camera_optical_frame transform.
ACCEPTED_T = (0.0723955522, 0.0048472604, -0.0838973150)
ACCEPTED_Q = (-0.4806489642, 0.5212435451, -0.4930249275, 0.5041905996)


def geometry_data(beams: int = 399, increment: float | None = None) -> dict[str, object]:
    return {
        "beam_count": beams,
        "angle_min": 0.0,
        "angle_max": TWO_PI,
        "angle_increment": TWO_PI / 400.0 if increment is None else increment,
        "range_min": 0.1,
        "range_max": 100.0,
    }


class RegressionCountTests(unittest.TestCase):
    def test_monotonic_stream_has_no_regressions(self) -> None:
        self.assertEqual(count_timestamp_regressions([1, 2, 3, 4, 5]), 0)

    def test_equal_timestamps_are_not_regressions(self) -> None:
        self.assertEqual(count_timestamp_regressions([1, 1, 2, 2]), 0)

    def test_counts_events_not_displaced_positions(self) -> None:
        # One early element out of place. A sort-based count would report many
        # displaced positions; there is exactly one backward step.
        self.assertEqual(count_timestamp_regressions([5, 1, 2, 3, 4]), 1)

    def test_counts_each_separate_backward_step(self) -> None:
        self.assertEqual(count_timestamp_regressions([1, 5, 2, 6, 3]), 2)

    def test_short_streams_are_handled(self) -> None:
        self.assertEqual(count_timestamp_regressions([]), 0)
        self.assertEqual(count_timestamp_regressions([7]), 0)


class ScanGeometryTests(unittest.TestCase):
    def test_constant_frame_is_reported_once(self) -> None:
        report = scan_geometry_report([geometry_data() for _ in range(50)])
        self.assertTrue(report.is_constant)
        assert report.frame is not None
        self.assertEqual(report.beam_counts, (399,))

    def test_beam_count_and_covarying_increment_are_not_a_frame_change(self) -> None:
        # The real driver emits 398-400 returns and sets angle_increment to
        # 2*pi/(beams+1) to match, so both vary together on healthy hardware.
        report = scan_geometry_report(
            [
                geometry_data(beams=399, increment=TWO_PI / 400.0),
                geometry_data(beams=400, increment=TWO_PI / 401.0),
                geometry_data(beams=398, increment=TWO_PI / 399.0),
            ]
        )
        self.assertTrue(report.is_constant)
        self.assertEqual(report.beam_counts, (398, 399, 400))
        self.assertTrue(report.as_manifest()["angle_increment_varies"])
        self.assertEqual(report.max_beam_count, 400)

    def test_bearing_spread_is_quantified(self) -> None:
        report = scan_geometry_report(
            [
                geometry_data(beams=400, increment=TWO_PI / 399.0),
                geometry_data(beams=400, increment=TWO_PI / 401.0),
            ]
        )
        # Roughly a degree of disagreement at the far beam.
        self.assertGreater(report.max_bearing_spread_deg, 0.5)
        self.assertLess(report.max_bearing_spread_deg, 2.0)

    def test_no_spread_when_increment_is_stable(self) -> None:
        self.assertEqual(
            scan_geometry_report([geometry_data(), geometry_data()]).max_bearing_spread_deg,
            0.0,
        )

    def test_angle_max_change_is_a_frame_change(self) -> None:
        shifted = geometry_data()
        shifted["angle_max"] = math.pi
        self.assertFalse(scan_geometry_report([geometry_data(), shifted]).is_constant)

    def test_angle_min_change_is_a_geometry_change(self) -> None:
        shifted = geometry_data()
        shifted["angle_min"] = -math.pi
        self.assertFalse(scan_geometry_report([geometry_data(), shifted]).is_constant)

    def test_range_limit_change_is_a_geometry_change(self) -> None:
        shifted = geometry_data()
        shifted["range_max"] = 12.0
        self.assertFalse(scan_geometry_report([geometry_data(), shifted]).is_constant)

    def test_bearings_use_the_per_scan_increment(self) -> None:
        frame = AngularFrame(
            angle_min=0.0, angle_max=TWO_PI, range_min=0.1, range_max=100.0
        )
        increment = TWO_PI / 400.0
        self.assertAlmostEqual(frame.bearing_rad(0, increment), 0.0)
        self.assertAlmostEqual(frame.bearing_rad(100, increment), math.pi / 2.0, places=6)
        # A different scan's increment yields a different bearing for beam 200.
        self.assertNotAlmostEqual(
            frame.bearing_rad(200, increment),
            frame.bearing_rad(200, TWO_PI / 401.0),
            places=4,
        )

    def test_manifest_carries_everything_needed_to_tokenize(self) -> None:
        manifest = scan_geometry_report([geometry_data()]).as_manifest()
        frame = manifest["angular_frame"]
        assert isinstance(frame, dict)
        for key in ("angle_min_rad", "angle_max_rad", "range_min_m", "range_max_m"):
            self.assertIn(key, frame)
        self.assertIn("max_beam_count", manifest)
        self.assertIn("angle_increments_observed_rad", manifest)


class TransformComparisonTests(unittest.TestCase):
    def test_identical_transform_matches(self) -> None:
        result = compare_transform(ACCEPTED_T, ACCEPTED_Q, ACCEPTED_T, ACCEPTED_Q)
        self.assertTrue(result.matches)
        self.assertAlmostEqual(result.translation_error_m, 0.0)
        self.assertAlmostEqual(result.rotation_error_rad, 0.0, places=6)

    def test_negated_quaternion_is_the_same_rotation(self) -> None:
        negated = tuple(-value for value in ACCEPTED_Q)
        result = compare_transform(ACCEPTED_T, negated, ACCEPTED_T, ACCEPTED_Q)
        self.assertTrue(result.matches)
        self.assertAlmostEqual(result.rotation_error_rad, 0.0, places=6)

    def test_translation_error_beyond_tolerance_fails(self) -> None:
        shifted = (ACCEPTED_T[0] + 0.01, ACCEPTED_T[1], ACCEPTED_T[2])
        result = compare_transform(shifted, ACCEPTED_Q, ACCEPTED_T, ACCEPTED_Q)
        self.assertFalse(result.matches)
        self.assertAlmostEqual(result.translation_error_m, 0.01, places=6)

    def test_small_translation_error_within_tolerance_passes(self) -> None:
        shifted = (ACCEPTED_T[0] + 0.001, ACCEPTED_T[1], ACCEPTED_T[2])
        self.assertTrue(compare_transform(shifted, ACCEPTED_Q, ACCEPTED_T, ACCEPTED_Q).matches)

    def test_rotation_error_beyond_tolerance_fails(self) -> None:
        # A five-degree yaw offset applied to the identity rotation.
        angle = math.radians(5.0)
        rotated = (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))
        identity = (0.0, 0.0, 0.0, 1.0)
        result = compare_transform((0, 0, 0), rotated, (0, 0, 0), identity)
        self.assertFalse(result.matches)
        self.assertAlmostEqual(math.degrees(result.rotation_error_rad), 5.0, places=3)

    def test_identity_frame_names_cannot_substitute_for_values(self) -> None:
        # An identity transform published under the right frame names is exactly
        # the failure that a name-only check would pass.
        result = compare_transform(
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), ACCEPTED_T, ACCEPTED_Q
        )
        self.assertFalse(result.matches)

    def test_manifest_reports_errors_in_readable_units(self) -> None:
        manifest = compare_transform(
            ACCEPTED_T, ACCEPTED_Q, ACCEPTED_T, ACCEPTED_Q
        ).as_manifest()
        self.assertIn("translation_error_mm", manifest)
        self.assertIn("rotation_error_deg", manifest)
        self.assertTrue(manifest["matches"])


if __name__ == "__main__":
    unittest.main()
