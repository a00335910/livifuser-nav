"""Measured LDS-03 nominal observation-model tests.

These tests pin the distinction that matters scientifically: geometric
no-returns come from ray casting, while only the 8.7e-5 stable-surface miss
probability is injected.  Treating the measured 3.34% aggregate occupancy as
dropout would silently corrupt the C0 reference.
"""

from __future__ import annotations

import math
import random
import statistics
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_sim.analytic_lidar import (  # noqa: E402
    AnalyticLidarGeometry,
    CircleObstacle,
    LaserSpecification,
    Pose2D,
)
from livifuser_sim.observation_model import (  # noqa: E402
    LIDAR_CONDITIONS,
    apply_nominal_observation,
    apply_observation_condition,
    condition_for,
    load_observation_model,
    simulate_observation,
)

MODEL_PATH = PACKAGE_ROOT / "config" / "lds03_observation_model_v1.json"
SOURCE_SHA256 = "5A515F681C0235497C77DBC756DAA1F3FA8B03DA5925D5D9B17488824E2280FD"


def enclosed_geometry() -> AnalyticLidarGeometry:
    """A circle around the scanner makes every ideal ray a 2 m hit."""

    return AnalyticLidarGeometry(
        schema_version=1,
        source="unit-test enclosure",
        laser=LaserSpecification(400, 0.0, math.tau, 0.1, 0.12, 8.0, "base_scan"),
        obstacles=(CircleObstacle("enclosure", 0.0, 0.0, 2.0),),
    )


class TestMeasuredContract(unittest.TestCase):
    def setUp(self) -> None:
        self.model = load_observation_model(MODEL_PATH)

    def test_source_identity_and_histogram_are_pinned(self) -> None:
        self.assertEqual(self.model.schema_version, "1.0.0")
        self.assertEqual(self.model.source_artifact_sha256, SOURCE_SHA256)
        self.assertEqual(self.model.source_scan_count, 15_295)
        self.assertEqual(self.model.beam_counts.total_weight, 15_295)
        self.assertEqual(self.model.beam_counts.values[0], 379)
        self.assertEqual(self.model.beam_counts.values[-1], 407)
        weights = dict(
            zip(self.model.beam_counts.values, self.model.beam_counts.weights, strict=True)
        )
        self.assertEqual(weights[399], 5_562)
        self.assertEqual(weights[400], 5_368)

    def test_frozen_measured_values_are_exact(self) -> None:
        self.assertEqual(self.model.angle_min_rad, 0.0)
        self.assertEqual(self.model.angle_max_rad, math.tau)
        self.assertEqual(self.model.scan_interval_sec, 0.099677066)
        self.assertEqual(self.model.range_noise_sigma_m, 0.0030821574918964006)
        self.assertEqual(self.model.range_quantization_step_m, 0.001)
        self.assertEqual(self.model.stochastic_missing_return_probability, 0.000087)
        self.assertEqual(self.model.no_return_encoding.zero_count, 124_051)
        self.assertEqual(self.model.no_return_encoding.nan_count, 79_723)
        self.assertEqual(
            self.model.excluded_aggregate_no_return_occupancy,
            0.033381709540223174,
        )

    def test_histogram_sampling_is_deterministic_and_variable(self) -> None:
        first = random.Random(20260821)
        second = random.Random(20260821)
        first_draws = [self.model.beam_counts.sample(first) for _ in range(200)]
        second_draws = [self.model.beam_counts.sample(second) for _ in range(200)]
        self.assertEqual(first_draws, second_draws)
        self.assertGreaterEqual(len(set(first_draws)), 5)
        self.assertTrue(all(379 <= value <= 407 for value in first_draws))

    def test_histogram_sampling_tracks_source_mass(self) -> None:
        generator = random.Random(90210)
        draws = [self.model.beam_counts.sample(generator) for _ in range(152_950)]
        source_mean = sum(
            value * weight
            for value, weight in zip(
                self.model.beam_counts.values,
                self.model.beam_counts.weights,
                strict=True,
            )
        ) / self.model.beam_counts.total_weight
        self.assertAlmostEqual(statistics.mean(draws), source_mean, delta=0.02)
        source_399 = 5_562 / 15_295
        source_400 = 5_368 / 15_295
        self.assertAlmostEqual(draws.count(399) / len(draws), source_399, delta=0.004)
        self.assertAlmostEqual(draws.count(400) / len(draws), source_400, delta=0.004)

    def test_dynamic_specification_uses_per_scan_increment(self) -> None:
        geometry = enclosed_geometry()
        for beam_count in (379, 399, 407):
            specification = self.model.specification_for(geometry, beam_count)
            self.assertEqual(specification.beam_count, beam_count)
            self.assertAlmostEqual(
                specification.angle_increment_rad,
                math.tau / (beam_count + 1),
                places=15,
            )
            self.assertEqual(specification.scan_time_sec, self.model.scan_interval_sec)


class TestNominalObservation(unittest.TestCase):
    def setUp(self) -> None:
        self.model = load_observation_model(MODEL_PATH)
        self.geometry = enclosed_geometry()

    def test_end_to_end_scans_vary_in_length_and_are_reproducible(self) -> None:
        first = random.Random(71)
        second = random.Random(71)
        pose = Pose2D(0.0, 0.0, 0.0)
        scans_a = [simulate_observation(self.geometry, pose, self.model, first) for _ in range(30)]
        scans_b = [simulate_observation(self.geometry, pose, self.model, second) for _ in range(30)]
        self.assertEqual(scans_a, scans_b)
        counts = {scan.specification.beam_count for scan in scans_a}
        self.assertGreaterEqual(len(counts), 4)
        for scan in scans_a:
            self.assertEqual(len(scan.ranges), scan.specification.beam_count)

    def test_finite_ranges_are_on_the_one_millimetre_lattice(self) -> None:
        scan = simulate_observation(
            self.geometry,
            Pose2D(0.0, 0.0, 0.0),
            self.model,
            random.Random(8128),
        )
        finite = [value for value in scan.ranges if math.isfinite(value) and value > 0.0]
        self.assertGreater(len(finite), 350)
        for value in finite:
            self.assertAlmostEqual(value / 0.001, round(value / 0.001), places=9)

    def test_noise_is_applied_before_quantization_at_measured_scale(self) -> None:
        beam_count = 100_000
        specification = LaserSpecification(
            beam_count, 0.0, math.tau, 0.1, 0.12, 8.0, "base_scan"
        )
        scan = apply_nominal_observation(
            (2.0,) * beam_count,
            specification,
            self.model,
            random.Random(3108),
        )
        residuals = [
            value - 2.0 for value in scan.ranges if math.isfinite(value) and value > 0.0
        ]
        self.assertGreater(len(residuals), 99_900)
        self.assertAlmostEqual(statistics.mean(residuals), 0.0, delta=0.00005)
        self.assertGreater(statistics.pstdev(residuals), 0.0030)
        self.assertLess(statistics.pstdev(residuals), 0.0032)

    def test_only_stable_surface_miss_probability_is_injected(self) -> None:
        beam_count = 400_000
        specification = LaserSpecification(
            beam_count, 0.0, math.tau, 0.1, 0.12, 8.0, "base_scan"
        )
        scan = apply_nominal_observation(
            (2.0,) * beam_count,
            specification,
            self.model,
            random.Random(8700),
        )
        fraction = scan.stochastic_missing_return_count / beam_count
        self.assertGreater(scan.stochastic_missing_return_count, 15)
        self.assertLess(scan.stochastic_missing_return_count, 60)
        self.assertLess(fraction, 0.0005)
        self.assertLess(fraction, self.model.excluded_aggregate_no_return_occupancy / 50.0)
        self.assertEqual(scan.geometric_no_return_count, 0)

    def test_genuine_no_returns_use_zero_nan_split_and_never_infinity(self) -> None:
        beam_count = 100_000
        specification = LaserSpecification(
            beam_count, 0.0, math.tau, 0.1, 0.12, 8.0, "base_scan"
        )
        scan = apply_nominal_observation(
            (math.inf,) * beam_count,
            specification,
            self.model,
            random.Random(609391),
        )
        zeros = sum(value == 0.0 for value in scan.ranges)
        nans = sum(math.isnan(value) for value in scan.ranges)
        self.assertEqual(zeros + nans, beam_count)
        self.assertEqual(scan.geometric_no_return_count, beam_count)
        self.assertEqual(scan.stochastic_missing_return_count, 0)
        self.assertAlmostEqual(
            zeros / beam_count,
            self.model.no_return_encoding.zero_probability,
            delta=0.005,
        )
        self.assertFalse(any(math.isinf(value) for value in scan.ranges))

    def test_out_of_range_noisy_values_become_geometric_no_returns(self) -> None:
        specification = LaserSpecification(2, 0.0, math.tau, 0.1, 0.12, 8.0, "base_scan")
        scan = apply_nominal_observation(
            (0.01, 9.0),
            specification,
            self.model,
            random.Random(1),
        )
        self.assertEqual(scan.geometric_no_return_count, 2)
        self.assertTrue(all(value == 0.0 or math.isnan(value) for value in scan.ranges))


class TestFrozenC3Conditions(unittest.TestCase):
    def setUp(self) -> None:
        self.model = load_observation_model(MODEL_PATH)
        self.geometry = enclosed_geometry()

    def test_preregistered_magnitudes_are_exact(self) -> None:
        self.assertEqual(LIDAR_CONDITIONS["C3a"].range_noise_sigma_m, 0.0154)
        self.assertEqual(LIDAR_CONDITIONS["C3a"].missing_return_probability, 0.01)
        self.assertEqual(LIDAR_CONDITIONS["C3a"].structured_dropout_width_rad, 0.0)
        self.assertEqual(LIDAR_CONDITIONS["C3b"].range_noise_sigma_m, 0.0616)
        self.assertEqual(LIDAR_CONDITIONS["C3b"].missing_return_probability, 0.05)
        self.assertAlmostEqual(
            math.degrees(LIDAR_CONDITIONS["C3b"].structured_dropout_width_rad),
            30.0,
        )
        self.assertEqual(
            condition_for(self.model, "C0").range_noise_sigma_m,
            0.0030821574918964006,
        )

    def test_unknown_condition_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown LiDAR condition"):
            condition_for(self.model, "C9")

    def test_c3b_is_reproducible_and_drops_one_thirty_degree_sector(self) -> None:
        specification = LaserSpecification(400, 0.0, math.tau, 0.1, 0.12, 8.0, "base_scan")
        first = apply_observation_condition(
            (2.0,) * 400,
            specification,
            self.model,
            random.Random(9901),
            LIDAR_CONDITIONS["C3b"],
        )
        second = apply_observation_condition(
            (2.0,) * 400,
            specification,
            self.model,
            random.Random(9901),
            LIDAR_CONDITIONS["C3b"],
        )
        self.assertEqual(first, second)
        self.assertEqual(first.condition, "C3b")
        # 30 degrees covers 33 or 34 sampled bearings at this increment.
        self.assertIn(first.structured_missing_return_count, (33, 34))
        self.assertGreater(first.stochastic_missing_return_count, 5)
        self.assertLess(first.stochastic_missing_return_count, 35)

    def test_c3b_noise_is_the_total_frozen_sigma_not_nominal_plus_severe(self) -> None:
        beam_count = 100_000
        specification = LaserSpecification(
            beam_count, 0.0, math.tau, 0.1, 0.12, 8.0, "base_scan"
        )
        scan = apply_observation_condition(
            (2.0,) * beam_count,
            specification,
            self.model,
            random.Random(613),
            LIDAR_CONDITIONS["C3b"],
        )
        residuals = [
            value - 2.0 for value in scan.ranges if math.isfinite(value) and value > 0.0
        ]
        self.assertGreater(len(residuals), 85_000)
        self.assertAlmostEqual(statistics.mean(residuals), 0.0, delta=0.0006)
        self.assertGreater(statistics.pstdev(residuals), 0.0610)
        self.assertLess(statistics.pstdev(residuals), 0.0622)

    def test_condition_changes_scan_but_not_ray_cast_geometry(self) -> None:
        pose = Pose2D(0.0, 0.0, 0.0)
        nominal = simulate_observation(
            self.geometry, pose, self.model, random.Random(44), "C0"
        )
        severe = simulate_observation(
            self.geometry, pose, self.model, random.Random(44), "C3b"
        )
        self.assertEqual(nominal.specification.beam_count, severe.specification.beam_count)
        self.assertNotEqual(nominal.ranges, severe.ranges)
        self.assertEqual(severe.condition, "C3b")


if __name__ == "__main__":
    unittest.main()
