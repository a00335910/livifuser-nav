"""Tests for offline LDS-03 observation characterization.

The cases here pin the three distinctions the module exists to protect: bearing
versus beam index, no-return versus dropout, and repeatability versus accuracy.
Each has already been an observed source of error on this sensor.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from livifuser_nav.lds03_characterization import (
    EVIDENCE_ODOMETRY_AND_COMMAND,
    EVIDENCE_ODOMETRY_ONLY,
    FULL_TURN_RAD,
    CharacterizationError,
    ScanRecord,
    beam_bearings,
    classify_returns,
    confirm_intervals_by_zero_command,
    detect_range_quantization,
    expected_angle_increment,
    find_motion_free_intervals,
    increment_convention_residual,
    interval_statistics,
    no_return_occupancy_by_sector,
    pooled_interval_statistics,
    robust_repeatability,
    sector_indices,
    sector_range_series,
    stable_bearing_eligibility,
    stochastic_missing_return,
)


def make_scan(
    ranges: np.ndarray,
    *,
    stamp_ns: int = 0,
    angle_min: float = 0.0,
    increment: float | None = None,
    range_min: float = 0.1,
    range_max: float = 100.0,
) -> ScanRecord:
    beam_count = int(np.asarray(ranges).size)
    step = expected_angle_increment(beam_count) if increment is None else increment
    return ScanRecord(
        stamp_ns=stamp_ns,
        angle_min=angle_min,
        angle_max=angle_min + beam_count * step,
        angle_increment=step,
        range_min=range_min,
        range_max=range_max,
        ranges=np.asarray(ranges, dtype=np.float64),
    )


class IncrementConventionTests(unittest.TestCase):
    def test_plus_one_convention_matches_recorded_driver_values(self) -> None:
        # Exactly the values decoded from stationary_pilot_2026-07-29_01.
        residual = increment_convention_residual(399, 0.015707964077591896)
        self.assertLess(residual["relative_error"], 1e-6)
        self.assertAlmostEqual(residual["expected_increment_rad"], FULL_TURN_RAD / 400)

    def test_dropping_the_plus_one_is_detectably_wrong(self) -> None:
        naive = FULL_TURN_RAD / 399
        residual = increment_convention_residual(399, naive)
        self.assertGreater(residual["relative_error"], 1e-3)

    def test_rejects_non_positive_beam_count(self) -> None:
        with self.assertRaisesRegex(CharacterizationError, "beam_count must be positive"):
            expected_angle_increment(0)


class BearingReconstructionTests(unittest.TestCase):
    def test_bearings_use_each_scans_own_increment(self) -> None:
        for beam_count in (396, 399, 404):
            bearings = beam_bearings(0.0, expected_angle_increment(beam_count), beam_count)
            self.assertEqual(bearings.size, beam_count)
            self.assertAlmostEqual(bearings[0], 0.0)
            # The emitted arc stops one step short of closing the circle.
            self.assertLess(bearings[-1], FULL_TURN_RAD)
            self.assertAlmostEqual(
                bearings[-1], FULL_TURN_RAD * (beam_count - 1) / (beam_count + 1)
            )

    def test_shared_bearing_disagreement_across_beam_counts_is_bounded(self) -> None:
        """Beam i drifts across scans; physical bearing does not."""

        low = beam_bearings(0.0, expected_angle_increment(396), 396)
        high = beam_bearings(0.0, expected_angle_increment(404), 404)
        # Same index, materially different bearing: this is the trap.
        far_index = 395
        drift = abs(low[far_index] - high[far_index])
        self.assertGreater(math.degrees(drift), 5.0)

    def test_rejects_non_finite_geometry(self) -> None:
        with self.assertRaisesRegex(CharacterizationError, "must be finite"):
            beam_bearings(float("nan"), 0.0157, 400)
        with self.assertRaisesRegex(CharacterizationError, "must be positive"):
            beam_bearings(0.0, -0.0157, 400)


class SectorAlignmentTests(unittest.TestCase):
    def test_same_physical_bearing_lands_in_one_sector_across_beam_counts(self) -> None:
        """The alignment guarantee the whole cross-scan comparison rests on."""

        sector_count = 72  # 5-degree sectors
        for beam_count in (396, 399, 404):
            bearings = beam_bearings(0.0, expected_angle_increment(beam_count), beam_count)
            sectors = sector_indices(bearings, sector_count)
            # The beam nearest 90 degrees must fall in the sector holding 90 degrees.
            nearest = int(np.argmin(np.abs(bearings - math.pi / 2)))
            expected = sector_indices(np.array([math.pi / 2]), sector_count)[0]
            self.assertEqual(sectors[nearest], expected)

    def test_sector_zero_brackets_the_origin_bearing(self) -> None:
        sector_count = 360
        just_below = sector_indices(np.array([FULL_TURN_RAD - 1e-6]), sector_count)[0]
        just_above = sector_indices(np.array([1e-6]), sector_count)[0]
        self.assertEqual(just_below, 0)
        self.assertEqual(just_above, 0)

    def test_indices_stay_in_range_for_wrapped_input(self) -> None:
        bearings = np.array([-0.4, 0.0, 3.0, 6.28, 12.0, -7.0])
        sectors = sector_indices(bearings, 36)
        self.assertTrue(np.all(sectors >= 0))
        self.assertTrue(np.all(sectors < 36))


class ReturnClassificationTests(unittest.TestCase):
    def test_nan_and_zero_are_distinct_no_return_codes(self) -> None:
        record = make_scan(np.array([1.0, float("nan"), 0.0, 2.0]))
        classes = classify_returns(record)
        self.assertEqual(classes["valid"].tolist(), [True, False, False, True])
        self.assertEqual(classes["nan"].tolist(), [False, True, False, False])
        self.assertEqual(classes["zero"].tolist(), [False, False, True, False])

    def test_declared_range_max_is_not_used_as_an_upper_bound(self) -> None:
        """range_max is declared 100 m, far past anything this sensor measures."""

        record = make_scan(np.array([11.7, 50.0]), range_max=100.0)
        self.assertTrue(bool(classify_returns(record)["valid"].all()))

    def test_below_declared_range_min_is_separated_from_valid(self) -> None:
        record = make_scan(np.array([0.05, 0.5]), range_min=0.1)
        classes = classify_returns(record)
        self.assertEqual(classes["valid"].tolist(), [False, True])
        self.assertEqual(classes["below_declared_range_min"].tolist(), [True, False])


class NoReturnOccupancyTests(unittest.TestCase):
    def test_open_space_reads_as_occupancy_not_dropout(self) -> None:
        """A bearing that never returns is open space, and must not imply a fault."""

        sector_count = 4
        beams = 7
        ranges = np.full(beams, 2.0)
        bearings = beam_bearings(0.0, expected_angle_increment(beams), beams)
        open_sector = sector_indices(bearings, sector_count) == 2
        ranges[open_sector] = np.nan
        records = [make_scan(ranges, stamp_ns=i) for i in range(10)]

        occupancy = no_return_occupancy_by_sector(records, sector_count)
        self.assertEqual(occupancy["no_return_occupancy"][2], 1.0)
        self.assertEqual(occupancy["no_return_occupancy"][0], 0.0)

        # The same sector is then excluded from any dropout estimate.
        values, valid, covered = sector_range_series(records, sector_count)
        eligibility = stable_bearing_eligibility(
            values,
            valid,
            covered,
            min_valid_fraction=0.9,
            min_observations=5,
            min_range_m=0.2,
            max_range_m=12.0,
            max_neighbor_step_m=0.5,
            max_mad_m=0.05,
            max_half_split_drift_m=0.05,
        )
        self.assertFalse(bool(eligibility["eligible"][2]))

    def test_unobserved_sector_reports_none_rather_than_zero(self) -> None:
        occupancy = no_return_occupancy_by_sector([make_scan(np.array([1.0]))], 8)
        self.assertIn(None, occupancy["no_return_occupancy"])


class EligibilityTests(unittest.TestCase):
    @staticmethod
    def flat_wall(sector_count: int = 16, scans: int = 40, noise: float = 0.0) -> tuple:
        rng = np.random.default_rng(0)
        values = np.full((scans, sector_count), 2.0)
        if noise:
            values = values + rng.normal(0.0, noise, size=values.shape)
        valid = np.ones((scans, sector_count), dtype=bool)
        covered = np.ones((scans, sector_count), dtype=bool)
        return values, valid, covered

    def test_depth_discontinuity_neighbours_are_excluded(self) -> None:
        values, valid, covered = self.flat_wall()
        values[:, 8] = 6.0  # a doorway: large step against both neighbours
        eligibility = stable_bearing_eligibility(
            values,
            valid,
            covered,
            min_valid_fraction=0.9,
            min_observations=5,
            min_range_m=0.2,
            max_range_m=12.0,
            max_neighbor_step_m=0.5,
            max_mad_m=0.05,
            max_half_split_drift_m=0.05,
        )
        eligible = eligibility["eligible"]
        self.assertFalse(bool(eligible[8]))
        self.assertFalse(bool(eligible[7]))
        self.assertFalse(bool(eligible[9]))
        self.assertTrue(bool(eligible[3]))
        self.assertEqual(eligibility["excluded_counts"]["angular_discontinuity"], 3)

    def test_unstable_surface_is_excluded(self) -> None:
        values, valid, covered = self.flat_wall()
        values[:, 5] += np.linspace(-1.0, 1.0, values.shape[0])
        eligibility = stable_bearing_eligibility(
            values,
            valid,
            covered,
            min_valid_fraction=0.9,
            min_observations=5,
            min_range_m=0.2,
            max_range_m=12.0,
            max_neighbor_step_m=0.5,
            max_mad_m=0.05,
            max_half_split_drift_m=0.05,
        )
        self.assertFalse(bool(eligibility["eligible"][5]))
        self.assertEqual(eligibility["excluded_counts"]["unstable_surface"], 1)

    def test_drifting_surface_is_excluded_without_thresholding_spread(self) -> None:
        """A surface that moved must go, even though its dispersion stays small."""

        values, valid, covered = self.flat_wall(scans=40)
        values[20:, 6] = 2.4  # object repositioned midway; MAD stays tiny either side
        eligibility = stable_bearing_eligibility(
            values,
            valid,
            covered,
            min_valid_fraction=0.9,
            min_observations=5,
            min_range_m=0.2,
            max_range_m=12.0,
            max_neighbor_step_m=0.5,
            max_mad_m=0.5,
            max_half_split_drift_m=0.05,
        )
        self.assertFalse(bool(eligibility["eligible"][6]))
        self.assertAlmostEqual(float(eligibility["half_split_drift_m"][6]), 0.4)
        self.assertTrue(bool(eligibility["eligible"][2]))

    def test_intermittent_sector_fails_the_valid_fraction_gate(self) -> None:
        values, valid, covered = self.flat_wall()
        valid[::2, 4] = False
        eligibility = stable_bearing_eligibility(
            values,
            valid,
            covered,
            min_valid_fraction=0.9,
            min_observations=5,
            min_range_m=0.2,
            max_range_m=12.0,
            max_neighbor_step_m=0.5,
            max_mad_m=0.05,
            max_half_split_drift_m=0.05,
        )
        self.assertFalse(bool(eligibility["eligible"][4]))


class RepeatabilityTests(unittest.TestCase):
    def test_robust_sigma_recovers_a_known_spread(self) -> None:
        rng = np.random.default_rng(7)
        sector_count, scans, sigma = 12, 400, 0.01
        values = 2.0 + rng.normal(0.0, sigma, size=(scans, sector_count))
        valid = np.ones_like(values, dtype=bool)
        eligible = np.ones(sector_count, dtype=bool)
        result = robust_repeatability(values, valid, eligible)
        self.assertAlmostEqual(result["overall"]["robust_sigma_m"], sigma, delta=0.002)
        self.assertEqual(result["sample_count"], scans * sector_count)

    def test_outliers_do_not_dominate_the_robust_estimate(self) -> None:
        rng = np.random.default_rng(11)
        values = 2.0 + rng.normal(0.0, 0.01, size=(200, 4))
        values[0, :] = 9.0  # one straddled edge
        valid = np.ones_like(values, dtype=bool)
        result = robust_repeatability(values, valid, np.ones(4, dtype=bool))
        self.assertLess(result["overall"]["robust_sigma_m"], 0.02)
        self.assertGreater(result["overall"]["max_abs_residual_m"], 6.0)

    def test_sparse_range_bin_is_reported_as_unsupported(self) -> None:
        values = np.full((10, 2), 2.0)
        valid = np.ones_like(values, dtype=bool)
        result = robust_repeatability(
            values,
            valid,
            np.ones(2, dtype=bool),
            range_bin_edges_m=[0.0, 3.0, 6.0],
            min_bin_samples=200,
        )
        first = result["range_binned"][0]
        self.assertIsNone(first["summary"])
        self.assertIn("below min_bin_samples", first["note"])

    def test_no_eligible_sector_returns_explicit_absence(self) -> None:
        values = np.full((5, 3), 2.0)
        valid = np.ones_like(values, dtype=bool)
        result = robust_repeatability(values, valid, np.zeros(3, dtype=bool))
        self.assertEqual(result["sample_count"], 0)
        self.assertIsNone(result["overall"])


class MissingReturnTests(unittest.TestCase):
    def test_estimate_is_restricted_to_eligible_sectors(self) -> None:
        scans, sector_count = 100, 4
        valid = np.ones((scans, sector_count), dtype=bool)
        valid[:10, 1] = False  # 10% intermittent miss on a real surface
        valid[:, 3] = False  # open space; must not enter the estimate
        eligible = np.array([True, True, True, False])
        covered = np.ones((scans, sector_count), dtype=bool)
        result = stochastic_missing_return(valid, covered, eligible)
        self.assertEqual(result["eligible_sector_count"], 3)
        self.assertAlmostEqual(result["estimate"]["pooled_probability"], 10 / 300)
        self.assertAlmostEqual(result["estimate"]["per_sector_max"], 0.1)

    def test_uncovered_scans_do_not_count_as_missing_returns(self) -> None:
        """Angular binning must never manufacture dropout."""

        scans, sector_count = 50, 3
        valid = np.ones((scans, sector_count), dtype=bool)
        covered = np.ones((scans, sector_count), dtype=bool)
        # Sector 1 simply received no beam in half the scans.
        valid[::2, 1] = False
        covered[::2, 1] = False
        result = stochastic_missing_return(valid, covered, np.ones(sector_count, dtype=bool))
        self.assertAlmostEqual(result["estimate"]["pooled_probability"], 0.0)
        self.assertAlmostEqual(result["estimate"]["per_sector_max"], 0.0)

    def test_zero_coverage_on_an_eligible_sector_is_rejected(self) -> None:
        valid = np.ones((4, 2), dtype=bool)
        covered = np.ones((4, 2), dtype=bool)
        covered[:, 1] = False
        with self.assertRaisesRegex(CharacterizationError, "zero beam coverage"):
            stochastic_missing_return(valid, covered, np.ones(2, dtype=bool))

    def test_no_eligible_sector_returns_none(self) -> None:
        result = stochastic_missing_return(
            np.ones((5, 2), dtype=bool),
            np.ones((5, 2), dtype=bool),
            np.zeros(2, dtype=bool),
        )
        self.assertIsNone(result["estimate"])


class MotionFreeIntervalTests(unittest.TestCase):
    def test_finds_still_window_inside_a_moving_run(self) -> None:
        stamps = [int(i * 1e8) for i in range(20)]  # 10 Hz
        linear = [0.0] * 8 + [0.05] * 4 + [0.0] * 8
        angular = [0.0] * 20
        intervals = find_motion_free_intervals(
            stamps,
            linear,
            angular,
            linear_tolerance_mps=1e-3,
            angular_tolerance_radps=1e-3,
            min_duration_s=0.5,
        )
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0].sample_count, 8)
        self.assertEqual(intervals[1].sample_count, 8)
        self.assertLessEqual(intervals[0].max_abs_linear_mps, 1e-3)

    def test_short_still_window_is_rejected(self) -> None:
        stamps = [0, int(1e8)]
        intervals = find_motion_free_intervals(
            stamps,
            [0.0, 0.0],
            [0.0, 0.0],
            linear_tolerance_mps=1e-3,
            angular_tolerance_radps=1e-3,
            min_duration_s=5.0,
        )
        self.assertEqual(intervals, [])

    def test_rotation_alone_breaks_stationarity(self) -> None:
        stamps = [int(i * 1e8) for i in range(20)]
        intervals = find_motion_free_intervals(
            stamps,
            [0.0] * 20,
            [0.0] * 10 + [0.4] * 10,
            linear_tolerance_mps=1e-3,
            angular_tolerance_radps=1e-3,
            min_duration_s=0.5,
        )
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0].sample_count, 10)

    def test_command_evidence_rejects_an_interval_odometry_called_still(self) -> None:
        """Odometry can under-report motion; the command record is independent."""

        stamps = [int(i * 1e8) for i in range(30)]
        intervals = find_motion_free_intervals(
            stamps,
            [0.0] * 30,
            [0.0] * 30,
            linear_tolerance_mps=1e-3,
            angular_tolerance_radps=5e-3,
            min_duration_s=0.5,
        )
        self.assertEqual(len(intervals), 1)

        command_linear = [0.0] * 30
        command_linear[15] = 0.08
        confirmed, rejected, basis = confirm_intervals_by_zero_command(
            intervals, stamps, command_linear, [0.0] * 30
        )
        self.assertEqual(confirmed, [])
        self.assertEqual(basis, EVIDENCE_ODOMETRY_AND_COMMAND)
        self.assertEqual(rejected[0]["reason"], "nonzero command inside interval")
        self.assertEqual(rejected[0]["nonzero_command_count"], 1)

    def test_interval_without_command_evidence_is_not_silently_confirmed(self) -> None:
        stamps = [int(i * 1e8) for i in range(30)]
        intervals = find_motion_free_intervals(
            stamps,
            [0.0] * 30,
            [0.0] * 30,
            linear_tolerance_mps=1e-3,
            angular_tolerance_radps=5e-3,
            min_duration_s=0.5,
        )
        confirmed, rejected, _ = confirm_intervals_by_zero_command(intervals, [], [], [])
        self.assertEqual(confirmed, [])
        self.assertEqual(rejected[0]["reason"], "no command sample inside interval")

        # A recording made with no velocity publisher is a different case.
        absent, absent_rejected, basis = confirm_intervals_by_zero_command(
            intervals, [], [], [], command_topic_recorded=False
        )
        self.assertEqual(len(absent), 1)
        self.assertEqual(absent_rejected, [])
        self.assertEqual(basis, EVIDENCE_ODOMETRY_ONLY)

    def test_all_zero_commands_confirm_the_interval(self) -> None:
        stamps = [int(i * 1e8) for i in range(30)]
        intervals = find_motion_free_intervals(
            stamps,
            [0.0] * 30,
            [0.0] * 30,
            linear_tolerance_mps=1e-3,
            angular_tolerance_radps=5e-3,
            min_duration_s=0.5,
        )
        confirmed, rejected, basis = confirm_intervals_by_zero_command(
            intervals, stamps, [0.0] * 30, [0.0] * 30
        )
        self.assertEqual(basis, EVIDENCE_ODOMETRY_AND_COMMAND)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(rejected, [])

    def test_stationary_gyro_noise_does_not_break_an_interval(self) -> None:
        """Measured: linear is exactly 0.0 when stopped, angular carries noise."""

        stamps = [int(i * 1e8) for i in range(40)]
        angular = [0.00135 * (-1) ** i for i in range(40)]
        intervals = find_motion_free_intervals(
            stamps,
            [0.0] * 40,
            angular,
            linear_tolerance_mps=1e-3,
            angular_tolerance_radps=5e-3,
            min_duration_s=0.5,
        )
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0].sample_count, 40)

    def test_mismatched_lengths_are_rejected(self) -> None:
        with self.assertRaisesRegex(CharacterizationError, "lengths must match"):
            find_motion_free_intervals(
                [0, 1],
                [0.0],
                [0.0, 0.0],
                linear_tolerance_mps=1e-3,
                angular_tolerance_radps=1e-3,
                min_duration_s=0.1,
            )


class QuantizationTests(unittest.TestCase):
    def test_detects_the_millimetre_lattice_this_sensor_reports_on(self) -> None:
        rng = np.random.default_rng(2)
        true = 2.0 + rng.normal(0.0, 0.01, size=(20, 400))
        lattice = np.round(true / 0.001) * 0.001
        records = [make_scan(row.astype(np.float32).astype(np.float64)) for row in lattice]
        result = detect_range_quantization(records)
        self.assertEqual(result["detected_step_m"], 0.001)
        self.assertAlmostEqual(result["min_gap_between_distinct_values_m"], 0.001, places=4)
        self.assertIn("cannot resolve", result["implication"])

    def test_continuous_values_report_no_lattice(self) -> None:
        rng = np.random.default_rng(4)
        records = [make_scan(2.0 + rng.normal(0.0, 0.01, size=400)) for _ in range(5)]
        self.assertIsNone(detect_range_quantization(records)["detected_step_m"])

    def test_coarser_lattice_is_preferred_when_both_fit(self) -> None:
        """A 5 mm lattice is also on the 1 mm lattice; report the coarser truth."""

        values = np.arange(1.0, 2.0, 0.005)
        records = [make_scan(values)]
        result = detect_range_quantization(records)
        self.assertEqual(result["detected_step_m"], 0.005)

    def test_empty_input_is_handled(self) -> None:
        self.assertEqual(detect_range_quantization([])["sample_count"], 0)


class IntervalStatisticsTests(unittest.TestCase):
    def test_reports_tail_and_threshold_counts(self) -> None:
        stamps = [0, int(1e8), int(2e8), int(5e8)]  # one 300 ms gap
        stats = interval_statistics(stamps, [0.15, 0.25])
        self.assertEqual(stats["interval_count"], 3)
        self.assertAlmostEqual(stats["intervals_s"]["median"], 0.1)
        self.assertAlmostEqual(stats["intervals_s"]["max"], 0.3)
        self.assertEqual(stats["gaps_above_threshold"]["0.15"], 1)
        self.assertEqual(stats["gaps_above_threshold"]["0.25"], 1)

    def test_single_sample_yields_no_intervals(self) -> None:
        stats = interval_statistics([5], [0.15])
        self.assertEqual(stats["interval_count"], 0)
        self.assertIsNone(stats["intervals_s"])

    def test_pooling_never_spans_a_recording_boundary(self) -> None:
        """Two bags an hour apart must not imply a one-hour scan gap."""

        hour_ns = 3_600_000_000_000
        first = [0, int(1e8), int(2e8)]
        second = [hour_ns, hour_ns + int(1e8), hour_ns + int(2e8)]
        pooled = pooled_interval_statistics([first, second], [0.15])
        self.assertEqual(pooled["interval_count"], 4)
        self.assertAlmostEqual(pooled["intervals_s"]["max"], 0.1)
        self.assertEqual(pooled["gaps_above_threshold"]["0.15"], 0)

        # The naive concatenation this guards against.
        naive = interval_statistics(first + second, [0.15])
        self.assertGreater(naive["intervals_s"]["max"], 3000.0)

    def test_pooling_skips_groups_too_short_to_difference(self) -> None:
        pooled = pooled_interval_statistics([[1], [], [0, int(1e8)]], [0.15])
        self.assertEqual(pooled["group_count"], 3)
        self.assertEqual(pooled["contributing_group_count"], 1)
        self.assertEqual(pooled["interval_count"], 1)


class DeterminismTests(unittest.TestCase):
    def test_identical_input_produces_identical_output(self) -> None:
        rng = np.random.default_rng(3)
        records = []
        for index in range(30):
            beams = 396 + (index % 9)
            ranges = 2.0 + rng.normal(0.0, 0.01, size=beams)
            ranges[::17] = np.nan
            records.append(make_scan(ranges, stamp_ns=int(index * 1e8)))

        def run() -> tuple:
            occupancy = no_return_occupancy_by_sector(records, 72)
            values, valid, covered = sector_range_series(records, 72)
            eligibility = stable_bearing_eligibility(
                values,
                valid,
                covered,
                min_valid_fraction=0.9,
                min_observations=5,
                min_range_m=0.2,
                max_range_m=12.0,
                max_neighbor_step_m=0.5,
                max_mad_m=0.05,
                max_half_split_drift_m=0.05,
            )
            repeat = robust_repeatability(values, valid, eligibility["eligible"])
            return occupancy, repeat

        first_occupancy, first_repeat = run()
        second_occupancy, second_repeat = run()
        self.assertEqual(first_occupancy, second_occupancy)
        self.assertEqual(first_repeat, second_repeat)

    def test_all_reported_statistics_are_finite(self) -> None:
        rng = np.random.default_rng(5)
        values = 2.0 + rng.normal(0.0, 0.01, size=(50, 8))
        valid = np.ones_like(values, dtype=bool)
        summary = robust_repeatability(values, valid, np.ones(8, dtype=bool))["overall"]
        for key, value in summary.items():
            self.assertTrue(math.isfinite(value), f"{key} is not finite")


if __name__ == "__main__":
    unittest.main()
