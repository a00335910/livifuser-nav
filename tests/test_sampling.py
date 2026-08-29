import unittest

from livifuser_nav.association import TimeSeries
from livifuser_nav.contracts import StampedValue
from livifuser_nav.export_schema import (
    ExportPolicy,
    RejectionCode,
    primary_rejection,
)
from livifuser_nav.sampling import (
    Payload,
    assemble_samples,
    build_grid,
)

MS = 1_000_000


def series(*stamps_ms: float, valid: bool = True) -> TimeSeries[Payload]:
    invalid_code = None if valid else RejectionCode.CAMERA_PAYLOAD_INVALID
    return TimeSeries(
        StampedValue(
            int(stamp_ms * MS),
            Payload(data={"t_ms": stamp_ms}, valid=valid, invalid_code=invalid_code),
        )
        for stamp_ms in stamps_ms
    )


class GridTests(unittest.TestCase):
    def test_inclusive_10_hz_grid(self) -> None:
        self.assertEqual(build_grid(0, 200 * MS), [0, 100 * MS, 200 * MS])

    def test_partial_final_period_is_excluded(self) -> None:
        self.assertEqual(build_grid(0, 150 * MS), [0, 100 * MS])

    def test_empty_when_end_precedes_start(self) -> None:
        self.assertEqual(build_grid(100 * MS, 0), [])

    def test_rejects_non_positive_period(self) -> None:
        with self.assertRaises(ValueError):
            build_grid(0, 100 * MS, 0)


class CausalAssociationTests(unittest.TestCase):
    """Odometry, goal, and action must never read a future value."""

    def test_latest_at_or_before_ignores_future_values(self) -> None:
        stream = series(0, 100, 200)
        selection = stream.latest_at_or_before(150 * MS)
        assert selection is not None
        self.assertEqual(selection.sample.timestamp_ns, 100 * MS)
        self.assertEqual(selection.signed_delta_ns, -50 * MS)
        self.assertFalse(selection.is_from_future)

    def test_latest_at_or_before_is_none_without_prior_value(self) -> None:
        self.assertIsNone(series(100, 200).latest_at_or_before(50 * MS))

    def test_boundary_is_inclusive(self) -> None:
        selection = series(0, 100).latest_at_or_before(100 * MS)
        assert selection is not None
        self.assertEqual(selection.signed_delta_ns, 0)

    def test_nearest_may_select_a_future_sample(self) -> None:
        selection = series(0, 100).nearest(60 * MS)
        assert selection is not None
        self.assertEqual(selection.sample.timestamp_ns, 100 * MS)
        self.assertTrue(selection.is_from_future)
        self.assertEqual(selection.delta_ns, 40 * MS)

    def test_full_stream_is_searched_without_window_trimming(self) -> None:
        # The bracketing scan sits beyond the camera's last frame; it must still
        # be reachable, which is the boundary bug the timing utilities hit.
        selection = series(0, 100, 205).nearest(200 * MS)
        assert selection is not None
        self.assertEqual(selection.sample.timestamp_ns, 205 * MS)


class AssembleTests(unittest.TestCase):
    def test_aligned_streams_produce_accepted_samples(self) -> None:
        result = assemble_samples(
            camera=series(0, 100, 200),
            lidar=series(10, 110, 210),
            odometry=series(0, 100, 200),
            goal=series(0, 100, 200),
            action=series(0, 100, 200),
        )
        self.assertEqual(len(result.samples), 3)
        self.assertEqual(len(result.accepted), 3)
        self.assertEqual(result.segment_lengths, (3,))
        self.assertEqual(result.rejection_counts(), {})

    def test_lidar_association_uses_camera_timestamp_not_grid_tick(self) -> None:
        # Camera frame lands 20 ms after the tick; the scan sits beside the frame.
        result = assemble_samples(
            camera=series(20),
            lidar=series(25),
            odometry=series(0),
            goal=series(0),
            action=series(0),
        )
        sample = result.samples[0]
        self.assertEqual(sample.observation_timestamp_ns, 20 * MS)
        self.assertEqual(sample.selections["lidar"].signed_delta_ns, 5 * MS)

    def test_signed_lidar_delta_exposes_future_scan_selection(self) -> None:
        result = assemble_samples(
            camera=series(0, 100),
            lidar=series(40, 140),
            odometry=series(0, 100),
            goal=series(0, 100),
            action=series(0, 100),
        )
        deltas = [
            sample.selections["lidar"].signed_delta_ns for sample in result.accepted
        ]
        self.assertEqual(deltas, [40 * MS, 40 * MS])
        self.assertTrue(all(delta > 0 for delta in deltas))

    def test_stale_lidar_is_rejected_at_the_75_ms_contract(self) -> None:
        result = assemble_samples(
            camera=series(0),
            lidar=series(80),
            odometry=series(0),
            goal=series(0),
            action=series(0),
        )
        sample = result.samples[0]
        self.assertFalse(sample.accepted)
        self.assertIn(RejectionCode.LIDAR_STALE, sample.rejection_codes)
        self.assertFalse(sample.selections["lidar"].eligible)

    def test_missing_and_stale_are_distinct_codes(self) -> None:
        missing = assemble_samples(
            camera=series(0), lidar=series(0), odometry=None, goal=series(0), action=series(0)
        ).samples[0]
        self.assertIn(RejectionCode.ODOM_MISSING, missing.rejection_codes)

        stale = assemble_samples(
            camera=series(500),
            lidar=series(500),
            odometry=series(0),
            goal=series(500),
            action=series(500),
        ).samples[0]
        self.assertIn(RejectionCode.ODOM_STALE, stale.rejection_codes)

    def test_invalid_payload_is_reported(self) -> None:
        result = assemble_samples(
            camera=series(0, valid=False),
            lidar=series(0),
            odometry=series(0),
            goal=series(0),
            action=series(0),
        )
        self.assertIn(
            RejectionCode.CAMERA_PAYLOAD_INVALID, result.samples[0].rejection_codes
        )

    def test_duplicate_camera_frame_kept_by_closest_tick(self) -> None:
        # A 300 ms camera gap leaves two interior ticks reaching for the same frames.
        result = assemble_samples(
            camera=series(0, 300),
            lidar=series(0, 100, 200, 300),
            odometry=series(0, 100, 200, 300),
            goal=series(0, 100, 200, 300),
            action=series(0, 100, 200, 300),
        )
        self.assertEqual(len(result.samples), 4)
        duplicated = [
            sample.grid_index
            for sample in result.samples
            if RejectionCode.DUPLICATE_CAMERA_FRAME in sample.rejection_codes
        ]
        self.assertEqual(duplicated, [1, 2])
        anchors = sorted(sample.grid_index for sample in result.samples if sample.accepted)
        self.assertEqual(anchors, [0, 3])

    def test_rejected_tick_breaks_the_contiguous_segment(self) -> None:
        result = assemble_samples(
            camera=series(0, 100, 200, 300),
            lidar=series(0, 100, 300),  # nothing near the 200 ms frame
            odometry=series(0, 100, 200, 300),
            goal=series(0, 100, 200, 300),
            action=series(0, 100, 200, 300),
        )
        self.assertEqual(result.segment_lengths, (2, 1))
        self.assertEqual(
            [sample.segment_id for sample in result.samples], [0, 0, None, 1]
        )

    def test_windowable_count_requires_context_and_horizon_in_one_segment(self) -> None:
        result = assemble_samples(
            camera=series(*[index * 100 for index in range(10)]),
            lidar=series(*[index * 100 for index in range(10)]),
            odometry=series(*[index * 100 for index in range(10)]),
            goal=series(*[index * 100 for index in range(10)]),
            action=series(*[index * 100 for index in range(10)]),
        )
        self.assertEqual(result.segment_lengths, (10,))
        # K=8 context plus H=8 horizon needs 15 consecutive samples.
        self.assertEqual(result.windowable_count(context_k=8, horizon_h=8), 0)
        self.assertEqual(result.windowable_count(context_k=1, horizon_h=8), 3)


class ActionPolicyTests(unittest.TestCase):
    """The stationary bag must yield no action-valid samples, yet stay usable."""

    def test_policy_view_rejects_samples_without_a_command(self) -> None:
        result = assemble_samples(
            camera=series(0, 100),
            lidar=series(0, 100),
            odometry=series(0, 100),
            goal=series(0, 100),
            action=None,
            require_action=True,
        )
        self.assertEqual(len(result.accepted), 0)
        self.assertEqual(
            result.rejection_counts(), {RejectionCode.ACTION_MISSING.value: 2}
        )

    def test_sensor_view_keeps_samples_and_records_advisories(self) -> None:
        result = assemble_samples(
            camera=series(0, 100),
            lidar=series(0, 100),
            odometry=series(0, 100),
            goal=series(0, 100),
            action=None,
            require_action=False,
        )
        self.assertEqual(len(result.accepted), 2)
        self.assertEqual(
            result.advisory_counts(), {RejectionCode.ACTION_MISSING.value: 2}
        )

    def test_zero_order_hold_never_reads_a_future_command(self) -> None:
        result = assemble_samples(
            camera=series(100),
            lidar=series(100),
            odometry=series(100),
            goal=series(100),
            action=series(60, 140),
        )
        selection = result.samples[0].selections["action"]
        self.assertEqual(selection.source_timestamp_ns, 60 * MS)
        self.assertEqual(selection.signed_delta_ns, -40 * MS)

    def test_command_older_than_the_staleness_bound_is_stale_not_missing(self) -> None:
        result = assemble_samples(
            camera=series(400),
            lidar=series(400),
            odometry=series(400),
            goal=series(400),
            action=series(0),
        )
        codes = result.samples[0].rejection_codes
        self.assertIn(RejectionCode.ACTION_STALE, codes)
        self.assertNotIn(RejectionCode.ACTION_MISSING, codes)


class LidarCausalModeTests(unittest.TestCase):
    def test_causal_mode_declines_the_future_scan(self) -> None:
        # The later scan is strictly closer, so nearest reaches forward for it.
        nearest = assemble_samples(
            camera=series(100),
            lidar=series(55, 140),
            odometry=series(100),
            goal=series(100),
            action=series(100),
        ).samples[0]
        self.assertEqual(nearest.selections["lidar"].source_timestamp_ns, 140 * MS)
        self.assertTrue(nearest.selections["lidar"].signed_delta_ns > 0)

        causal = assemble_samples(
            camera=series(100),
            lidar=series(55, 140),
            odometry=series(100),
            goal=series(100),
            action=series(100),
            lidar_causal=True,
        ).samples[0]
        self.assertEqual(causal.selections["lidar"].source_timestamp_ns, 55 * MS)
        self.assertTrue(causal.selections["lidar"].signed_delta_ns < 0)

    def test_nearest_prefers_the_earlier_scan_on_an_exact_tie(self) -> None:
        tied = assemble_samples(
            camera=series(100),
            lidar=series(60, 140),
            odometry=series(100),
            goal=series(100),
            action=series(100),
        ).samples[0]
        self.assertEqual(tied.selections["lidar"].source_timestamp_ns, 60 * MS)


class RejectionCodeTests(unittest.TestCase):
    def test_primary_reason_is_the_most_diagnostic_one(self) -> None:
        self.assertEqual(
            primary_rejection([RejectionCode.ODOM_STALE, RejectionCode.CAMERA_MISSING]),
            RejectionCode.CAMERA_MISSING,
        )

    def test_primary_reason_is_order_independent(self) -> None:
        codes = [RejectionCode.LIDAR_STALE, RejectionCode.ACTION_MISSING]
        self.assertEqual(primary_rejection(codes), primary_rejection(list(reversed(codes))))

    def test_no_codes_means_no_primary_reason(self) -> None:
        self.assertIsNone(primary_rejection([]))

    def test_run_level_codes_apply_to_every_sample(self) -> None:
        result = assemble_samples(
            camera=series(0, 100),
            lidar=series(0, 100),
            odometry=series(0, 100),
            goal=series(0, 100),
            action=series(0, 100),
            run_level_codes=(RejectionCode.CALIBRATION_MISMATCH,),
        )
        self.assertEqual(len(result.accepted), 0)
        self.assertEqual(
            result.rejection_counts(), {RejectionCode.CALIBRATION_MISMATCH.value: 2}
        )


class PolicyManifestTests(unittest.TestCase):
    def test_policy_serializes_thresholds_and_timestamp_sources(self) -> None:
        manifest = ExportPolicy().as_manifest()
        streams = manifest["streams"]
        assert isinstance(streams, dict)
        self.assertEqual(streams["lidar"]["max_delta_ms"], 75.0)
        self.assertEqual(streams["action"]["policy"], "zero_order_hold")
        self.assertEqual(streams["action"]["max_delta_ms"], 150.0)
        self.assertEqual(streams["action"]["timestamp_source"], "bag_receive_timestamp")
        self.assertEqual(streams["odometry"]["policy"], "latest_at_or_before")
        self.assertEqual(manifest["grid_rate_hz"], 10.0)


if __name__ == "__main__":
    unittest.main()
