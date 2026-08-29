"""Unit tests for the read-only live overlay used in the teleop demonstration.

The two properties worth pinning are that the K-step window never bridges a gap
and that the overlay cannot construct a command publisher. Everything else here
protects the geometry the dashboard draws.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from livifuser_nav.learning_data import tokenize_lidar
from livifuser_nav.live_overlay import (
    TICK_PERIOD_S,
    VISUAL_TOKENS,
    VISUAL_WIDTH,
    LiveWindow,
    OverlayConfig,
    agreement_error,
    assert_no_command_publishers,
    assert_publishers_are_inert,
    calibration_context,
    goal_xy,
    rollout_unicycle,
    scan_points,
    sigma_from_log_variance,
    tokenize_live_scan,
)
from livifuser_nav.replay_safety import SafetyAuditError

from .test_learning_data import manifest_for

CONFIG = OverlayConfig(context_k=8, horizon_h=8, lidar_sectors=80)
CONTEXT = {"calibration": manifest_for("live_test")["calibration"], "run_id": "live_test"}


def scan_of(beam_count: int, value: float = 1.5) -> np.ndarray:
    return np.full(beam_count, value, dtype=np.float64)


def tokens_for(beam_count: int = 400, value: float = 1.5):
    return tokenize_live_scan(
        scan_of(beam_count, value), 2.0 * math.pi / (beam_count + 1), CONTEXT, CONFIG
    )


def fill(window: LiveWindow, ticks: int, *, start: int = 0, step: float = TICK_PERIOD_S):
    tokens = tokens_for()
    ready = False
    for index in range(ticks):
        ready = window.push(
            timestamp_s=(start + index) * step,
            tokens=tokens,
            goal=(1.0, 0.0, 1.0),
            robot_state=(0.05, 0.0),
        )
    return ready


class LiveWindowTest(unittest.TestCase):
    def test_window_is_not_ready_until_k_ticks_have_arrived(self) -> None:
        window = LiveWindow(CONFIG)
        for index in range(CONFIG.context_k - 1):
            self.assertFalse(fill(window, 1, start=index))
        self.assertTrue(fill(window, 1, start=CONFIG.context_k - 1))
        self.assertEqual(window.depth, CONFIG.context_k)

    def test_window_holds_at_k_and_slides(self) -> None:
        window = LiveWindow(CONFIG)
        fill(window, 40)
        self.assertEqual(window.depth, CONFIG.context_k)
        self.assertEqual(window.resets, 0)

    def test_a_dropped_tick_resets_rather_than_bridging(self) -> None:
        window = LiveWindow(CONFIG)
        fill(window, CONFIG.context_k)
        self.assertTrue(window.ready)
        # One grid period late is jitter; three is a dropped observation.
        window.push(
            timestamp_s=CONFIG.context_k * TICK_PERIOD_S + 0.3,
            tokens=tokens_for(),
            goal=(1.0, 0.0, 1.0),
            robot_state=(0.0, 0.0),
        )
        self.assertFalse(window.ready)
        self.assertEqual(window.depth, 1)
        self.assertEqual(window.resets, 1)

    def test_jitter_inside_the_tolerance_does_not_reset(self) -> None:
        window = LiveWindow(CONFIG)
        fill(window, CONFIG.context_k, step=0.12)
        self.assertTrue(window.ready)
        self.assertEqual(window.resets, 0)

    def test_a_backwards_timestamp_resets(self) -> None:
        window = LiveWindow(CONFIG)
        fill(window, CONFIG.context_k)
        window.push(
            timestamp_s=0.0,
            tokens=tokens_for(),
            goal=(1.0, 0.0, 1.0),
            robot_state=(0.0, 0.0),
        )
        self.assertEqual(window.resets, 1)
        self.assertEqual(window.depth, 1)

    def test_a_non_finite_timestamp_is_refused(self) -> None:
        window = LiveWindow(CONFIG)
        with self.assertRaises(ValueError):
            window.push(
                timestamp_s=float("nan"),
                tokens=tokens_for(),
                goal=(1.0, 0.0, 1.0),
                robot_state=(0.0, 0.0),
            )

    def test_arrays_match_the_model_input_contract(self) -> None:
        window = LiveWindow(CONFIG)
        fill(window, CONFIG.context_k)
        arrays = window.arrays()
        k, sectors = CONFIG.context_k, CONFIG.lidar_sectors
        self.assertEqual(arrays["visual_tokens"].shape, (1, k, VISUAL_TOKENS, VISUAL_WIDTH))
        self.assertEqual(arrays["lidar_features"].shape, (1, k, sectors, 4))
        self.assertEqual(arrays["visual_mask"].shape, (1, k, sectors, VISUAL_TOKENS))
        self.assertEqual(arrays["in_fov"].shape, (1, k, sectors))
        self.assertEqual(arrays["goal"].shape, (1, k, 3))
        self.assertEqual(arrays["robot_state"].shape, (1, k, 2))
        self.assertEqual(arrays["lidar_features"].dtype, np.float32)
        self.assertEqual(arrays["visual_mask"].dtype, np.bool_)
        self.assertEqual(arrays["in_fov"].dtype, np.bool_)

    def test_arrays_are_refused_before_the_window_fills(self) -> None:
        window = LiveWindow(CONFIG)
        fill(window, CONFIG.context_k - 1)
        with self.assertRaises(ValueError):
            window.arrays()

    def test_the_window_keeps_the_most_recent_ticks_in_order(self) -> None:
        window = LiveWindow(CONFIG)
        for index in range(CONFIG.context_k + 4):
            window.push(
                timestamp_s=index * TICK_PERIOD_S,
                tokens=tokens_for(),
                goal=(float(index), 0.0, 1.0),
                robot_state=(0.0, 0.0),
            )
        goals = window.arrays()["goal"][0, :, 0]
        expected = np.arange(4, CONFIG.context_k + 4, dtype=np.float32)
        np.testing.assert_allclose(goals, expected)


class TokenizationTest(unittest.TestCase):
    def test_live_tokenization_matches_the_export_path(self) -> None:
        beams = 401
        ranges = scan_of(beams, 2.0)
        increment = 2.0 * math.pi / (beams + 1)
        live = tokenize_live_scan(ranges, increment, CONTEXT, CONFIG)
        recorded = tokenize_lidar(
            ranges,
            beams,
            increment,
            CONTEXT,
            sectors=CONFIG.lidar_sectors,
            range_clip_m=CONFIG.lidar_range_clip_m,
            visual_radius=CONFIG.visual_mask_radius_tokens,
        )
        np.testing.assert_array_equal(live.features, recorded.features)
        np.testing.assert_array_equal(live.in_fov, recorded.in_fov)
        np.testing.assert_array_equal(live.visual_mask, recorded.visual_mask)

    def test_an_empty_scan_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            tokenize_live_scan(np.asarray([]), 0.0157, CONTEXT, CONFIG)

    def test_some_sectors_fall_inside_the_calibrated_camera_field(self) -> None:
        tokens = tokens_for(400, 1.2)
        self.assertTrue(0 < int(np.count_nonzero(tokens.in_fov)) < CONFIG.lidar_sectors)


class ScanGeometryTest(unittest.TestCase):
    def test_invalid_returns_are_dropped(self) -> None:
        ranges = np.asarray([1.0, np.nan, 0.01, 200.0, 2.0])
        xs, ys = scan_points(ranges, 2.0 * math.pi / 6, CONTEXT)
        self.assertEqual(xs.shape, (2,))
        self.assertEqual(ys.shape, (2,))

    def test_bearings_use_the_supplied_increment(self) -> None:
        # Beam one of a four-beam scan sits a quarter turn round from zero.
        xs, ys = scan_points(np.asarray([1.0, 1.0]), math.pi / 2, CONTEXT)
        np.testing.assert_allclose(xs, [1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(ys, [0.0, 1.0], atol=1e-9)


class RolloutTest(unittest.TestCase):
    def test_a_straight_command_integrates_along_x(self) -> None:
        poses = rollout_unicycle(np.tile([0.1, 0.0], (8, 1)))
        self.assertEqual(poses.shape, (9, 3))
        np.testing.assert_allclose(poses[-1, 0], 0.08, atol=1e-9)
        np.testing.assert_allclose(poses[-1, 1], 0.0, atol=1e-9)

    def test_a_pure_rotation_stays_at_the_origin(self) -> None:
        poses = rollout_unicycle(np.tile([0.0, 1.0], (8, 1)))
        np.testing.assert_allclose(poses[:, :2], 0.0, atol=1e-12)
        np.testing.assert_allclose(poses[-1, 2], 0.8, atol=1e-9)

    def test_a_left_turn_curves_to_positive_y(self) -> None:
        poses = rollout_unicycle(np.tile([0.2, 1.0], (8, 1)))
        self.assertGreater(poses[-1, 1], 0.0)

    def test_the_shape_contract_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            rollout_unicycle(np.zeros((8, 3)))


class GoalTest(unittest.TestCase):
    def test_a_forward_goal_lies_on_the_x_axis(self) -> None:
        x, y = goal_xy((1.5, 0.0, 1.0))
        self.assertAlmostEqual(x, 1.5)
        self.assertAlmostEqual(y, 0.0)

    def test_a_left_goal_lies_on_the_y_axis(self) -> None:
        x, y = goal_xy((2.0, 1.0, 0.0))
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 2.0)

    def test_an_unnormalized_bearing_is_normalized(self) -> None:
        x, y = goal_xy((1.0, 3.0, 4.0))
        self.assertAlmostEqual(math.hypot(x, y), 1.0)

    def test_a_degenerate_bearing_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            goal_xy((1.0, 0.0, 0.0))


class UncertaintyTest(unittest.TestCase):
    def test_sigma_is_the_root_of_the_exponentiated_log_variance(self) -> None:
        sigma = sigma_from_log_variance(np.asarray([[0.0, math.log(4.0)]]))
        np.testing.assert_allclose(sigma, [[1.0, 2.0]], atol=1e-12)


class AgreementTest(unittest.TestCase):
    def test_only_the_first_horizon_step_is_scored(self) -> None:
        predicted = np.asarray([[0.20, 0.50], [9.0, 9.0], [9.0, 9.0]])
        errors = agreement_error(predicted, (0.05, 0.10))
        self.assertAlmostEqual(errors["linear_error_mps"], 0.15)
        self.assertAlmostEqual(errors["angular_error_radps"], 0.40)

    def test_the_shape_contract_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            agreement_error(np.zeros(2), (0.0, 0.0))


class SafetyTest(unittest.TestCase):
    def test_sensor_topics_are_accepted(self) -> None:
        assert_no_command_publishers(["/scan", "/odom", "/livifuser/goal_relative"])

    def test_a_command_topic_is_refused(self) -> None:
        with self.assertRaises(SafetyAuditError):
            assert_no_command_publishers(["/scan", "/cmd_vel"])

    def test_a_stamped_command_topic_is_refused(self) -> None:
        with self.assertRaises(SafetyAuditError):
            assert_no_command_publishers(["/livifuser/cmd_vel_stamped"])

    def test_rclpy_infrastructure_publishers_are_accepted(self) -> None:
        assert_publishers_are_inert(["/rosout", "/parameter_events"])

    def test_no_publishers_at_all_is_accepted(self) -> None:
        assert_publishers_are_inert([])

    def test_an_innocent_looking_extra_publisher_is_refused(self) -> None:
        with self.assertRaises(SafetyAuditError) as raised:
            assert_publishers_are_inert(["/rosout", "/livifuser/overlay_debug"])
        self.assertIn("/livifuser/overlay_debug", str(raised.exception))

    def test_a_command_publisher_is_refused_by_the_pattern_check_first(self) -> None:
        with self.assertRaises(SafetyAuditError) as raised:
            assert_publishers_are_inert(["/rosout", "/cmd_vel"])
        self.assertIn("command topics", str(raised.exception))


class CalibrationContextTest(unittest.TestCase):
    def test_a_manifest_calibration_block_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest_for("run_a")), "utf-8")
            context = calibration_context(path)
        self.assertEqual(context["run_id"], "run_a")
        self.assertIn("lidar_geometry", context["calibration"])

    def test_a_manifest_without_calibration_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"run_id": "run_b"}), "utf-8")
            with self.assertRaises(ValueError):
                calibration_context(path)

    def test_a_partial_calibration_block_is_refused(self) -> None:
        manifest = manifest_for("run_c")
        del manifest["calibration"]["lidar_geometry"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), "utf-8")
            with self.assertRaises(ValueError):
                calibration_context(path)


class OverlayConfigTest(unittest.TestCase):
    def test_the_sweep_config_supplies_the_window_geometry(self) -> None:
        config = OverlayConfig.from_sweep_config(
            {
                "context_k": 8,
                "horizon_h": 8,
                "lidar_sectors": 80,
                "lidar_range_clip_m": 10.0,
                "visual_mask_radius_tokens": 1,
            }
        )
        self.assertEqual(config.context_k, 8)
        self.assertEqual(config.lidar_sectors, 80)
        self.assertAlmostEqual(config.lidar_range_clip_m, 10.0)

    def test_the_shipped_pilot_config_loads(self) -> None:
        path = Path(__file__).resolve().parents[1] / "config" / "baseline_sweep_pilot5_v1.json"
        config = OverlayConfig.from_sweep_config(json.loads(path.read_text("utf-8")))
        self.assertEqual(config.context_k, 8)
        self.assertEqual(config.horizon_h, 8)


if __name__ == "__main__":
    unittest.main()
