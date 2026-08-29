from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "ros2_ws"
    / "src"
    / "livifuser_command_watchdog"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_command_watchdog.episode_policy import (  # noqa: E402
    EpisodeConfig,
    EpisodeIdentity,
    EpisodeLifecycle,
    EpisodePhase,
    GoalReachTracker,
    OperatorIntent,
    StreamObservation,
    StreamRequirement,
    evaluate_readiness,
    gate_operator_intent,
)
from livifuser_command_watchdog.keyboard_policy import (  # noqa: E402
    ZERO_COMMAND,
    KeyboardCommand,
)

MOVING_COMMAND = KeyboardCommand(0.08, 0.40)


class EpisodeIdentityTests(unittest.TestCase):
    def valid(self, **overrides: str) -> EpisodeIdentity:
        values = {
            "episode_id": "train_room_a_001",
            "environment_id": "room_a",
            "split": "train",
            "route_id": "route_straight_3m",
            "layout_id": "box_left",
            "code_revision": "e281fbf",
        }
        values.update(overrides)
        return EpisodeIdentity(**values)

    def test_valid_identity_and_matching_output_are_accepted(self) -> None:
        identity = self.valid()
        identity.validate_output_basename("train_room_a_001")

    def test_split_identifier_revision_and_output_mismatch_are_refused(self) -> None:
        cases = (
            {"split": "val"},
            {"episode_id": "UPPER"},
            {"environment_id": "x"},
            {"code_revision": "not-a-hash"},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.valid(**values)
        with self.assertRaises(ValueError):
            self.valid().validate_output_basename("different_episode")


class ReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requirements = (
            StreamRequirement("/camera/image_raw", 3, 0.2),
            StreamRequirement("/tf_static", 1, None),
        )

    def test_all_declared_streams_must_be_present_fresh_and_populated(self) -> None:
        observations = {
            "/camera/image_raw": StreamObservation(3, 9.9),
            "/tf_static": StreamObservation(1, 1.0),
        }
        self.assertTrue(
            evaluate_readiness(
                self.requirements, observations, now_monotonic_s=10.0
            ).ready
        )

    def test_missing_insufficient_and_stale_reasons_are_explicit(self) -> None:
        missing = evaluate_readiness(self.requirements, {}, now_monotonic_s=10.0)
        self.assertEqual(
            missing.reasons,
            ("stream_missing:/camera/image_raw", "stream_missing:/tf_static"),
        )
        observations = {
            "/camera/image_raw": StreamObservation(2, 9.0),
            "/tf_static": StreamObservation(1, 1.0),
        }
        decision = evaluate_readiness(
            self.requirements, observations, now_monotonic_s=10.0
        )
        self.assertIn("stream_insufficient:/camera/image_raw:2/3", decision.reasons)
        self.assertIn("stream_stale:/camera/image_raw:1.000s", decision.reasons)
        self.assertFalse(any("tf_static" in reason for reason in decision.reasons))

    def test_clock_regression_is_not_misreported_as_fresh(self) -> None:
        observations = {
            "/camera/image_raw": StreamObservation(3, 11.0),
            "/tf_static": StreamObservation(1, 1.0),
        }
        decision = evaluate_readiness(
            self.requirements, observations, now_monotonic_s=10.0
        )
        self.assertIn("stream_clock_invalid:/camera/image_raw", decision.reasons)

    def test_invalid_requirements_are_refused(self) -> None:
        for args in (("scan", 1, 1.0), ("/scan", 0, 1.0), ("/scan", 1, 0.0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                StreamRequirement(*args)


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EpisodeConfig(
            duration_s=45.0,
            zero_warmup_s=2.0,
            zero_cooldown_s=2.0,
        )
        self.lifecycle = EpisodeLifecycle(self.config, now_monotonic_s=100.0)

    def arm(self) -> None:
        self.lifecycle.begin_recorder(now_monotonic_s=101.0)
        self.lifecycle.recorder_ready(now_monotonic_s=102.0)

    def test_recording_starts_only_after_recorder_and_zero_warmup(self) -> None:
        self.arm()
        self.assertEqual(
            self.lifecycle.advance(now_monotonic_s=103.999), EpisodePhase.ZERO_WARMUP
        )
        self.assertEqual(
            self.lifecycle.advance(now_monotonic_s=104.0), EpisodePhase.RECORDING
        )
        self.assertTrue(
            self.lifecycle.snapshot(now_monotonic_s=104.0).motion_permitted
        )

    def test_local_deadline_forces_zero_cooldown_at_exact_boundary(self) -> None:
        self.arm()
        self.lifecycle.advance(now_monotonic_s=104.0)
        self.assertEqual(
            self.lifecycle.advance(now_monotonic_s=148.999), EpisodePhase.RECORDING
        )
        self.assertEqual(
            self.lifecycle.advance(now_monotonic_s=149.0), EpisodePhase.ZERO_COOLDOWN
        )
        snapshot = self.lifecycle.snapshot(now_monotonic_s=149.0)
        self.assertFalse(snapshot.motion_permitted)
        self.assertEqual(snapshot.recording_remaining_s, 0.0)

    def test_delayed_tick_catches_up_without_extending_authorized_duration(self) -> None:
        self.arm()
        self.assertEqual(
            self.lifecycle.advance(now_monotonic_s=151.0), EpisodePhase.COMPLETE
        )
        snapshot = self.lifecycle.snapshot(now_monotonic_s=151.0)
        self.assertEqual(snapshot.recording_elapsed_s, 45.0)
        self.assertFalse(snapshot.motion_permitted)

    def test_goal_stop_enters_zero_cooldown_immediately(self) -> None:
        self.arm()
        self.lifecycle.advance(now_monotonic_s=104.0)
        self.lifecycle.request_stop("goal_reached", now_monotonic_s=110.0)
        self.assertEqual(self.lifecycle.phase, EpisodePhase.ZERO_COOLDOWN)
        self.assertEqual(self.lifecycle.reason, "goal_reached")

    def test_failure_is_terminal_and_disallows_motion(self) -> None:
        self.lifecycle.fail("preflight_timeout", now_monotonic_s=110.0)
        snapshot = self.lifecycle.snapshot(now_monotonic_s=110.0)
        self.assertEqual(snapshot.phase, EpisodePhase.FAILED)
        self.assertFalse(snapshot.motion_permitted)

    def test_clock_regression_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.lifecycle.advance(now_monotonic_s=99.0)

    def test_invalid_config_is_refused(self) -> None:
        for kwargs in (
            {"duration_s": 0.0},
            {"duration_s": 301.0},
            {"linear_mps": 0.11},
            {"angular_radps": 0.51},
            {"operator_timeout_s": math.nan},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EpisodeConfig(**kwargs)


class CommandGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lifecycle = EpisodeLifecycle(EpisodeConfig(), now_monotonic_s=0.0)
        self.lifecycle.begin_recorder(now_monotonic_s=1.0)
        self.lifecycle.recorder_ready(now_monotonic_s=2.0)
        self.lifecycle.advance(now_monotonic_s=4.0)

    def intent(
        self,
        command: KeyboardCommand = MOVING_COMMAND,
        *,
        arrival: float = 4.0,
        valid: bool = True,
    ) -> OperatorIntent:
        return OperatorIntent(command, arrival, valid)

    def test_fresh_forward_commands_pass_only_during_recording(self) -> None:
        decision = gate_operator_intent(
            self.lifecycle, self.intent(), now_monotonic_s=4.1
        )
        self.assertTrue(decision.permitted)
        self.assertEqual(decision.command, KeyboardCommand(0.08, 0.40))
        self.lifecycle.request_stop("operator_stop", now_monotonic_s=4.2)
        stopped = gate_operator_intent(
            self.lifecycle, self.intent(arrival=4.2), now_monotonic_s=4.2
        )
        self.assertEqual(stopped.command, ZERO_COMMAND)

    def test_missing_stale_invalid_and_regressed_intent_fail_to_zero(self) -> None:
        cases = (
            (None, 4.1, "operator_missing"),
            (self.intent(arrival=3.0), 4.1, "operator_stale"),
            (self.intent(arrival=5.0), 4.1, "operator_clock_invalid"),
            (self.intent(valid=False), 4.1, "operator_invalid"),
        )
        for intent, now, reason in cases:
            with self.subTest(reason=reason):
                decision = gate_operator_intent(
                    self.lifecycle, intent, now_monotonic_s=now
                )
                self.assertEqual(decision.command, ZERO_COMMAND)
                self.assertEqual(decision.reason, reason)

    def test_reverse_rotate_and_off_grid_commands_are_rejected(self) -> None:
        commands = (
            KeyboardCommand(-0.08, 0.0),
            KeyboardCommand(0.0, 0.40),
            KeyboardCommand(0.07, 0.0),
            KeyboardCommand(0.08, 0.20),
        )
        for command in commands:
            with self.subTest(command=command):
                decision = gate_operator_intent(
                    self.lifecycle,
                    self.intent(command),
                    now_monotonic_s=4.1,
                )
                self.assertEqual(decision.reason, "operator_command_not_whitelisted")
                self.assertEqual(decision.command, ZERO_COMMAND)

    def test_explicit_zero_is_forwarded_as_zero_without_motion_permission(self) -> None:
        decision = gate_operator_intent(
            self.lifecycle,
            self.intent(ZERO_COMMAND),
            now_monotonic_s=4.1,
        )
        self.assertEqual(decision.command, ZERO_COMMAND)
        self.assertFalse(decision.permitted)
        self.assertEqual(decision.reason, "operator_fresh")


class GoalReachTests(unittest.TestCase):
    def test_goal_requires_three_consecutive_samples(self) -> None:
        tracker = GoalReachTracker(tolerance_m=0.25, required_samples=3)
        self.assertFalse(tracker.update(0.24))
        self.assertFalse(tracker.update(0.20))
        self.assertTrue(tracker.update(0.10))

    def test_outside_or_invalid_sample_resets_the_latch(self) -> None:
        tracker = GoalReachTracker(tolerance_m=0.25, required_samples=2)
        self.assertFalse(tracker.update(0.2))
        self.assertFalse(tracker.update(0.3))
        self.assertFalse(tracker.update(math.nan))
        self.assertFalse(tracker.update(0.1))
        self.assertTrue(tracker.update(0.1))


if __name__ == "__main__":
    unittest.main()
