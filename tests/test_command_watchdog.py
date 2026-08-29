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

from livifuser_command_watchdog.policy import (  # noqa: E402
    DecisionReason,
    GraphCache,
    VelocityIntent,
    WatchdogLimits,
    apply_graph_cache,
    count_external_publishers,
    decide_command,
    intent_from_message_fields,
    publisher_conflict_decision,
    stale_detection_bound_ms,
)


class CommandWatchdogPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = WatchdogLimits(
            timeout_s=0.25,
            max_abs_linear_mps=0.10,
            max_abs_angular_radps=0.50,
        )

    def test_missing_intent_is_zero(self) -> None:
        decision = decide_command(None, None, 10.0, self.limits)
        self.assertEqual(decision.reason, DecisionReason.MISSING)
        self.assertEqual(decision.output, VelocityIntent(0.0, 0.0))
        self.assertFalse(decision.intent_present)
        self.assertEqual(decision.intent_age_s, -1.0)

    def test_fresh_intent_passes_through(self) -> None:
        intent = VelocityIntent(0.05, -0.2)
        decision = decide_command(intent, 10.0, 10.1, self.limits)
        self.assertEqual(decision.reason, DecisionReason.FRESH)
        self.assertEqual(decision.output, intent)
        self.assertFalse(decision.clamped)
        self.assertAlmostEqual(decision.intent_age_s, 0.1)

    def test_positive_and_negative_limits_are_clamped(self) -> None:
        positive = decide_command(VelocityIntent(0.4, 1.2), 2.0, 2.01, self.limits)
        negative = decide_command(VelocityIntent(-0.4, -1.2), 2.0, 2.01, self.limits)
        self.assertEqual(positive.output, VelocityIntent(0.10, 0.50))
        self.assertEqual(negative.output, VelocityIntent(-0.10, -0.50))
        self.assertTrue(positive.clamped)
        self.assertTrue(negative.clamped)

    def test_timeout_boundary_is_stale_and_zero(self) -> None:
        decision = decide_command(VelocityIntent(0.05, 0.0), 1.0, 1.25, self.limits)
        self.assertEqual(decision.reason, DecisionReason.STALE)
        self.assertEqual(decision.output, VelocityIntent(0.0, 0.0))

    def test_nonfinite_intent_is_invalid_and_zero(self) -> None:
        for bad_value in (math.nan, math.inf, -math.inf):
            with self.subTest(bad_value=bad_value):
                decision = decide_command(
                    VelocityIntent(bad_value, 0.0),
                    1.0,
                    1.01,
                    self.limits,
                )
                self.assertEqual(decision.reason, DecisionReason.INVALID)
                self.assertEqual(decision.output, VelocityIntent(0.0, 0.0))

    def test_structurally_invalid_intent_is_zero(self) -> None:
        decision = decide_command(
            VelocityIntent(0.05, 0.0, structurally_valid=False),
            1.0,
            1.01,
            self.limits,
        )
        self.assertEqual(decision.reason, DecisionReason.INVALID)
        self.assertEqual(decision.output, VelocityIntent(0.0, 0.0))

    def test_monotonic_clock_regression_is_zero(self) -> None:
        decision = decide_command(VelocityIntent(0.05, 0.0), 2.0, 1.9, self.limits)
        self.assertEqual(decision.reason, DecisionReason.CLOCK_REGRESSION)
        self.assertEqual(decision.output, VelocityIntent(0.0, 0.0))

    def test_publisher_conflict_overrides_fresh_command(self) -> None:
        fresh = decide_command(VelocityIntent(0.05, 0.1), 1.0, 1.01, self.limits)
        conflict = publisher_conflict_decision(fresh)
        self.assertEqual(conflict.reason, DecisionReason.PUBLISHER_CONFLICT)
        self.assertEqual(conflict.output, VelocityIntent(0.0, 0.0))
        self.assertEqual(conflict.requested, fresh.requested)

    def test_message_boundary_requires_stamp_frame_and_planar_axes(self) -> None:
        base = {
            "linear_x": 0.05,
            "linear_y": 0.0,
            "linear_z": 0.0,
            "angular_x": 0.0,
            "angular_y": 0.0,
            "angular_z": 0.2,
            "frame_id": "base_link",
            "stamp_is_set": True,
            "expected_frame_id": "base_link",
        }
        self.assertTrue(intent_from_message_fields(**base).is_valid)
        for change in (
            {"frame_id": "odom"},
            {"stamp_is_set": False},
            {"linear_y": 0.01},
            {"linear_z": math.nan},
            {"angular_x": 0.01},
            {"angular_y": -0.01},
        ):
            with self.subTest(change=change):
                fields = {**base, **change}
                self.assertFalse(intent_from_message_fields(**fields).is_valid)

    def test_external_publisher_count_exempts_exactly_one_own_endpoint(self) -> None:
        own = ("livifuser_command_watchdog", "/")
        self.assertEqual(count_external_publishers([], own), 0)
        self.assertEqual(count_external_publishers([own], own), 0)
        self.assertEqual(count_external_publishers([own, ("teleop", "/")], own), 1)
        self.assertEqual(count_external_publishers([own, own], own), 1)
        self.assertEqual(count_external_publishers([("teleop", "/")], own), 1)

    def test_invalid_limits_are_rejected(self) -> None:
        for values in ((0.0, 0.1, 0.5), (0.25, -0.1, 0.5), (0.25, 0.1, math.nan)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                WatchdogLimits(*values)

    def test_stale_detection_bound_adds_timeout_and_accepted_tick_gap(self) -> None:
        self.assertEqual(stale_detection_bound_ms(250.0, 150.0), 400.0)

    def test_stale_detection_bound_rejects_invalid_inputs(self) -> None:
        for values in ((0.0, 150.0), (250.0, -1.0), (math.nan, 150.0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                stale_detection_bound_ms(*values)


class GraphCachePolicyTests(unittest.TestCase):
    """The command path consumes a cached publisher picture, never a live query."""

    def setUp(self) -> None:
        self.limits = WatchdogLimits()
        self.fresh = decide_command(VelocityIntent(0.08, 0.0), 10.0, 10.01, self.limits)
        self.assertIs(self.fresh.reason, DecisionReason.FRESH)

    def test_never_probed_forces_zero(self) -> None:
        decision = apply_graph_cache(self.fresh, GraphCache(), 10.01, 3.0)
        self.assertIs(decision.reason, DecisionReason.GRAPH_UNKNOWN)
        self.assertEqual(decision.output.linear_mps, 0.0)
        self.assertEqual(decision.output.angular_radps, 0.0)

    def test_half_populated_cache_forces_zero(self) -> None:
        for cache in (GraphCache(0, None), GraphCache(None, 10.0)):
            with self.subTest(cache=cache):
                decision = apply_graph_cache(self.fresh, cache, 10.01, 3.0)
                self.assertIs(decision.reason, DecisionReason.GRAPH_UNKNOWN)

    def test_stale_cache_forces_zero(self) -> None:
        decision = apply_graph_cache(self.fresh, GraphCache(0, 1.0), 10.01, 3.0)
        self.assertIs(decision.reason, DecisionReason.GRAPH_STALE)
        self.assertEqual(decision.output.linear_mps, 0.0)

    def test_cache_age_boundary_is_inclusive(self) -> None:
        exactly_at_limit = apply_graph_cache(self.fresh, GraphCache(0, 7.01), 10.01, 3.0)
        self.assertIs(exactly_at_limit.reason, DecisionReason.FRESH)
        just_past = apply_graph_cache(self.fresh, GraphCache(0, 7.0), 10.01, 3.0)
        self.assertIs(just_past.reason, DecisionReason.GRAPH_STALE)

    def test_probe_timestamp_from_the_future_forces_zero(self) -> None:
        decision = apply_graph_cache(self.fresh, GraphCache(0, 99.0), 10.01, 3.0)
        self.assertIs(decision.reason, DecisionReason.GRAPH_STALE)

    def test_nonfinite_probe_timestamp_forces_zero(self) -> None:
        decision = apply_graph_cache(self.fresh, GraphCache(0, math.nan), 10.01, 3.0)
        self.assertIs(decision.reason, DecisionReason.GRAPH_STALE)

    def test_fresh_cache_without_conflict_passes_the_decision_through(self) -> None:
        decision = apply_graph_cache(self.fresh, GraphCache(0, 10.0), 10.01, 3.0)
        self.assertIs(decision, self.fresh)

    def test_fresh_cache_with_conflict_forces_zero(self) -> None:
        decision = apply_graph_cache(self.fresh, GraphCache(1, 10.0), 10.01, 3.0)
        self.assertIs(decision.reason, DecisionReason.PUBLISHER_CONFLICT)
        self.assertEqual(decision.output.linear_mps, 0.0)

    def test_unknown_cache_outranks_an_already_zero_decision(self) -> None:
        stale_intent = decide_command(VelocityIntent(0.08, 0.0), 1.0, 10.0, self.limits)
        self.assertIs(stale_intent.reason, DecisionReason.STALE)
        decision = apply_graph_cache(stale_intent, GraphCache(), 10.0, 3.0)
        self.assertIs(decision.reason, DecisionReason.GRAPH_UNKNOWN)
        self.assertEqual(decision.requested.linear_mps, 0.08)


if __name__ == "__main__":
    unittest.main()
