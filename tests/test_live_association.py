from __future__ import annotations

import unittest

from livifuser_nav.live_association import LiveAssociator


class LiveAssociatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.associator = LiveAssociator()

    def push(self, stream: str, stamp_ns: int, payload: str | None = None):
        return self.associator.push(
            stream,
            stamp_ns=stamp_ns,
            arrival_monotonic_ns=stamp_ns + 1,
            payload=payload or f"{stream}-{stamp_ns}",
        )

    def test_exact_selection_and_scan_tie_break(self) -> None:
        self.push("rgb", 1_000_000_000)
        self.push("scan", 950_000_000, "earlier")
        self.push("scan", 1_050_000_000, "later")
        self.push("odometry", 925_000_000)
        self.push("goal", 900_000_000)
        result = self.associator.select(1_100_000_000)
        self.assertTrue(result.accepted)
        self.assertEqual(result.context.scan.payload, "earlier")
        self.assertEqual(result.context.odometry.stamp_ns, 925_000_000)
        self.assertEqual(result.context.goal.stamp_ns, 900_000_000)

    def test_newest_unused_rgb_is_selected(self) -> None:
        self.push("rgb", 1_000_000_000)
        self.push("rgb", 1_020_000_000)
        self.push("scan", 1_020_000_000)
        self.push("odometry", 1_020_000_000)
        self.push("goal", 1_020_000_000)
        result = self.associator.select(1_100_000_000)
        self.assertTrue(result.accepted)
        self.assertEqual(result.context.rgb.stamp_ns, 1_020_000_000)

    def test_no_future_rgb_and_missing_input_resets_everything(self) -> None:
        self.push("rgb", 2_000_000_000)
        result = self.associator.select(1_900_000_000)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "rgb_missing")
        self.assertEqual(self.associator.reset_count, 1)

    def test_duplicate_and_regression_are_integrity_failures(self) -> None:
        self.push("scan", 1_000)
        duplicate = self.push("scan", 1_000)
        self.assertFalse(duplicate.accepted)
        self.assertEqual(duplicate.reason, "scan_duplicate")
        self.push("goal", 2_000)
        regression = self.push("goal", 1_999)
        self.assertFalse(regression.accepted)
        self.assertEqual(regression.reason, "goal_regression")

    def test_staleness_clears_state(self) -> None:
        self.push("rgb", 1_000_000_000)
        self.push("scan", 900_000_000)
        self.push("odometry", 1_000_000_000)
        self.push("goal", 1_000_000_000)
        result = self.associator.select(1_000_000_001)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "scan_stale")

    def test_tick_regression_is_rejected(self) -> None:
        self.push("rgb", 1_000)
        self.push("scan", 1_000)
        self.push("odometry", 1_000)
        self.push("goal", 1_000)
        self.assertTrue(self.associator.select(2_000).accepted)
        resets = self.associator.reset_count
        result = self.associator.select(2_000)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "clock_regression_or_duplicate_tick")
        self.assertEqual(self.associator.reset_count, resets)


if __name__ == "__main__":
    unittest.main()
