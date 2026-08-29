"""Pin the live runner's /tf_static QoS against a real pod failure.

The a7 development smoke produced 8,904 proposals over 890 s of sim time and
zero terminals. Every status was startup_calibration_not_verified. The bag
contained all three required static transforms and matching camera_info; the
runner's TRANSIENT_LOCAL subscription used depth 1, so two of the three
latched publishers were dropped and calibration never completed.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "ros2_ws"
    / "src"
    / "livifuser_sim"
    / "livifuser_sim"
    / "live_policy_runner_node.py"
)
REQUIRED_FRAMES = (
    "base_link->base_scan",
    "base_scan->camera",
    "camera->camera_optical_frame",
)


class LivePolicyRunnerStaticTfQosTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8")

    def test_static_tf_subscription_depth_covers_all_three_publishers(self) -> None:
        self.assertIn("QoSProfile(depth=10)", self.source)
        self.assertNotIn("QoSProfile(depth=1)", self.source)
        self.assertIn("DurabilityPolicy.TRANSIENT_LOCAL", self.source)
        for frame in REQUIRED_FRAMES:
            self.assertIn(f'"{frame}"', self.source)

    def test_duplicate_sim_time_ticks_are_ignored(self) -> None:
        self.assertIn("tick_ns <= self._last_tick_ns", self.source)
        self.assertIn("def _claim_stamp", self.source)
        self.assertIn("now().nanoseconds <= 0", self.source)

    def test_control_loop_waits_for_calibration_on_wall_time(self) -> None:
        self.assertIn("_on_wait_for_calibration", self.source)
        self.assertIn("ClockType.STEADY_TIME", self.source)
        self.assertIn("_warmup_loaded_models", self.source)
        self.assertIn("starting 10 Hz control loop", self.source)
        self.assertIn("camera_info mismatch", self.source)

    def test_stale_watchdog_is_disarmed_until_inference_ready(self) -> None:
        supervisor = (
            ROOT
            / "ros2_ws"
            / "src"
            / "livifuser_sim"
            / "livifuser_sim"
            / "simulation_supervisor_node.py"
        ).read_text(encoding="utf-8")
        self.assertIn("_stale_watchdog_armed", supervisor)
        self.assertIn("if message.inference_ready:", supervisor)


if __name__ == "__main__":
    unittest.main()
