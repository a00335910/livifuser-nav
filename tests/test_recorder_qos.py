"""Recorder QoS regression checks for the /cmd_vel backpressure mitigation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMAND_PACKAGE = REPO_ROOT / "ros2_ws" / "src" / "livifuser_command_watchdog"
QOS_OVERRIDE = COMMAND_PACKAGE / "config" / "rosbag_qos_overrides_v1.yaml"
sys.path.insert(0, str(COMMAND_PACKAGE))

from livifuser_command_watchdog.recorder_qos import (  # noqa: E402
    QOS_OVERRIDE_FILENAME,
    build_record_command,
)


class RecorderQosTests(unittest.TestCase):
    def test_override_changes_only_cmd_vel_reliability(self) -> None:
        self.assertEqual(QOS_OVERRIDE.name, QOS_OVERRIDE_FILENAME)
        meaningful_lines = [
            line.strip()
            for line in QOS_OVERRIDE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(
            meaningful_lines,
            [
                "/cmd_vel:",
                "history: keep_last",
                "depth: 10",
                "reliability: best_effort",
            ],
        )

    def test_record_command_requires_versioned_override(self) -> None:
        command = build_record_command(
            storage_id="mcap",
            output_path=Path("/tmp/example"),
            topics=("/cmd_vel", "/odom"),
            qos_override_path=QOS_OVERRIDE,
        )
        flag_index = command.index("--qos-profile-overrides-path")
        self.assertEqual(command[flag_index + 1], str(QOS_OVERRIDE))
        self.assertEqual(command[-2:], ["/cmd_vel", "/odom"])

    def test_both_recorders_apply_the_override(self) -> None:
        launch_source = (
            REPO_ROOT
            / "ros2_ws"
            / "src"
            / "livifuser_bringup"
            / "launch"
            / "record_pilot.launch.py"
        ).read_text(encoding="utf-8")
        episode_source = (
            COMMAND_PACKAGE / "livifuser_command_watchdog" / "episode_node.py"
        ).read_text(encoding="utf-8")
        setup_source = (COMMAND_PACKAGE / "setup.py").read_text(encoding="utf-8")

        self.assertIn("--qos-profile-overrides-path", launch_source)
        self.assertIn(QOS_OVERRIDE_FILENAME, launch_source)
        self.assertIn("build_record_command(", episode_source)
        self.assertIn('OPERATOR_STOP_TOPIC = "/livifuser/operator_stop"', episode_source)
        self.assertIn("OPERATOR_STOP_TOPIC,", episode_source)
        self.assertIn(f'"config/{QOS_OVERRIDE_FILENAME}"', setup_source)


if __name__ == "__main__":
    unittest.main()
