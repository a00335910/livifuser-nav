import math
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "livifuser_sim"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_sim.expert_policy import reactive_command  # noqa: E402


def scan(default: float = 2.0, count: int = 400):
    angles = [index * 2.0 * math.pi / count for index in range(count)]
    return [default] * count, angles


class TestReactiveExpertPolicy(unittest.TestCase):
    def test_clear_path_tracks_goal_with_locked_limits(self):
        ranges, angles = scan()
        command = reactive_command(ranges, angles, goal_rho_m=2.0, goal_alpha_rad=0.1)
        self.assertEqual(command.reason, "track_goal")
        self.assertGreater(command.linear_mps, 0.0)
        self.assertLessEqual(command.linear_mps, 0.08)
        self.assertAlmostEqual(command.angular_radps, 0.15)

    def test_front_obstacle_turns_toward_clearer_left_side(self):
        ranges, angles = scan()
        for index, angle in enumerate(angles):
            signed = math.atan2(math.sin(angle), math.cos(angle))
            if abs(signed) < 0.2:
                ranges[index] = 0.30
            if -1.2 <= signed <= -0.25:
                ranges[index] = 0.35
        command = reactive_command(ranges, angles, goal_rho_m=2.0, goal_alpha_rad=0.0)
        self.assertEqual(command.reason, "avoid_obstacle")
        self.assertEqual(command.linear_mps, 0.02)
        self.assertEqual(command.angular_radps, 0.40)

    def test_goal_reached_and_invalid_scan_are_zero(self):
        ranges, angles = scan()
        self.assertEqual(
            reactive_command(
                ranges, angles, goal_rho_m=0.10, goal_alpha_rad=0.0
            ).linear_mps,
            0.0,
        )
        self.assertEqual(
            reactive_command([], [], goal_rho_m=1.0, goal_alpha_rad=0.0).reason,
            "scan_invalid",
        )
