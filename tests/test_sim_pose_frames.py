"""Generated-world/Gazebo odometry frame contract."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "livifuser_sim"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_sim.pose_frames import compose_world_pose  # noqa: E402


class TestPoseFrames(unittest.TestCase):
    def test_identity_when_world_has_no_generated_start(self):
        self.assertEqual(compose_world_pose(None, (1.0, 2.0, 0.3)), (1.0, 2.0, 0.3))

    def test_zero_odometry_maps_to_generated_start(self):
        start = (0.4, -0.2, 0.7)
        actual = compose_world_pose(start, (0.0, 0.0, 0.0))
        for value, expected in zip(actual, start, strict=True):
            self.assertAlmostEqual(value, expected)

    def test_translation_is_rotated_by_start_yaw(self):
        actual = compose_world_pose((1.0, 2.0, math.pi / 2.0), (0.5, 0.0, 0.0))
        self.assertAlmostEqual(actual[0], 1.0)
        self.assertAlmostEqual(actual[1], 2.5)
        self.assertAlmostEqual(actual[2], math.pi / 2.0)

    def test_yaw_wraps_to_principal_interval(self):
        actual = compose_world_pose((0.0, 0.0, 3.0), (0.0, 0.0, 1.0))
        self.assertAlmostEqual(actual[2], 4.0 - 2.0 * math.pi)


if __name__ == "__main__":
    unittest.main()
