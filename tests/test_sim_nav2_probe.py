"""Pure safety helper checks for the Nav2 command relay."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"
sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from livifuser_sim.nav2_probe_node import bounded
except ModuleNotFoundError as error:  # Windows intentionally has no ROS install.
    if error.name in {
        "action_msgs",
        "geometry_msgs",
        "lifecycle_msgs",
        "nav2_msgs",
        "rclpy",
    }:
        bounded = None
    else:
        raise


@unittest.skipIf(bounded is None, "ROS 2 Python packages are unavailable")
class TestNav2CommandBounds(unittest.TestCase):
    def test_preserves_in_envelope_values(self) -> None:
        assert bounded is not None
        self.assertEqual(bounded(0.08, 0.08), 0.08)
        self.assertEqual(bounded(-0.4, 0.4), -0.4)

    def test_clamps_out_of_envelope_values(self) -> None:
        assert bounded is not None
        self.assertEqual(bounded(0.5, 0.08), 0.08)
        self.assertEqual(bounded(-2.0, 0.4), -0.4)

    def test_nonfinite_values_stop(self) -> None:
        assert bounded is not None
        for value in (math.nan, math.inf, -math.inf):
            self.assertEqual(bounded(value, 0.08), 0.0)


if __name__ == "__main__":
    unittest.main()
