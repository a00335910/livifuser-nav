from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "ros2_ws"
    / "src"
    / "livifuser_goal_publisher"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_goal_publisher.goal_policy import (  # noqa: E402
    LockedRelativeWaypoint,
    RelativeGoalFields,
    RobotPose2D,
    relative_goal_from_world_waypoint,
    resolve_body_offset_waypoint,
    yaw_from_quaternion,
)


class GoalGeometryTests(unittest.TestCase):
    def assertGoalAlmostEqual(
        self, actual: RelativeGoalFields, expected: RelativeGoalFields
    ) -> None:
        self.assertAlmostEqual(actual.rho_m, expected.rho_m, places=7)
        self.assertAlmostEqual(actual.sin_alpha, expected.sin_alpha, places=7)
        self.assertAlmostEqual(actual.cos_alpha, expected.cos_alpha, places=7)

    def test_world_goal_ahead_and_left_use_robot_frame_signs(self) -> None:
        robot = RobotPose2D(1.0, 2.0, 0.0)
        ahead = relative_goal_from_world_waypoint(robot, target_x_m=4.0, target_y_m=2.0)
        left = relative_goal_from_world_waypoint(robot, target_x_m=1.0, target_y_m=4.0)
        self.assertGoalAlmostEqual(ahead, RelativeGoalFields(3.0, 0.0, 1.0))
        self.assertGoalAlmostEqual(left, RelativeGoalFields(2.0, 1.0, 0.0))

    def test_robot_yaw_rotates_world_delta_into_body_frame(self) -> None:
        robot = RobotPose2D(0.0, 0.0, math.pi / 2.0)
        goal = relative_goal_from_world_waypoint(robot, target_x_m=0.0, target_y_m=2.0)
        self.assertGoalAlmostEqual(goal, RelativeGoalFields(2.0, 0.0, 1.0))

    def test_goal_at_robot_has_defined_unit_direction(self) -> None:
        robot = RobotPose2D(2.0, -1.0, 1.2)
        goal = relative_goal_from_world_waypoint(robot, target_x_m=2.0, target_y_m=-1.0)
        self.assertEqual(goal, RelativeGoalFields(0.0, 0.0, 1.0))

    def test_body_offset_is_resolved_from_first_pose(self) -> None:
        start = RobotPose2D(10.0, 5.0, math.pi / 2.0)
        x_m, y_m = resolve_body_offset_waypoint(start, forward_m=3.0, left_m=1.0)
        self.assertAlmostEqual(x_m, 9.0)
        self.assertAlmostEqual(y_m, 8.0)

    def test_locked_waypoint_does_not_move_with_the_robot(self) -> None:
        waypoint = LockedRelativeWaypoint(forward_m=3.0, left_m=0.0)
        initial = waypoint.update(RobotPose2D(0.0, 0.0, 0.0))
        moved = waypoint.update(RobotPose2D(1.0, 0.0, 0.0))
        self.assertGoalAlmostEqual(initial, RelativeGoalFields(3.0, 0.0, 1.0))
        self.assertGoalAlmostEqual(moved, RelativeGoalFields(2.0, 0.0, 1.0))
        self.assertEqual(waypoint.target_xy_m, (3.0, 0.0))

    def test_quaternion_is_normalized_before_yaw_extraction(self) -> None:
        half = math.pi / 4.0
        yaw = yaw_from_quaternion(x=0.0, y=0.0, z=2.0 * math.sin(half), w=2.0 * math.cos(half))
        self.assertAlmostEqual(yaw, math.pi / 2.0)

    def test_invalid_geometry_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            RobotPose2D(math.nan, 0.0, 0.0)
        with self.assertRaises(ValueError):
            yaw_from_quaternion(x=0.0, y=0.0, z=0.0, w=0.0)
        with self.assertRaises(ValueError):
            resolve_body_offset_waypoint(
                RobotPose2D(0.0, 0.0, 0.0), forward_m=0.2, left_m=0.0
            )


if __name__ == "__main__":
    unittest.main()
