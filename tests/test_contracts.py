import unittest

from livifuser_nav.contracts import RelativeGoal, RobotState


class ContractTests(unittest.TestCase):
    def test_accepts_valid_relative_goal(self) -> None:
        goal = RelativeGoal(rho_m=1.5, sin_alpha=0.0, cos_alpha=1.0)
        self.assertEqual(goal.rho_m, 1.5)

    def test_rejects_non_unit_goal_direction(self) -> None:
        with self.assertRaises(ValueError):
            RelativeGoal(rho_m=1.0, sin_alpha=0.5, cos_alpha=0.5)

    def test_rejects_non_finite_robot_state(self) -> None:
        with self.assertRaises(ValueError):
            RobotState(linear_velocity_mps=float("nan"), angular_velocity_radps=0.0)


if __name__ == "__main__":
    unittest.main()
