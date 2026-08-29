import math
import unittest
from types import SimpleNamespace

from livifuser_nav.decode import (
    goal_payload,
    image_payload,
    odometry_payload,
    scan_payload,
    twist_payload,
    yaw_from_quaternion,
)
from livifuser_nav.export_schema import RejectionCode

TWO_PI = 2.0 * math.pi
CAPTURE = (320, 240, "bgra8")


def header(frame_id: str = "camera") -> SimpleNamespace:
    return SimpleNamespace(frame_id=frame_id, stamp=SimpleNamespace(sec=0, nanosec=0))


def image(
    width: int = 320,
    height: int = 240,
    encoding: str = "bgra8",
    step: int | None = None,
    data_len: int | None = None,
) -> SimpleNamespace:
    resolved_step = width * 4 if step is None else step
    resolved_len = height * resolved_step if data_len is None else data_len
    return SimpleNamespace(
        width=width,
        height=height,
        encoding=encoding,
        step=resolved_step,
        data=bytes(resolved_len),
        header=header(),
    )


def scan(
    beams: int = 399,
    angle_min: float = 0.0,
    angle_max: float = TWO_PI,
    increment: float | None = None,
    fill: float = 1.25,
) -> SimpleNamespace:
    return SimpleNamespace(
        ranges=[fill] * beams,
        angle_min=angle_min,
        angle_max=angle_max,
        angle_increment=TWO_PI / 400.0 if increment is None else increment,
        range_min=0.1,
        range_max=100.0,
        scan_time=0.0996,
        header=header("base_scan"),
    )


def odom(linear: float = 0.05, angular: float = 0.0, x: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=linear), angular=SimpleNamespace(z=angular)
            )
        ),
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        ),
        header=header("odom"),
    )


def goal(rho: float = 1.0, sin_alpha: float = 0.0, cos_alpha: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        rho_m=rho, sin_alpha=sin_alpha, cos_alpha=cos_alpha, header=header("base_link")
    )


class ImageTests(unittest.TestCase):
    def test_accepts_the_locked_capture_contract(self) -> None:
        payload = image_payload(image(), CAPTURE)
        self.assertTrue(payload.valid)
        self.assertEqual(payload.data["data_bytes"], 307200)
        self.assertEqual(payload.data["channels"], 4)

    def test_rejects_truncated_payload(self) -> None:
        payload = image_payload(image(data_len=1000), CAPTURE)
        self.assertFalse(payload.valid)
        self.assertEqual(payload.invalid_code, RejectionCode.CAMERA_PAYLOAD_INVALID)
        self.assertIn("payload_length", payload.data["problems"])

    def test_rejects_empty_payload(self) -> None:
        payload = image_payload(image(data_len=0), CAPTURE)
        self.assertIn("empty_payload", payload.data["problems"])

    def test_rejects_wrong_resolution(self) -> None:
        payload = image_payload(image(width=640, height=480), CAPTURE)
        self.assertIn("geometry_or_encoding", payload.data["problems"])

    def test_rejects_inconsistent_step(self) -> None:
        payload = image_payload(image(step=960), CAPTURE)
        self.assertIn("step", payload.data["problems"])

    def test_rejects_unknown_encoding(self) -> None:
        payload = image_payload(image(encoding="yuv422"), CAPTURE)
        self.assertIn("unknown_encoding", payload.data["problems"])

    def test_three_channel_encoding_uses_three_bytes(self) -> None:
        payload = image_payload(
            image(encoding="rgb8", step=960), (320, 240, "rgb8")
        )
        self.assertTrue(payload.valid)
        self.assertEqual(payload.data["channels"], 3)


class ScanTests(unittest.TestCase):
    def test_accepts_the_measured_lds03_geometry(self) -> None:
        # The real driver reports angle_max = 2*pi with a 2*pi/400 increment while
        # publishing 399 beams, omitting the duplicate final beam.
        payload = scan_payload(scan())
        self.assertTrue(payload.valid, payload.data["problems"])
        self.assertEqual(payload.data["beam_count"], 399)
        self.assertEqual(payload.data["valid_fraction"], 1.0)

    def test_accepts_a_full_400_beam_scan(self) -> None:
        self.assertTrue(scan_payload(scan(beams=400)).valid)

    def test_rejects_beam_count_inconsistent_with_the_span(self) -> None:
        payload = scan_payload(scan(beams=399, angle_max=math.pi / 2.0))
        self.assertFalse(payload.valid)
        self.assertIn("beam_count_mismatch", payload.data["problems"])

    def test_rejects_empty_scan(self) -> None:
        payload = scan_payload(scan(beams=0))
        self.assertIn("empty", payload.data["problems"])

    def test_rejects_zero_angle_increment(self) -> None:
        payload = scan_payload(scan(increment=0.0))
        self.assertIn("zero_angle_increment", payload.data["problems"])

    def test_rejects_scan_with_no_usable_returns(self) -> None:
        payload = scan_payload(scan(fill=float("inf")))
        self.assertFalse(payload.valid)
        self.assertIn("no_valid_returns", payload.data["problems"])
        self.assertEqual(payload.data["valid_return_count"], 0)

    def test_counts_out_of_range_returns_as_invalid(self) -> None:
        payload = scan_payload(scan(fill=0.05))  # below range_min
        self.assertEqual(payload.data["valid_return_count"], 0)

    def test_partial_dropout_is_still_a_usable_scan(self) -> None:
        message = scan()
        message.ranges = [1.25] * 389 + [float("inf")] * 10
        payload = scan_payload(message)
        self.assertTrue(payload.valid)
        self.assertEqual(payload.data["valid_return_count"], 389)


class OdometryTests(unittest.TestCase):
    def test_extracts_the_locked_robot_state(self) -> None:
        payload = odometry_payload(odom(linear=0.05, angular=-0.1))
        self.assertTrue(payload.valid)
        self.assertAlmostEqual(payload.data["linear_velocity_mps"], 0.05)
        self.assertAlmostEqual(payload.data["angular_velocity_radps"], -0.1)

    def test_rejects_non_finite_velocity(self) -> None:
        payload = odometry_payload(odom(linear=float("nan")))
        self.assertFalse(payload.valid)
        self.assertEqual(payload.invalid_code, RejectionCode.ODOM_INVALID)

    def test_identity_orientation_is_zero_yaw(self) -> None:
        self.assertAlmostEqual(
            yaw_from_quaternion(SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)), 0.0
        )

    def test_quarter_turn_yaw(self) -> None:
        half = math.sqrt(0.5)
        self.assertAlmostEqual(
            yaw_from_quaternion(SimpleNamespace(x=0.0, y=0.0, z=half, w=half)),
            math.pi / 2.0,
            places=6,
        )


class GoalTests(unittest.TestCase):
    def test_accepts_a_valid_relative_goal(self) -> None:
        payload = goal_payload(goal())
        self.assertTrue(payload.valid)
        self.assertEqual(payload.data["rho_m"], 1.0)

    def test_rejects_non_unit_direction(self) -> None:
        payload = goal_payload(goal(sin_alpha=0.5, cos_alpha=0.5))
        self.assertFalse(payload.valid)
        self.assertEqual(payload.invalid_code, RejectionCode.GOAL_INVALID)

    def test_rejects_negative_range(self) -> None:
        self.assertFalse(goal_payload(goal(rho=-1.0)).valid)

    def test_rejects_non_finite_goal(self) -> None:
        self.assertFalse(goal_payload(goal(rho=float("inf"))).valid)


class TwistTests(unittest.TestCase):
    def test_accepts_a_finite_command(self) -> None:
        payload = twist_payload(0.05, 0.0)
        self.assertTrue(payload.valid)
        self.assertEqual(payload.data["linear_velocity_mps"], 0.05)

    def test_rejects_non_finite_command(self) -> None:
        payload = twist_payload(float("nan"), 0.0)
        self.assertFalse(payload.valid)
        self.assertEqual(payload.invalid_code, RejectionCode.ACTION_INVALID)

    def test_zero_command_is_valid_and_not_treated_as_absent(self) -> None:
        self.assertTrue(twist_payload(0.0, 0.0).valid)


if __name__ == "__main__":
    unittest.main()
