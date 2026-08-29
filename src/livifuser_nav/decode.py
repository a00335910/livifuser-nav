"""Structural validation of recorded messages, independent of ROS.

Each function takes a duck-typed message (anything with the right attributes) and
returns a :class:`~livifuser_nav.sampling.Payload` carrying the extracted scalars
plus a validity verdict. Keeping these here rather than in the exporter script
means the rules that decide whether a frame is corrupt are unit tested.

Nothing here normalizes or resizes. These are corruption checks, not
preprocessing.
"""

from __future__ import annotations

import math
from typing import Any

from .contracts import RelativeGoal
from .export_schema import RejectionCode
from .sampling import Payload

#: Bytes per pixel for the encodings this pilot can produce.
ENCODING_CHANNELS: dict[str, int] = {
    "bgra8": 4,
    "rgba8": 4,
    "bgr8": 3,
    "rgb8": 3,
    "mono8": 1,
    "mono16": 2,
}

#: Slack in beams when checking a scan's count against its angular span.
#: A full-surround scan may omit the duplicate final beam, and drivers round the
#: reported span inconsistently, so a couple of beams of tolerance is normal.
BEAM_COUNT_TOLERANCE = 2.0


def image_payload(message: Any, expected: tuple[int, int, str]) -> Payload:
    """Validate an Image against the locked capture contract."""

    width, height, encoding = expected
    data_len = len(message.data)
    channels = ENCODING_CHANNELS.get(message.encoding)
    problems: list[str] = []

    if (message.width, message.height, message.encoding) != (width, height, encoding):
        problems.append("geometry_or_encoding")
    if channels is None:
        problems.append("unknown_encoding")
    elif message.step != message.width * channels:
        problems.append("step")
    if data_len == 0:
        problems.append("empty_payload")
    elif data_len != message.height * message.step:
        problems.append("payload_length")

    return Payload(
        data={
            "width": message.width,
            "height": message.height,
            "encoding": message.encoding,
            "step": message.step,
            "channels": channels,
            "data_bytes": data_len,
            "frame_id": message.header.frame_id,
            "problems": problems,
        },
        valid=not problems,
        invalid_code=RejectionCode.CAMERA_PAYLOAD_INVALID if problems else None,
    )


def scan_payload(message: Any) -> Payload:
    """Validate a LaserScan's geometry and confirm it carries real returns."""

    ranges = list(message.ranges)
    problems: list[str] = []

    if not ranges:
        problems.append("empty")
    if message.angle_increment == 0.0:
        problems.append("zero_angle_increment")
    elif ranges:
        implied = abs(
            (message.angle_max - message.angle_min) / message.angle_increment
        )
        if not implied - BEAM_COUNT_TOLERANCE <= len(ranges) <= implied + BEAM_COUNT_TOLERANCE:
            problems.append("beam_count_mismatch")

    valid_returns = [
        value
        for value in ranges
        if math.isfinite(value) and message.range_min <= value <= message.range_max
    ]
    if ranges and not valid_returns:
        problems.append("no_valid_returns")

    return Payload(
        data={
            "beam_count": len(ranges),
            "valid_return_count": len(valid_returns),
            "valid_fraction": (len(valid_returns) / len(ranges)) if ranges else 0.0,
            "angle_min": message.angle_min,
            "angle_max": message.angle_max,
            "angle_increment": message.angle_increment,
            "range_min": message.range_min,
            "range_max": message.range_max,
            "scan_time": message.scan_time,
            "frame_id": message.header.frame_id,
            "problems": problems,
        },
        valid=not problems,
        invalid_code=RejectionCode.LIDAR_PAYLOAD_INVALID if problems else None,
    )


def yaw_from_quaternion(orientation: Any) -> float:
    """Yaw about z from a ROS quaternion."""

    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
    )


def odometry_payload(message: Any) -> Payload:
    """Extract the locked robot state, plus pose for later evaluation use."""

    linear = message.twist.twist.linear.x
    angular = message.twist.twist.angular.z
    position = message.pose.pose.position
    values = (linear, angular, position.x, position.y)
    valid = all(math.isfinite(value) for value in values)
    return Payload(
        data={
            "linear_velocity_mps": linear,
            "angular_velocity_radps": angular,
            "pose_x_m": position.x,
            "pose_y_m": position.y,
            "pose_yaw_rad": yaw_from_quaternion(message.pose.pose.orientation),
        },
        valid=valid,
        invalid_code=None if valid else RejectionCode.ODOM_INVALID,
    )


def goal_payload(message: Any) -> Payload:
    """Validate a RelativeGoal through the locked contract itself."""

    data: dict[str, object] = {
        "rho_m": message.rho_m,
        "sin_alpha": message.sin_alpha,
        "cos_alpha": message.cos_alpha,
        "frame_id": message.header.frame_id,
    }
    try:
        # Reuse the runtime contract so the export cannot admit a goal the
        # publisher would itself have rejected.
        RelativeGoal(
            rho_m=message.rho_m,
            sin_alpha=message.sin_alpha,
            cos_alpha=message.cos_alpha,
        )
    except ValueError as error:
        return Payload(
            data={**data, "problem": str(error)},
            valid=False,
            invalid_code=RejectionCode.GOAL_INVALID,
        )
    return Payload(data=data)


def twist_payload(linear: float, angular: float) -> Payload:
    """Validate a velocity command. Never substitutes a value for a bad one."""

    valid = math.isfinite(linear) and math.isfinite(angular)
    return Payload(
        data={"linear_velocity_mps": linear, "angular_velocity_radps": angular},
        valid=valid,
        invalid_code=None if valid else RejectionCode.ACTION_INVALID,
    )
