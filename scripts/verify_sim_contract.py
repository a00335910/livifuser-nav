#!/usr/bin/env python3
"""Verify the isolated Fortress world against the LiViFuser sensor contract."""

from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import TwistStamped
from livifuser_interfaces.msg import RelativeGoal
from livifuser_sim.world_layers import LAYER_COLLISION, load_world, point_clearance
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan

EXPECTED_K = [316.21156, 0.0, 223.13834, 0.0, 315.6497, 107.39364, 0.0, 0.0, 1.0]
EXPECTED_D = [0.012344, 0.038138, -0.016819, 0.004823, 0.0]
MAX_ODOM_WORLD_POSITION_ERROR_M = 0.20
MAX_ODOM_WORLD_YAW_ERROR_RAD = 0.20
MIN_CHANGED_MOVING_PAIR_FRACTION = 0.50


def stamp_seconds(message) -> float:
    return message.header.stamp.sec + message.header.stamp.nanosec / 1_000_000_000


def rate_hz(stamps: list[float]) -> float:
    intervals = [later - earlier for earlier, later in zip(stamps, stamps[1:], strict=False)]
    positive = [value for value in intervals if value > 0.0]
    return 1.0 / statistics.median(positive) if positive else 0.0


def span_seconds(stamps: list[float]) -> float:
    return max(stamps) - min(stamps) if len(stamps) >= 2 else 0.0


def odom_pose_xy_yaw(message: Odometry) -> tuple[float, float, float]:
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    yaw = math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
    )
    return float(position.x), float(position.y), yaw


def quaternion_rpy(message: Odometry) -> tuple[float, float, float]:
    orientation = message.pose.pose.orientation
    norm = math.sqrt(
        orientation.x**2
        + orientation.y**2
        + orientation.z**2
        + orientation.w**2
    )
    if norm <= 1e-9:
        return math.nan, math.nan, math.nan
    x = orientation.x / norm
    y = orientation.y / norm
    z = orientation.z / norm
    w = orientation.w / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def angle_error(first: float, second: float) -> float:
    return math.atan2(math.sin(first - second), math.cos(first - second))


class ContractSampler(Node):
    def __init__(self) -> None:
        super().__init__("livifuser_sim_contract_verifier")
        self.images: list[Image] = []
        self.camera_info: list[CameraInfo] = []
        self.scans: list[LaserScan] = []
        self.odom: list[Odometry] = []
        self.ground_truth: list[Odometry] = []
        self.goals: list[RelativeGoal] = []
        self.actions: list[TwistStamped] = []
        self.create_subscription(
            Image, "/camera/image_raw", self.images.append, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo,
            "/camera/camera_info",
            self.camera_info.append,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LaserScan, "/scan", self.scans.append, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, "/odom", self.odom.append, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry,
            "/livifuser/sim/ground_truth/odom",
            self.ground_truth.append,
            qos_profile_sensor_data,
        )
        self.create_subscription(RelativeGoal, "/livifuser/goal_relative", self.goals.append, 10)
        self.create_subscription(
            TwistStamped, "/livifuser/cmd_vel_stamped", self.actions.append, 10
        )


def close(actual: list[float], expected: list[float], tolerance: float = 1e-7) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)
        for a, b in zip(actual, expected, strict=True)
    )


def planar_displacement(messages: list[Odometry]) -> float:
    if len(messages) < 2:
        return 0.0
    first = messages[0].pose.pose.position
    last = messages[-1].pose.pose.position
    return math.hypot(last.x - first.x, last.y - first.y)


def scan_clearance(message: LaserScan | None, minimum: float, maximum: float) -> float | None:
    if message is None:
        return None
    values = []
    for index, distance in enumerate(message.ranges):
        angle = message.angle_min + index * message.angle_increment
        signed = math.atan2(math.sin(angle), math.cos(angle))
        if minimum <= signed <= maximum and math.isfinite(distance) and distance > 0.0:
            values.append(float(distance))
    return min(values) if values else None


def scan_statistics(message: LaserScan | None) -> dict[str, float | int] | None:
    if message is None:
        return None
    values = sorted(
        float(value)
        for value in message.ranges
        if math.isfinite(value) and value > 0.0
    )
    if not values:
        return {
            "finite_positive_count": 0,
            "zero_count": sum(value == 0.0 for value in message.ranges),
            "nan_count": sum(math.isnan(value) for value in message.ranges),
        }
    return {
        "finite_positive_count": len(values),
        "zero_count": sum(value == 0.0 for value in message.ranges),
        "nan_count": sum(math.isnan(value) for value in message.ranges),
        "at_minimum_count": sum(value <= message.range_min + 1e-6 for value in values),
        "minimum_m": values[0],
        "p10_m": values[int(0.10 * (len(values) - 1))],
        "median_m": statistics.median(values),
        "maximum_m": values[-1],
    }


def image_statistics(message: Image | None) -> dict[str, object] | None:
    """Summarize one active rgb8 frame without including row-padding bytes."""

    if message is None or message.encoding != "rgb8" or message.width <= 0 or message.height <= 0:
        return None
    row_bytes = int(message.width) * 3
    if message.step < row_bytes or len(message.data) < message.step * message.height:
        return None
    active = bytearray(row_bytes * int(message.height))
    for row in range(int(message.height)):
        source_start = row * int(message.step)
        target_start = row * row_bytes
        active[target_start : target_start + row_bytes] = message.data[
            source_start : source_start + row_bytes
        ]

    count = int(message.width) * int(message.height)
    channel_sums = [0.0, 0.0, 0.0]
    luminance_sum = 0.0
    luminance_square_sum = 0.0
    near_black = 0
    near_white = 0
    for offset in range(0, len(active), 3):
        red, green, blue = active[offset], active[offset + 1], active[offset + 2]
        channel_sums[0] += red
        channel_sums[1] += green
        channel_sums[2] += blue
        luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0
        luminance_sum += luminance
        luminance_square_sum += luminance * luminance
        near_black += luminance <= 0.02
        near_white += luminance >= 0.98
    luminance_mean = luminance_sum / count
    luminance_variance = max(
        0.0, luminance_square_sum / count - luminance_mean * luminance_mean
    )
    return {
        "active_rgb_sha256": hashlib.sha256(active).hexdigest().upper(),
        "channel_mean_rgb_normalized": [
            value / (count * 255.0) for value in channel_sums
        ],
        "luminance_mean_normalized": luminance_mean,
        "luminance_std_normalized": math.sqrt(luminance_variance),
        "near_black_fraction": near_black / count,
        "near_white_fraction": near_white / count,
    }


def active_rgb_bytes(message: Image) -> bytes | None:
    if message.encoding != "rgb8" or message.width <= 0 or message.height <= 0:
        return None
    row_bytes = int(message.width) * 3
    if message.step < row_bytes or len(message.data) < message.step * message.height:
        return None
    if message.step == row_bytes:
        return bytes(message.data[: row_bytes * int(message.height)])
    active = bytearray(row_bytes * int(message.height))
    for row in range(int(message.height)):
        source_start = row * int(message.step)
        target_start = row * row_bytes
        active[target_start : target_start + row_bytes] = message.data[
            source_start : source_start + row_bytes
        ]
    return bytes(active)


def temporal_rgb_statistics(
    images: list[Image], ground_truth: list[Odometry]
) -> dict[str, object] | None:
    if not images:
        return None
    image_records: list[tuple[float, str, bytes]] = []
    for message in images:
        active = active_rgb_bytes(message)
        if active is None:
            continue
        image_records.append(
            (stamp_seconds(message), hashlib.sha256(active).hexdigest().upper(), active)
        )
    if not image_records:
        return None

    counts = collections.Counter(record[1] for record in image_records)
    maximum_run_frames = 1
    maximum_run_sec = 0.0
    run_start = 0
    for index in range(1, len(image_records) + 1):
        if index < len(image_records) and image_records[index][1] == image_records[run_start][1]:
            continue
        maximum_run_frames = max(maximum_run_frames, index - run_start)
        maximum_run_sec = max(
            maximum_run_sec,
            image_records[index - 1][0] - image_records[run_start][0],
        )
        run_start = index

    gt_records = sorted(
        (stamp_seconds(message), odom_pose_xy_yaw(message)) for message in ground_truth
    )
    gt_stamps = [record[0] for record in gt_records]

    def nearest_pose(stamp: float) -> tuple[float, float, float] | None:
        if not gt_records:
            return None
        index = bisect.bisect_left(gt_stamps, stamp)
        candidates = []
        if index < len(gt_records):
            candidates.append(gt_records[index])
        if index > 0:
            candidates.append(gt_records[index - 1])
        _, pose = min(candidates, key=lambda record: abs(record[0] - stamp))
        return pose

    sampled = [image_records[0]]
    for record in image_records[1:]:
        if record[0] - sampled[-1][0] >= 0.45:
            sampled.append(record)
    if sampled[-1] != image_records[-1]:
        sampled.append(image_records[-1])

    maximum_identical_motion_run_sec = 0.0
    run_start = 0
    for index in range(1, len(image_records) + 1):
        if index < len(image_records) and image_records[index][1] == image_records[run_start][1]:
            continue
        earlier_pose = nearest_pose(image_records[run_start][0])
        later_pose = nearest_pose(image_records[index - 1][0])
        if earlier_pose is not None and later_pose is not None:
            translation = math.hypot(
                later_pose[0] - earlier_pose[0], later_pose[1] - earlier_pose[1]
            )
            rotation = abs(angle_error(later_pose[2], earlier_pose[2]))
            if translation >= 0.01 or rotation >= 0.03:
                maximum_identical_motion_run_sec = max(
                    maximum_identical_motion_run_sec,
                    image_records[index - 1][0] - image_records[run_start][0],
                )
        run_start = index

    moving_pair_differences: list[float] = []
    changed_moving_pairs = 0
    for earlier, later in zip(sampled, sampled[1:], strict=False):
        earlier_pose = nearest_pose(earlier[0])
        later_pose = nearest_pose(later[0])
        if earlier_pose is None or later_pose is None:
            continue
        translation = math.hypot(
            later_pose[0] - earlier_pose[0], later_pose[1] - earlier_pose[1]
        )
        rotation = abs(angle_error(later_pose[2], earlier_pose[2]))
        if translation < 0.01 and rotation < 0.03:
            continue
        difference = sum(
            abs(int(first) - int(second))
            for first, second in zip(earlier[2], later[2], strict=True)
        ) / (len(earlier[2]) * 255.0)
        moving_pair_differences.append(difference)
        if earlier[1] != later[1] and difference >= 0.001:
            changed_moving_pairs += 1

    modal_count = max(counts.values())
    return {
        "valid_rgb_frames": len(image_records),
        "unique_frame_hashes": len(counts),
        "modal_frame_fraction": modal_count / len(image_records),
        "maximum_identical_run_frames": maximum_run_frames,
        "maximum_identical_run_sec": maximum_run_sec,
        "maximum_identical_motion_run_sec": maximum_identical_motion_run_sec,
        "sample_interval_sec": 0.45,
        "moving_pair_count": len(moving_pair_differences),
        "changed_moving_pair_fraction": (
            changed_moving_pairs / len(moving_pair_differences)
            if moving_pair_differences
            else 0.0
        ),
        "motion_pair_mean_absolute_rgb_difference_median": (
            statistics.median(moving_pair_differences)
            if moving_pair_differences
            else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-sec", type=float, default=8.0)
    parser.add_argument("--expect-actions", action="store_true")
    parser.add_argument("--max-linear-mps", type=float, default=0.08)
    parser.add_argument("--max-angular-radps", type=float, default=0.40)
    parser.add_argument("--world-json", type=Path)
    parser.add_argument(
        "--until-goal",
        action="store_true",
        help="treat duration as a wall-time timeout and stop at the generated goal",
    )
    parser.add_argument("--output", type=Path)
    args, ros_args = parser.parse_known_args()
    if args.until_goal and args.world_json is None:
        parser.error("--until-goal requires --world-json")
    if args.max_linear_mps <= 0.0 or args.max_angular_radps <= 0.0:
        parser.error("action bounds must be positive")
    world = None if args.world_json is None else load_world(args.world_json)
    if world is not None and (world.start_pose_xy_yaw is None or world.goal_xy_m is None):
        parser.error("--world-json must declare a generated start pose and goal")
    rclpy.init(args=ros_args)
    node = ContractSampler()
    started_wall = time.monotonic()
    deadline = started_wall + args.duration_sec
    goal_reached = False
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if args.until_goal and node.ground_truth and world is not None:
            world_pose = odom_pose_xy_yaw(node.ground_truth[-1])
            goal_reached = (
                math.hypot(
                    world.goal_xy_m[0] - world_pose[0],
                    world.goal_xy_m[1] - world_pose[1],
                )
                <= 0.25
            )
            if goal_reached:
                # Capture several terminal zero labels rather than stopping on
                # the first pose inside tolerance.
                for _ in range(5):
                    rclpy.spin_once(node, timeout_sec=0.12)
                break
    wall_elapsed_sec = time.monotonic() - started_wall

    issues: list[str] = []
    image_stamps = [stamp_seconds(item) for item in node.images]
    scan_stamps = [stamp_seconds(item) for item in node.scans]
    odom_stamps = [stamp_seconds(item) for item in node.odom]
    ground_truth_stamps = [stamp_seconds(item) for item in node.ground_truth]
    goal_stamps = [stamp_seconds(item) for item in node.goals]
    action_stamps = [stamp_seconds(item) for item in node.actions]
    rates = {
        "image": rate_hz(image_stamps),
        "scan": rate_hz(scan_stamps),
        "odom": rate_hz(odom_stamps),
        "ground_truth": rate_hz(ground_truth_stamps),
        "goal": rate_hz(goal_stamps),
        "action": rate_hz(action_stamps),
    }
    if len(node.images) < 30 or not 25.0 <= rates["image"] <= 35.0:
        issues.append("image_rate_or_availability")
    if len(node.camera_info) < 30:
        issues.append("camera_info_rate_or_availability")
    if len(node.scans) < 10 or not 8.0 <= rates["scan"] <= 12.0:
        issues.append("scan_rate_or_availability")
    if len(node.odom) < 20 or not 16.0 <= rates["odom"] <= 24.0:
        issues.append("odom_rate_or_availability")
    if len(node.ground_truth) < 20 or rates["ground_truth"] < 10.0:
        issues.append("ground_truth_rate_or_availability")
    if len(node.goals) < 10 or not 8.0 <= rates["goal"] <= 12.0:
        issues.append("goal_rate_or_availability")
    if args.expect_actions and (
        len(node.actions) < 10 or not 8.0 <= rates["action"] <= 12.0
    ):
        issues.append("action_rate_or_availability")
    moving_actions = [
        item
        for item in node.actions
        if abs(item.twist.linear.x) > 1e-9 or abs(item.twist.angular.z) > 1e-9
    ]
    if args.expect_actions and not moving_actions:
        issues.append("action_nonzero_missing")
    if args.expect_actions and any(
        abs(item.twist.linear.x) > args.max_linear_mps + 1e-7
        or abs(item.twist.angular.z) > args.max_angular_radps + 1e-7
        for item in node.actions
    ):
        issues.append("action_bounds")
    odom_displacement_m = planar_displacement(node.odom)
    ground_truth_displacement_m = planar_displacement(node.ground_truth)
    if args.expect_actions and ground_truth_displacement_m < 0.05:
        issues.append("ground_truth_motion_missing")

    ground_truth_rpy = [quaternion_rpy(message) for message in node.ground_truth]
    maximum_abs_roll_rad = (
        max(abs(values[0]) for values in ground_truth_rpy) if ground_truth_rpy else None
    )
    maximum_abs_pitch_rad = (
        max(abs(values[1]) for values in ground_truth_rpy) if ground_truth_rpy else None
    )
    if (
        maximum_abs_roll_rad is not None
        and maximum_abs_pitch_rad is not None
        and (
            not math.isfinite(maximum_abs_roll_rad)
            or not math.isfinite(maximum_abs_pitch_rad)
            or maximum_abs_roll_rad > math.radians(5.0)
            or maximum_abs_pitch_rad > math.radians(5.0)
        )
    ):
        issues.append("ground_truth_tilt")

    odom_world_position_error_m = None
    odom_world_yaw_error_rad = None
    if len(node.odom) >= 2 and len(node.ground_truth) >= 2:
        odom_first = odom_pose_xy_yaw(node.odom[0])
        odom_last = odom_pose_xy_yaw(node.odom[-1])
        truth_first = odom_pose_xy_yaw(node.ground_truth[0])
        truth_last = odom_pose_xy_yaw(node.ground_truth[-1])
        odom_delta_x = odom_last[0] - odom_first[0]
        odom_delta_y = odom_last[1] - odom_first[1]
        frame_rotation = truth_first[2] - odom_first[2]
        expected_x = (
            truth_first[0]
            + math.cos(frame_rotation) * odom_delta_x
            - math.sin(frame_rotation) * odom_delta_y
        )
        expected_y = (
            truth_first[1]
            + math.sin(frame_rotation) * odom_delta_x
            + math.cos(frame_rotation) * odom_delta_y
        )
        expected_yaw = truth_first[2] + angle_error(odom_last[2], odom_first[2])
        odom_world_position_error_m = math.hypot(
            expected_x - truth_last[0], expected_y - truth_last[1]
        )
        odom_world_yaw_error_rad = abs(angle_error(expected_yaw, truth_last[2]))
        if (
            odom_world_position_error_m > MAX_ODOM_WORLD_POSITION_ERROR_M
            or odom_world_yaw_error_rad > MAX_ODOM_WORLD_YAW_ERROR_RAD
        ):
            issues.append("odom_world_pose_disagreement")

    collision_clearances: list[float] = []
    final_goal_distance_m = None
    if world is not None:
        collision_obstacles = world.layer(LAYER_COLLISION)
        world_poses = [odom_pose_xy_yaw(message) for message in node.ground_truth]
        collision_clearances = [
            point_clearance(collision_obstacles, x_m, y_m)
            for x_m, y_m, _ in world_poses
        ]
        if world_poses:
            final_x, final_y, _ = world_poses[-1]
            final_goal_distance_m = math.hypot(
                world.goal_xy_m[0] - final_x,
                world.goal_xy_m[1] - final_y,
            )
        if collision_clearances and min(collision_clearances) < 0.105:
            issues.append("ground_truth_collision")
    if args.until_goal and not goal_reached:
        issues.append("goal_not_reached")

    first_image = node.images[0] if node.images else None
    first_info = node.camera_info[0] if node.camera_info else None
    first_scan = node.scans[0] if node.scans else None
    first_odom = node.odom[0] if node.odom else None
    first_goal = node.goals[0] if node.goals else None
    if first_image and (
        first_image.width != 320
        or first_image.height != 240
        or first_image.encoding != "rgb8"
        or first_image.header.frame_id != "camera"
    ):
        issues.append("image_contract")
    if first_info and (
        first_info.width != 320
        or first_info.height != 240
        or first_info.header.frame_id != "camera"
        or first_info.distortion_model != "plumb_bob"
        or not close(list(first_info.k), EXPECTED_K)
        or not close(list(first_info.d), EXPECTED_D)
    ):
        issues.append("camera_info_contract")
    scan_beam_counts = [len(item.ranges) for item in node.scans]
    scan_contract_valid = all(
        item.header.frame_id == "base_scan"
        and 379 <= len(item.ranges) <= 407
        and math.isclose(item.angle_min, 0.0, abs_tol=1e-6)
        and math.isclose(item.angle_max, 2.0 * math.pi, abs_tol=1e-5)
        and math.isclose(
            item.angle_increment,
            2.0 * math.pi / (len(item.ranges) + 1),
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        and math.isclose(item.scan_time, 0.099677066, rel_tol=0.0, abs_tol=1e-6)
        and not any(math.isinf(value) for value in item.ranges)
        and all(
            math.isnan(value)
            or value == 0.0
            or (
                item.range_min <= value <= item.range_max
                and math.isclose(value / 0.001, round(value / 0.001), abs_tol=0.001)
            )
            for value in item.ranges
        )
        for item in node.scans
    )
    if first_scan and not scan_contract_valid:
        issues.append("scan_contract")
    if len(node.scans) >= 10 and len(set(scan_beam_counts)) < 2:
        issues.append("scan_beam_count_not_variable")
    first_scan_stats = scan_statistics(first_scan)
    first_image_stats = image_statistics(first_image)
    temporal_image_stats = temporal_rgb_statistics(node.images, node.ground_truth)
    if temporal_image_stats is not None:
        temporal_image_stats["gates"] = {
            "minimum_valid_rgb_frames": 60,
            "minimum_unique_frame_hashes": 20,
            "maximum_modal_frame_fraction": 0.50,
            "minimum_moving_pair_count": 10,
            "minimum_changed_moving_pair_fraction": (
                MIN_CHANGED_MOVING_PAIR_FRACTION
            ),
            "minimum_motion_pair_median_rgb_difference": 0.001,
            "maximum_identical_motion_run_sec": 1.0,
        }
    if (
        temporal_image_stats is None
        or temporal_image_stats["valid_rgb_frames"] < 60
        or temporal_image_stats["unique_frame_hashes"] < 20
        or temporal_image_stats["modal_frame_fraction"] > 0.50
    ):
        issues.append("image_temporal_degenerate")
    if (
        temporal_image_stats is None
        or temporal_image_stats["moving_pair_count"] < 10
        or temporal_image_stats["changed_moving_pair_fraction"]
        < MIN_CHANGED_MOVING_PAIR_FRACTION
        or temporal_image_stats["motion_pair_mean_absolute_rgb_difference_median"]
        < 0.001
        or temporal_image_stats["maximum_identical_motion_run_sec"] > 1.0
    ):
        issues.append("image_motion_change_missing")
    if first_scan_stats and (
        first_scan_stats.get("finite_positive_count", 0) == 0
        or first_scan_stats.get("at_minimum_count", 0)
        >= 0.90 * first_scan_stats.get("finite_positive_count", 0)
    ):
        issues.append("scan_payload_degenerate")
    if first_odom and (
        first_odom.header.frame_id != "odom" or first_odom.child_frame_id != "base_link"
    ):
        issues.append("odom_contract")
    if first_goal and first_goal.header.frame_id != "base_link":
        issues.append("goal_contract")

    result = {
        "valid": not issues,
        "issues": issues,
        "duration_sec": args.duration_sec,
        "wall_elapsed_sec": wall_elapsed_sec,
        "counts": {
            "image": len(node.images),
            "camera_info": len(node.camera_info),
            "scan": len(node.scans),
            "odom": len(node.odom),
            "ground_truth": len(node.ground_truth),
            "goal": len(node.goals),
            "action": len(node.actions),
        },
        "header_rates_hz": rates,
        "simulated_span_sec": {
            "image": span_seconds(image_stamps),
            "scan": span_seconds(scan_stamps),
            "odom": span_seconds(odom_stamps),
            "ground_truth": span_seconds(ground_truth_stamps),
            "goal": span_seconds(goal_stamps),
            "action": span_seconds(action_stamps),
        },
        "camera_realtime_factor": (
            span_seconds(image_stamps) / wall_elapsed_sec if wall_elapsed_sec else 0.0
        ),
        "closed_loop": {
            "until_goal": args.until_goal,
            "goal_reached": goal_reached,
            "final_goal_distance_m": final_goal_distance_m,
            "minimum_collision_clearance_m": (
                None if not collision_clearances else min(collision_clearances)
            ),
            "collision_radius_m": 0.105,
            "collision": bool(
                collision_clearances and min(collision_clearances) < 0.105
            ),
        },
        "simulated_motion": {
            "ground_truth_planar_displacement_m": ground_truth_displacement_m,
            "odom_planar_displacement_m": odom_displacement_m,
            "maximum_abs_roll_rad": maximum_abs_roll_rad,
            "maximum_abs_pitch_rad": maximum_abs_pitch_rad,
            "tilt_limit_rad": math.radians(5.0),
            "final_odom_world_position_error_m": odom_world_position_error_m,
            "final_odom_world_yaw_error_rad": odom_world_yaw_error_rad,
            "odom_world_position_error_limit_m": MAX_ODOM_WORLD_POSITION_ERROR_M,
            "odom_world_yaw_error_limit_rad": MAX_ODOM_WORLD_YAW_ERROR_RAD,
            "moving_action_count": len(moving_actions),
            "configured_action_bounds": {
                "linear_mps": args.max_linear_mps,
                "angular_radps": args.max_angular_radps,
            },
            "linear_x_range_mps": None
            if not node.actions
            else [
                min(item.twist.linear.x for item in node.actions),
                max(item.twist.linear.x for item in node.actions),
            ],
            "angular_z_range_radps": None
            if not node.actions
            else [
                min(item.twist.angular.z for item in node.actions),
                max(item.twist.angular.z for item in node.actions),
            ],
        },
        "first_scan_clearance_m": {
            "front": scan_clearance(first_scan, -0.30, 0.30),
            "left": scan_clearance(first_scan, 0.25, 1.20),
            "right": scan_clearance(first_scan, -1.20, -0.25),
        },
        "first_scan_statistics": first_scan_stats,
        "first_image_statistics": first_image_stats,
        "temporal_image_statistics": temporal_image_stats,
        "scan_observation_model": {
            "beam_counts_observed": sorted(set(scan_beam_counts)),
            "beam_count_range_observed": None
            if not scan_beam_counts
            else [min(scan_beam_counts), max(scan_beam_counts)],
            "variable_beam_count": len(set(scan_beam_counts)) >= 2,
            "expected_increment_rule": "2*pi/(beam_count+1)",
            "expected_quantization_m": 0.001,
            "expected_scan_interval_sec": 0.099677066,
        },
        "first_odom_pose": None
        if first_odom is None
        else {
            "position": [
                first_odom.pose.pose.position.x,
                first_odom.pose.pose.position.y,
                first_odom.pose.pose.position.z,
            ],
            "quaternion_xyzw": [
                first_odom.pose.pose.orientation.x,
                first_odom.pose.pose.orientation.y,
                first_odom.pose.pose.orientation.z,
                first_odom.pose.pose.orientation.w,
            ],
        },
        "contracts": {
            "image": None
            if first_image is None
            else {
                "width": first_image.width,
                "height": first_image.height,
                "encoding": first_image.encoding,
                "frame_id": first_image.header.frame_id,
            },
            "scan": None
            if first_scan is None
            else {
                "beam_count": len(first_scan.ranges),
                "angle_min": first_scan.angle_min,
                "angle_max": first_scan.angle_max,
                "frame_id": first_scan.header.frame_id,
            },
        },
    }
    node.destroy_node()
    rclpy.shutdown()
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
