"""Validate a robot-local LiViFuser Stage 1 pilot MCAP bag."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter
from math import atan2, hypot
from pathlib import Path
from typing import Any

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

STATIONARY_REQUIRED_TOPICS = {
    "/camera/image_raw",
    "/camera/camera_info",
    "/scan",
    "/odom",
    "/livifuser/goal_relative",
    "/tf",
    "/tf_static",
}


def stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return math.nan
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def statistics_ms(values_ns: list[int]) -> dict[str, float | int]:
    values_ms = sorted(value / 1_000_000 for value in values_ns)
    if not values_ms:
        return {"count": 0}
    return {
        "count": len(values_ms),
        "min_ms": values_ms[0],
        "median_ms": percentile(values_ms, 0.5),
        "mean_ms": sum(values_ms) / len(values_ms),
        "p95_ms": percentile(values_ms, 0.95),
        "max_ms": values_ms[-1],
    }


def interval_statistics(stamps: list[int]) -> dict[str, float | int]:
    intervals = [later - earlier for earlier, later in zip(stamps, stamps[1:], strict=False)]
    result = statistics_ms(intervals)
    positive = sorted(interval for interval in intervals if interval > 0)
    if positive:
        result["median_hz"] = 1_000_000_000 / percentile(positive, 0.5)
    result["nonpositive_intervals"] = sum(interval <= 0 for interval in intervals)
    return result


def nearest_offsets(source_stamps: list[int], target_stamps: list[int]) -> list[int]:
    targets = sorted(target_stamps)
    offsets: list[int] = []
    for source in source_stamps:
        index = bisect.bisect_left(targets, source)
        candidates: list[int] = []
        if index < len(targets):
            candidates.append(abs(targets[index] - source))
        if index > 0:
            candidates.append(abs(targets[index - 1] - source))
        if candidates:
            offsets.append(min(candidates))
    return offsets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-lidar-delta-ms", type=float, default=75.0)
    parser.add_argument("--require-action", action="store_true")
    parser.add_argument("--max-linear-mps", type=float, default=0.1)
    parser.add_argument("--max-angular-rps", type=float, default=0.5)
    args = parser.parse_args()

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    topic_types = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }
    message_types = {name: get_message(type_name) for name, type_name in topic_types.items()}

    counts: Counter[str] = Counter()
    header_stamps: dict[str, list[int]] = {
        "/camera/image_raw": [],
        "/scan": [],
        "/odom": [],
        "/livifuser/goal_relative": [],
    }
    record_ages: dict[str, list[int]] = {
        "/camera/image_raw": [],
        "/scan": [],
    }
    first_image: dict[str, Any] | None = None
    first_camera_info: dict[str, Any] | None = None
    first_scan: dict[str, Any] | None = None
    first_goal: dict[str, Any] | None = None
    static_tf_pairs: set[tuple[str, str]] = set()
    record_timestamps: list[int] = []
    command_samples: list[tuple[int, float, float]] = []
    odom_samples: list[tuple[int, float, float, float]] = []

    while reader.has_next():
        topic, serialized, record_timestamp = reader.read_next()
        counts[topic] += 1
        record_timestamps.append(record_timestamp)
        message = deserialize_message(serialized, message_types[topic])

        if topic in header_stamps:
            header_timestamp = stamp_ns(message)
            header_stamps[topic].append(header_timestamp)
            if topic in record_ages:
                record_ages[topic].append(record_timestamp - header_timestamp)

        if topic == "/cmd_vel":
            command_samples.append(
                (record_timestamp, message.linear.x, message.angular.z)
            )
        elif topic == "/odom":
            orientation = message.pose.pose.orientation
            yaw = atan2(
                2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
                1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
            )
            position = message.pose.pose.position
            odom_samples.append((stamp_ns(message), position.x, position.y, yaw))

        if topic == "/camera/image_raw" and first_image is None:
            first_image = {
                "width": message.width,
                "height": message.height,
                "encoding": message.encoding,
                "step": message.step,
                "data_bytes": len(message.data),
                "frame_id": message.header.frame_id,
            }
        elif topic == "/camera/camera_info" and first_camera_info is None:
            first_camera_info = {
                "width": message.width,
                "height": message.height,
                "distortion_model": message.distortion_model,
                "k": list(message.k),
                "d": list(message.d),
                "frame_id": message.header.frame_id,
            }
        elif topic == "/scan" and first_scan is None:
            first_scan = {
                "frame_id": message.header.frame_id,
                "beam_count": len(message.ranges),
                "scan_time_sec": message.scan_time,
                "time_increment_sec": message.time_increment,
            }
        elif topic == "/livifuser/goal_relative" and first_goal is None:
            first_goal = {
                "frame_id": message.header.frame_id,
                "rho_m": message.rho_m,
                "sin_alpha": message.sin_alpha,
                "cos_alpha": message.cos_alpha,
            }
        elif topic == "/tf_static":
            static_tf_pairs.update(
                (transform.header.frame_id, transform.child_frame_id)
                for transform in message.transforms
            )

    image_stamps = header_stamps["/camera/image_raw"]
    scan_stamps = header_stamps["/scan"]
    overlap_start = max(min(image_stamps), min(scan_stamps))
    overlap_end = min(max(image_stamps), max(scan_stamps))
    overlap_images = [stamp for stamp in image_stamps if overlap_start <= stamp <= overlap_end]
    # Retain all scans so a scan just outside the camera recording boundary can
    # still bracket the first or last overlapping image correctly.
    nearest = nearest_offsets(overlap_images, scan_stamps)
    max_delta_ns = int(args.max_lidar_delta_ms * 1_000_000)

    issues: list[str] = []
    missing = sorted(STATIONARY_REQUIRED_TOPICS - counts.keys())
    if missing:
        issues.append(f"missing stationary topics: {missing}")
    if first_image is None or (
        first_image["width"], first_image["height"], first_image["encoding"]
    ) != (320, 240, "bgra8"):
        issues.append(f"unexpected image contract: {first_image}")
    if first_camera_info is None or not any(first_camera_info["k"]):
        issues.append("camera calibration matrix is absent or zero")
    if any(offset > max_delta_ns for offset in nearest):
        issues.append("one or more camera frames exceed the LiDAR association limit")
    required_static_pairs = {
        ("base_scan", "camera"),
        ("camera", "camera_optical_frame"),
    }
    if not required_static_pairs.issubset(static_tf_pairs):
        issues.append("accepted camera-LiDAR static TF chain is missing")

    moving_commands = [
        sample
        for sample in command_samples
        if abs(sample[1]) > 1e-6 or abs(sample[2]) > 1e-6
    ]
    if args.require_action:
        if not command_samples:
            issues.append("moving pilot has no /cmd_vel messages")
        elif not moving_commands:
            issues.append("moving pilot has no nonzero /cmd_vel messages")
        else:
            if max(abs(sample[1]) for sample in command_samples) > args.max_linear_mps:
                issues.append("recorded linear command exceeded the authorized limit")
            if max(abs(sample[2]) for sample in command_samples) > args.max_angular_rps:
                issues.append("recorded angular command exceeded the authorized limit")
            if abs(command_samples[-1][1]) > 1e-6 or abs(command_samples[-1][2]) > 1e-6:
                issues.append("final recorded velocity command is not zero")

    action_summary: dict[str, Any] = {"count": len(command_samples)}
    if command_samples:
        action_summary.update(
            {
                "zero_count": len(command_samples) - len(moving_commands),
                "moving_count": len(moving_commands),
                "linear_x_min_mps": min(sample[1] for sample in command_samples),
                "linear_x_max_mps": max(sample[1] for sample in command_samples),
                "angular_z_min_rps": min(sample[2] for sample in command_samples),
                "angular_z_max_rps": max(sample[2] for sample in command_samples),
                "record_intervals": interval_statistics(
                    [sample[0] for sample in command_samples]
                ),
                "final_is_zero": (
                    abs(command_samples[-1][1]) <= 1e-6
                    and abs(command_samples[-1][2]) <= 1e-6
                ),
            }
        )
    if moving_commands:
        action_summary["moving_command_span_sec"] = (
            moving_commands[-1][0] - moving_commands[0][0]
        ) / 1_000_000_000

    odometry_summary: dict[str, Any] = {"count": len(odom_samples)}
    if odom_samples:
        start = odom_samples[0]
        end = odom_samples[-1]
        path_length = sum(
            hypot(later[1] - earlier[1], later[2] - earlier[2])
            for earlier, later in zip(odom_samples, odom_samples[1:], strict=False)
        )
        odometry_summary.update(
            {
                "start_xy_m": [start[1], start[2]],
                "end_xy_m": [end[1], end[2]],
                "net_displacement_m": hypot(end[1] - start[1], end[2] - start[2]),
                "integrated_path_length_m": path_length,
                "yaw_change_rad": end[3] - start[3],
            }
        )

    duration_ns = max(record_timestamps) - min(record_timestamps)
    record_start_ns = min(record_timestamps)
    scan_gap_events = [
        {
            "start_sec": (earlier - record_start_ns) / 1_000_000_000,
            "end_sec": (later - record_start_ns) / 1_000_000_000,
            "duration_ms": (later - earlier) / 1_000_000,
        }
        for earlier, later in zip(scan_stamps, scan_stamps[1:], strict=False)
        if later - earlier > 150_000_000
    ]
    association_failures = [
        {
            "image_sec": (stamp - record_start_ns) / 1_000_000_000,
            "nearest_scan_delta_ms": offset / 1_000_000,
        }
        for stamp, offset in zip(overlap_images, nearest, strict=True)
        if offset > max_delta_ns
    ]
    if moving_commands:
        action_summary["moving_window_sec"] = [
            (moving_commands[0][0] - record_start_ns) / 1_000_000_000,
            (moving_commands[-1][0] - record_start_ns) / 1_000_000_000,
        ]
    record = {
        "valid_pilot": not issues,
        "pilot_mode": "moving" if args.require_action else "stationary",
        "issues": issues,
        "bag": str(args.bag),
        "duration_sec": duration_ns / 1_000_000_000,
        "message_counts": dict(sorted(counts.items())),
        "image": first_image,
        "camera_info": first_camera_info,
        "scan": first_scan,
        "goal": first_goal,
        "action": action_summary,
        "odometry": odometry_summary,
        "static_tf_pairs": sorted([list(pair) for pair in static_tf_pairs]),
        "header_intervals": {
            name: interval_statistics(stamps) for name, stamps in header_stamps.items()
        },
        "record_arrival_minus_header": {
            name: statistics_ms(ages) for name, ages in record_ages.items()
        },
        "nearest_scan_for_each_image": {
            **statistics_ms(nearest),
            "limit_ms": args.max_lidar_delta_ms,
            "within_limit_percent": 100.0
            * sum(offset <= max_delta_ns for offset in nearest)
            / len(nearest),
        },
        "scan_gap_events_over_150_ms": scan_gap_events,
        "association_failures": association_failures,
        "action_note": (
            "Moving command validation enabled."
            if args.require_action
            else "The stationary safety pilot intentionally has no /cmd_vel messages."
        ),
    }
    rendered = json.dumps(record, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
