#!/usr/bin/env python3
"""Analyze a recorded watchdog zero gate without replaying any topic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "livifuser_command_watchdog"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_command_watchdog.policy import (  # noqa: E402, I001
    stale_detection_bound_ms,
)


SCHEMA_VERSION = "1.0.0"
TOPICS = (
    "/cmd_vel",
    "/livifuser/cmd_vel_stamped",
    "/livifuser/command_watchdog_status",
    "/livifuser/command_watchdog_timing",
    "/livifuser/teleop_intent_stamped",
    "/odom",
)


def stamp_ns(message: Any) -> int:
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


def twist_values(twist: Any) -> tuple[float, ...]:
    return (
        twist.linear.x,
        twist.linear.y,
        twist.linear.z,
        twist.angular.x,
        twist.angular.y,
        twist.angular.z,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gaps_ms(times_ns: list[int]) -> list[float]:
    return [
        (later - earlier) / 1_000_000.0
        for earlier, later in zip(times_ns, times_ns[1:], strict=False)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intent-timeout-ms", type=float, default=250.0)
    parser.add_argument("--max-command-gap-ms", type=float, default=150.0)
    parser.add_argument("--max-publish-duration-ms", type=float, default=25.0)
    parser.add_argument("--max-displacement-m", type=float, default=0.005)
    parser.add_argument("--max-linear-speed-mps", type=float, default=0.01)
    args = parser.parse_args()

    metadata_path = args.bag / "metadata.yaml"
    mcap_paths = sorted(args.bag.glob("*.mcap"))
    if len(mcap_paths) != 1 or not metadata_path.is_file():
        raise SystemExit("expected exactly one MCAP plus metadata.yaml")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    missing = set(TOPICS) - topic_types.keys()
    if missing:
        raise SystemExit(f"bag is missing required topics: {sorted(missing)}")

    samples: dict[str, list[tuple[int, Any]]] = {topic: [] for topic in TOPICS}
    while reader.has_next():
        topic, data, record_ns = reader.read_next()
        if topic in samples:
            samples[topic].append(
                (record_ns, deserialize_message(data, get_message(topic_types[topic])))
            )

    raw = samples["/cmd_vel"]
    stamped = samples["/livifuser/cmd_vel_stamped"]
    statuses = samples["/livifuser/command_watchdog_status"]
    timings = samples["/livifuser/command_watchdog_timing"]
    intents = samples["/livifuser/teleop_intent_stamped"]
    odom = samples["/odom"]

    raw_max_abs = max(abs(v) for _, msg in raw for v in twist_values(msg))
    stamped_max_abs = max(abs(v) for _, msg in stamped for v in twist_values(msg.twist))
    intent_max_abs = max(abs(v) for _, msg in intents for v in twist_values(msg.twist))
    status_output_max_abs = max(
        abs(v) for _, msg in statuses for v in (msg.output_linear_mps, msg.output_angular_radps)
    )

    raw_record_gaps = gaps_ms([record_ns for record_ns, _ in raw])
    usable_ticks = [msg.tick_interval_s for _, msg in timings if msg.tick_interval_s >= 0.0]
    usable_publishes = [
        msg.command_publish_duration_s
        for _, msg in timings
        if msg.command_publish_duration_s >= 0.0
    ]
    shutdown_indices = [
        index
        for index, (_, msg) in enumerate(timings)
        if msg.tick_interval_s < 0.0
        and msg.command_publish_duration_s < 0.0
        and msg.graph_cache_age_s < 0.0
    ]
    final_indices = list(range(len(timings) - 5, len(timings)))
    final_five_are_sentinels = shutdown_indices[-5:] == final_indices
    final_stamps = {
        "stamped_command": [stamp_ns(msg) for _, msg in stamped[-5:]],
        "status": [stamp_ns(msg) for _, msg in statuses[-5:]],
        "timing": [stamp_ns(msg) for _, msg in timings[-5:]],
    }
    final_stamps_aligned = (
        len(stamped) >= 5
        and len(statuses) >= 5
        and len(timings) >= 5
        and final_stamps["stamped_command"] == final_stamps["status"] == final_stamps["timing"]
    )
    final_raw_zero = len(raw) >= 5 and all(
        value == 0.0 for _, msg in raw[-5:] for value in twist_values(msg)
    )
    final_stamped_zero = len(stamped) >= 5 and all(
        value == 0.0 for _, msg in stamped[-5:] for value in twist_values(msg.twist)
    )

    reason_counts = Counter(msg.reason for _, msg in statuses)
    last_intent_stamp_ns = max((stamp_ns(msg) for _, msg in intents), default=0)
    stale_status_stamps = [
        stamp_ns(msg)
        for _, msg in statuses
        if msg.reason == "intent_stale" and stamp_ns(msg) >= last_intent_stamp_ns
    ]
    stale_transition_ms = (
        (min(stale_status_stamps) - last_intent_stamp_ns) / 1_000_000.0
        if stale_status_stamps and last_intent_stamp_ns
        else math.nan
    )
    stale_bound_ms = stale_detection_bound_ms(args.intent_timeout_ms, args.max_command_gap_ms)

    first_odom = odom[0][1]
    last_odom = odom[-1][1]
    net_displacement_m = math.hypot(
        last_odom.pose.pose.position.x - first_odom.pose.pose.position.x,
        last_odom.pose.pose.position.y - first_odom.pose.pose.position.y,
    )
    max_planar_linear_speed_mps = max(
        math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y) for _, msg in odom
    )
    max_abs_angular_speed_radps = max(abs(msg.twist.twist.angular.z) for _, msg in odom)

    counts = {topic: len(topic_samples) for topic, topic_samples in samples.items()}
    four_stream_counts_aligned = (
        len(
            {
                counts["/cmd_vel"],
                counts["/livifuser/cmd_vel_stamped"],
                counts["/livifuser/command_watchdog_status"],
                counts["/livifuser/command_watchdog_timing"],
            }
        )
        == 1
    )
    max_raw_gap_ms = max(raw_record_gaps)
    max_tick_ms = max(usable_ticks) * 1000.0
    max_publish_ms = max(usable_publishes) * 1000.0
    max_drops = max(msg.diagnostic_drops for _, msg in timings)
    max_external_publishers = max(msg.external_cmd_vel_publishers for _, msg in statuses)

    checks = {
        "all_raw_commands_zero": raw_max_abs == 0.0,
        "all_stamped_commands_zero": stamped_max_abs == 0.0,
        "all_intents_zero": intent_max_abs == 0.0,
        "all_status_outputs_zero": status_output_max_abs == 0.0,
        "final_five_raw_commands_zero": final_raw_zero,
        "final_five_stamped_commands_zero": final_stamped_zero,
        "five_final_shutdown_sentinels": final_five_are_sentinels,
        "final_five_stamps_aligned": final_stamps_aligned,
        "four_watchdog_stream_counts_aligned": four_stream_counts_aligned,
        "max_raw_record_gap_within_limit": max_raw_gap_ms <= args.max_command_gap_ms,
        "max_tick_interval_within_limit": max_tick_ms <= args.max_command_gap_ms,
        "max_publish_duration_within_limit": max_publish_ms <= args.max_publish_duration_ms,
        "diagnostic_drops_zero": max_drops == 0,
        "external_publishers_zero": max_external_publishers == 0,
        "odometry_displacement_negligible": net_displacement_m <= args.max_displacement_m,
        "odometry_speed_negligible": max_planar_linear_speed_mps <= args.max_linear_speed_mps,
        "stale_transition_within_derived_bound": math.isfinite(stale_transition_ms)
        and stale_transition_ms <= stale_bound_ms,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis_mode": "offline_deserialization_no_replay",
        "bag": {
            "path": str(args.bag),
            "mcap_sha256": sha256(mcap_paths[0]),
            "metadata_sha256": sha256(metadata_path),
        },
        "limits": {
            "intent_timeout_ms": args.intent_timeout_ms,
            "max_command_gap_ms": args.max_command_gap_ms,
            "stale_detection_bound_ms": stale_bound_ms,
            "max_publish_duration_ms": args.max_publish_duration_ms,
            "max_displacement_m": args.max_displacement_m,
            "max_linear_speed_mps": args.max_linear_speed_mps,
        },
        "counts": counts,
        "commands": {
            "raw_max_abs_all_components": raw_max_abs,
            "stamped_max_abs_all_components": stamped_max_abs,
            "intent_max_abs_all_components": intent_max_abs,
            "status_output_max_abs": status_output_max_abs,
            "final_raw_values": list(twist_values(raw[-1][1])),
            "final_stamped_values": list(twist_values(stamped[-1][1].twist)),
        },
        "shutdown": {
            "sentinel_count": len(shutdown_indices),
            "sentinel_indices": shutdown_indices,
            "final_stamps_ns": final_stamps,
        },
        "timing": {
            "max_raw_cmd_vel_record_gap_ms": max_raw_gap_ms,
            "max_internal_tick_interval_ms": max_tick_ms,
            "max_publish_duration_ms": max_publish_ms,
            "max_diagnostic_drops": max_drops,
            "max_external_publishers": max_external_publishers,
            "stale_transition_after_last_intent_ms": stale_transition_ms,
        },
        "status_reason_counts": dict(sorted(reason_counts.items())),
        "odometry": {
            "net_displacement_m": net_displacement_m,
            "max_planar_linear_speed_mps": max_planar_linear_speed_mps,
            "max_abs_angular_speed_radps": max_abs_angular_speed_radps,
        },
        "checks": checks,
        "accepted": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
