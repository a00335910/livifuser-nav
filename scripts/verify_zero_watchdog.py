#!/usr/bin/env python3
"""Verify the live watchdog using only zero-valued stamped operator intent."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from livifuser_command_watchdog.policy import stale_detection_bound_ms
from livifuser_interfaces.msg import CommandWatchdogStatus, CommandWatchdogTiming
from nav_msgs.msg import Odometry


def planar_values(message: Twist) -> tuple[float, ...]:
    return (
        message.linear.x,
        message.linear.y,
        message.linear.z,
        message.angular.x,
        message.angular.y,
        message.angular.z,
    )


def median_rate_hz(arrivals: list[float]) -> float:
    intervals = [
        later - earlier for earlier, later in zip(arrivals, arrivals[1:], strict=False)
    ]
    return 1.0 / statistics.median(intervals) if intervals else math.nan


def spin_until(node: Any, deadline: float) -> None:
    while time.monotonic() < deadline:
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(0.005)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-sec", type=float, default=1.2)
    parser.add_argument("--zero-intent-sec", type=float, default=1.5)
    parser.add_argument("--dropout-sec", type=float, default=1.0)
    parser.add_argument("--intent-rate-hz", type=float, default=20.0)
    parser.add_argument("--discovery-timeout-sec", type=float, default=5.0)
    parser.add_argument(
        "--max-command-gap-ms",
        type=float,
        default=150.0,
        help="largest tolerated interval between consecutive command ticks",
    )
    parser.add_argument(
        "--intent-timeout-ms",
        type=float,
        default=250.0,
        help="watchdog intent timeout used to derive the stale-detection bound",
    )
    parser.add_argument(
        "--max-publish-duration-ms",
        type=float,
        default=25.0,
        help="largest tolerated /cmd_vel publish call duration",
    )
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("livifuser_zero_watchdog_verifier")
    command_samples: list[tuple[float, tuple[float, ...]]] = []
    stamped_samples: list[tuple[float, int, str, tuple[float, ...]]] = []
    status_samples: list[tuple[float, dict[str, Any]]] = []
    timing_samples: list[tuple[float, dict[str, Any]]] = []
    odom_samples: list[tuple[float, float, float, float, float]] = []

    node.create_subscription(
        Twist,
        "/cmd_vel",
        lambda message: command_samples.append((time.monotonic(), planar_values(message))),
        100,
    )

    def receive_stamped(message: TwistStamped) -> None:
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        stamped_samples.append(
            (time.monotonic(), stamp_ns, message.header.frame_id, planar_values(message.twist))
        )

    node.create_subscription(TwistStamped, "/livifuser/cmd_vel_stamped", receive_stamped, 100)

    def receive_status(message: CommandWatchdogStatus) -> None:
        status_samples.append(
            (
                time.monotonic(),
                {
                    "reason": message.reason,
                    "intent_present": message.intent_present,
                    "clamped": message.clamped,
                    "intent_age_s": message.intent_age_s,
                    "requested": [
                        message.requested_linear_mps,
                        message.requested_angular_radps,
                    ],
                    "output": [message.output_linear_mps, message.output_angular_radps],
                    "external_publishers": message.external_cmd_vel_publishers,
                },
            )
        )

    node.create_subscription(
        CommandWatchdogStatus,
        "/livifuser/command_watchdog_status",
        receive_status,
        100,
    )

    def receive_timing(message: CommandWatchdogTiming) -> None:
        timing_samples.append(
            (
                time.monotonic(),
                {
                    "tick_interval_s": message.tick_interval_s,
                    "command_publish_duration_s": message.command_publish_duration_s,
                    "graph_cache_age_s": message.graph_cache_age_s,
                    "diagnostic_drops": message.diagnostic_drops,
                },
            )
        )

    node.create_subscription(
        CommandWatchdogTiming,
        "/livifuser/command_watchdog_timing",
        receive_timing,
        100,
    )

    def receive_odom(message: Odometry) -> None:
        odom_samples.append(
            (
                time.monotonic(),
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.twist.twist.linear.x,
                message.twist.twist.angular.z,
            )
        )

    node.create_subscription(Odometry, "/odom", receive_odom, 100)
    intent_publisher = node.create_publisher(
        TwistStamped,
        "/livifuser/teleop_intent_stamped",
        10,
    )

    issues: list[str] = []
    last_intent_monotonic: float | None = None
    try:
        discovery_deadline = time.monotonic() + args.discovery_timeout_sec
        command_nodes: list[str] = []
        while time.monotonic() < discovery_deadline:
            spin_until(node, time.monotonic() + 0.02)
            command_endpoints = node.get_publishers_info_by_topic("/cmd_vel")
            command_nodes = [endpoint.node_name for endpoint in command_endpoints]
            if (
                command_nodes == ["livifuser_command_watchdog"]
                and node.count_subscribers("/livifuser/teleop_intent_stamped") == 1
                and len(command_samples) >= 3
                and len(stamped_samples) >= 3
                and len(status_samples) >= 3
            ):
                break
        if command_nodes != ["livifuser_command_watchdog"]:
            issues.append(f"unexpected /cmd_vel publishers before test: {command_nodes}")

        command_samples.clear()
        stamped_samples.clear()
        status_samples.clear()
        timing_samples.clear()
        odom_samples.clear()

        spin_until(node, time.monotonic() + args.baseline_sec)

        zero_intent = TwistStamped()
        zero_intent.header.frame_id = "base_link"
        assert all(value == 0.0 for value in planar_values(zero_intent.twist))
        publish_period = 1.0 / args.intent_rate_hz
        publish_deadline = time.monotonic() + args.zero_intent_sec
        next_publish = time.monotonic()
        while time.monotonic() < publish_deadline:
            now = time.monotonic()
            if now >= next_publish:
                zero_intent.header.stamp = node.get_clock().now().to_msg()
                assert all(value == 0.0 for value in planar_values(zero_intent.twist))
                intent_publisher.publish(zero_intent)
                last_intent_monotonic = time.monotonic()
                next_publish += publish_period
            for _ in range(20):
                rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.002)

        spin_until(node, time.monotonic() + args.dropout_sec)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    max_abs_command = max(
        (abs(value) for _, values in command_samples for value in values),
        default=math.nan,
    )
    max_abs_stamped = max(
        (abs(value) for _, _, _, values in stamped_samples for value in values),
        default=math.nan,
    )
    reasons = Counter(sample["reason"] for _, sample in status_samples)
    stale_times = [
        arrival
        for arrival, sample in status_samples
        if sample["reason"] == "intent_stale"
        and last_intent_monotonic is not None
        and arrival >= last_intent_monotonic
    ]
    stale_transition_ms = (
        (min(stale_times) - last_intent_monotonic) * 1000.0
        if stale_times and last_intent_monotonic is not None
        else math.nan
    )

    if len(command_samples) < 25:
        issues.append(f"too few /cmd_vel samples: {len(command_samples)}")
    if len(stamped_samples) < 25:
        issues.append(f"too few stamped command samples: {len(stamped_samples)}")
    if not math.isfinite(max_abs_command) or max_abs_command != 0.0:
        issues.append(f"nonzero or absent raw command: max_abs={max_abs_command}")
    if not math.isfinite(max_abs_stamped) or max_abs_stamped != 0.0:
        issues.append(f"nonzero or absent stamped command: max_abs={max_abs_stamped}")
    if any(
        stamp_ns <= 0 or frame_id != "base_link"
        for _, stamp_ns, frame_id, _ in stamped_samples
    ):
        issues.append("stamped command has an unset timestamp or wrong frame")
    for required_reason in ("intent_missing", "fresh", "intent_stale"):
        if reasons[required_reason] == 0:
            issues.append(f"status never entered {required_reason}")
    if any(sample["external_publishers"] != 0 for _, sample in status_samples):
        issues.append("watchdog reported an external /cmd_vel publisher")
    if any(any(value != 0.0 for value in sample["output"]) for _, sample in status_samples):
        issues.append("watchdog status reported a nonzero output")
    stale_bound_ms = stale_detection_bound_ms(
        args.intent_timeout_ms,
        args.max_command_gap_ms,
    )
    if not math.isfinite(stale_transition_ms) or stale_transition_ms > stale_bound_ms:
        issues.append(
            f"stale transition exceeded derived {stale_bound_ms:.0f} ms bound: "
            f"{stale_transition_ms}"
        )

    odom_net_displacement_m = math.nan
    odom_max_abs_linear_mps = math.nan
    if len(odom_samples) >= 2:
        odom_net_displacement_m = math.hypot(
            odom_samples[-1][1] - odom_samples[0][1],
            odom_samples[-1][2] - odom_samples[0][2],
        )
        odom_max_abs_linear_mps = max(abs(sample[3]) for sample in odom_samples)
        if odom_net_displacement_m > 0.005:
            issues.append(f"odometry displacement exceeded 5 mm: {odom_net_displacement_m}")
        if odom_max_abs_linear_mps > 0.01:
            issues.append(f"odometry linear speed exceeded 0.01 m/s: {odom_max_abs_linear_mps}")
    else:
        issues.append(f"too few odometry samples: {len(odom_samples)}")

    # Timing stream. tick_interval_s and command_publish_duration_s are measured
    # inside the node, so they are authoritative on command-timer health wherever
    # this verifier runs. The arrival gap below also includes transport, so it is
    # corroboration rather than the primary signal when observing over Wi-Fi.
    def timing_values(key: str) -> list[float]:
        return [
            sample[key] for _, sample in timing_samples if sample[key] >= 0.0
        ]

    tick_intervals = timing_values("tick_interval_s")
    publish_durations = timing_values("command_publish_duration_s")
    graph_ages = timing_values("graph_cache_age_s")
    max_diagnostic_drops = max(
        (sample["diagnostic_drops"] for _, sample in timing_samples), default=-1
    )
    max_tick_interval_ms = max(tick_intervals) * 1000.0 if tick_intervals else math.nan
    max_publish_duration_ms = (
        max(publish_durations) * 1000.0 if publish_durations else math.nan
    )
    max_graph_cache_age_s = max(graph_ages) if graph_ages else math.nan
    command_arrivals = [sample[0] for sample in command_samples]
    command_arrival_gaps_ms = [
        (later - earlier) * 1000.0
        for earlier, later in zip(command_arrivals, command_arrivals[1:], strict=False)
    ]
    max_command_arrival_gap_ms = (
        max(command_arrival_gaps_ms) if command_arrival_gaps_ms else math.nan
    )

    if not timing_samples:
        issues.append("no /livifuser/command_watchdog_timing samples received")
    else:
        # The acceptance condition that documentation cannot enforce: a dropped
        # diagnostic is silent label corruption, because the exporter's 150 ms
        # zero-order hold carries the previous action across one missing sample.
        if max_diagnostic_drops > 0:
            issues.append(f"diagnostic_drops was nonzero: {max_diagnostic_drops}")
        if not tick_intervals:
            issues.append("timing stream carried no usable tick intervals")
        elif max_tick_interval_ms > args.max_command_gap_ms:
            issues.append(
                f"command tick interval exceeded {args.max_command_gap_ms:.0f} ms: "
                f"{max_tick_interval_ms:.1f} ms"
            )
        if publish_durations and max_publish_duration_ms > args.max_publish_duration_ms:
            issues.append(
                f"/cmd_vel publish duration exceeded "
                f"{args.max_publish_duration_ms:.0f} ms: {max_publish_duration_ms:.1f} ms"
            )

    if command_arrival_gaps_ms and max_command_arrival_gap_ms > args.max_command_gap_ms:
        issues.append(
            f"/cmd_vel arrival gap exceeded {args.max_command_gap_ms:.0f} ms: "
            f"{max_command_arrival_gap_ms:.1f} ms"
        )

    for forbidden_reason in ("cmd_vel_graph_unknown", "cmd_vel_graph_stale"):
        if reasons[forbidden_reason]:
            issues.append(
                f"graph cache was not trustworthy during the test: "
                f"{forbidden_reason} x{reasons[forbidden_reason]}"
            )

    # Status and timing are emitted together per tick, so their counts must agree.
    stream_counts = {
        "stamped_command": len(stamped_samples),
        "status": len(status_samples),
        "timing": len(timing_samples),
    }
    if stream_counts["timing"] and max(stream_counts.values()) - min(
        stream_counts.values()
    ) > 2:
        issues.append(f"diagnostic stream counts disagree: {stream_counts}")

    result = {
        "valid_zero_only_test": not issues,
        "issues": issues,
        "configuration": {
            "baseline_sec": args.baseline_sec,
            "zero_intent_sec": args.zero_intent_sec,
            "dropout_sec": args.dropout_sec,
            "intent_rate_hz": args.intent_rate_hz,
            "discovery_timeout_sec": args.discovery_timeout_sec,
            "intent_timeout_ms": args.intent_timeout_ms,
            "max_command_gap_ms": args.max_command_gap_ms,
            "stale_detection_bound_ms": stale_bound_ms,
            "max_publish_duration_ms": args.max_publish_duration_ms,
        },
        "command": {
            "count": len(command_samples),
            "median_rate_hz": median_rate_hz([sample[0] for sample in command_samples]),
            "max_abs_all_twist_components": max_abs_command,
        },
        "stamped_command": {
            "count": len(stamped_samples),
            "median_rate_hz": median_rate_hz([sample[0] for sample in stamped_samples]),
            "max_abs_all_twist_components": max_abs_stamped,
        },
        "status": {
            "count": len(status_samples),
            "reasons": dict(sorted(reasons.items())),
            "stale_transition_after_last_intent_ms": stale_transition_ms,
            "max_external_publishers": max(
                (sample["external_publishers"] for _, sample in status_samples),
                default=-1,
            ),
        },
        "timing": {
            "count": len(timing_samples),
            "max_diagnostic_drops": max_diagnostic_drops,
            "max_tick_interval_ms": max_tick_interval_ms,
            "max_command_publish_duration_ms": max_publish_duration_ms,
            "max_graph_cache_age_s": max_graph_cache_age_s,
            "max_command_arrival_gap_ms": max_command_arrival_gap_ms,
            "stream_counts": stream_counts,
        },
        "odometry": {
            "count": len(odom_samples),
            "net_displacement_m": odom_net_displacement_m,
            "max_abs_linear_mps": odom_max_abs_linear_mps,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
