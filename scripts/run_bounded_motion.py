"""Publish one bounded stamped intent through the Stage 1 command watchdog."""

from __future__ import annotations

import argparse
import time

import rclpy
from geometry_msgs.msg import TwistStamped


def publish_for(
    node: object,
    publisher: object,
    message: TwistStamped,
    duration_sec: float,
    rate_hz: float,
) -> None:
    period = 1.0 / rate_hz
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        message.header.stamp = node.get_clock().now().to_msg()
        publisher.publish(message)
        time.sleep(period)


def wait_for_exclusive_watchdog(node: object, timeout_sec: float = 3.0) -> None:
    """Require one watchdog as the sole `/cmd_vel` publisher before intent flows."""

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        command_publishers = node.get_publishers_info_by_topic("/cmd_vel")
        watchdog_publishers = [
            endpoint
            for endpoint in command_publishers
            if endpoint.node_name == "livifuser_command_watchdog"
        ]
        if (
            len(command_publishers) == 1
            and len(watchdog_publishers) == 1
            and node.count_subscribers("/livifuser/teleop_intent_stamped") == 1
            and node.count_subscribers("/cmd_vel") >= 1
        ):
            return
        rclpy.spin_once(node, timeout_sec=0.05)
    raise RuntimeError(
        "exclusive watchdog graph not ready: require exactly one /cmd_vel publisher "
        "named livifuser_command_watchdog and one intent subscriber"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-stop-sec", type=float, default=3.0)
    parser.add_argument("--linear-mps", type=float, default=0.05)
    parser.add_argument("--move-sec", type=float, default=2.0)
    parser.add_argument("--post-stop-sec", type=float, default=5.0)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    args = parser.parse_args()

    if not 0.0 < args.linear_mps <= 0.1:
        raise ValueError("linear-mps must be in (0, 0.1]")
    if not 0.0 < args.move_sec <= 3.0:
        raise ValueError("move-sec must be in (0, 3]")
    if min(args.pre_stop_sec, args.post_stop_sec) < 1.0:
        raise ValueError("pre/post stop durations must be at least 1 second")
    if not 5.0 <= args.rate_hz <= 20.0:
        raise ValueError("rate-hz must be in [5, 20]")

    rclpy.init()
    node = rclpy.create_node("livifuser_bounded_motion")
    publisher = node.create_publisher(
        TwistStamped,
        "/livifuser/teleop_intent_stamped",
        10,
    )
    zero = TwistStamped()
    zero.header.frame_id = "base_link"
    forward = TwistStamped()
    forward.header.frame_id = "base_link"
    forward.twist.linear.x = args.linear_mps

    try:
        wait_for_exclusive_watchdog(node)
        publish_for(node, publisher, zero, args.pre_stop_sec, args.rate_hz)
        publish_for(node, publisher, forward, args.move_sec, args.rate_hz)
    finally:
        publish_for(node, publisher, zero, args.post_stop_sec, args.rate_hz)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
