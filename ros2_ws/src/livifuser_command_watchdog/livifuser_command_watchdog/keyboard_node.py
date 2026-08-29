"""Fresh-stamped 10 Hz keyboard intent publisher for supervised acquisition.

Unlike Ubuntu Humble's event-only ``teleop_twist_keyboard`` 2.4.1, this node
publishes the selected command continuously with a new ``base_link`` timestamp.
Commands are deliberately latched: ``k`` is the explicit stop key. Losing this
process or its network path still makes the robot-local watchdog stop after its
250 ms intent timeout. This node never publishes ``/cmd_vel``.
"""

from __future__ import annotations

import math
import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node

from .keyboard_policy import ZERO_COMMAND, command_for_key

INTENT_TOPIC = "/livifuser/teleop_intent_stamped"
COMMAND_TOPIC = "/cmd_vel"
WATCHDOG_NODE = ("livifuser_command_watchdog", "/")
ALLOWED_INTENT_SUBSCRIBERS = {
    WATCHDOG_NODE,
    ("rosbag2_recorder", "/"),
}


class KeyboardIntentPublisher(Node):
    """Publish a latched keyboard selection with a fresh stamp every tick."""

    def __init__(self) -> None:
        super().__init__("livifuser_keyboard_teleop")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("linear_mps", 0.08)
        self.declare_parameter("angular_radps", 0.40)
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("max_runtime_s", 70.0)

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.linear_mps = float(self.get_parameter("linear_mps").value)
        self.angular_radps = float(self.get_parameter("angular_radps").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.max_runtime_s = float(self.get_parameter("max_runtime_s").value)

        if not math.isfinite(self.rate_hz) or not 5.0 <= self.rate_hz <= 20.0:
            raise ValueError("rate_hz must be finite and in [5, 20]")
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        if not math.isfinite(self.max_runtime_s) or self.max_runtime_s <= 0.0:
            raise ValueError("max_runtime_s must be finite and positive")
        # Validate the configured limits at startup, before creating authority.
        command_for_key(
            "i",
            linear_mps=self.linear_mps,
            angular_radps=self.angular_radps,
        )

        self._publisher = self.create_publisher(TwistStamped, INTENT_TOPIC, 10)
        self._command = ZERO_COMMAND

    def verify_graph(self) -> None:
        """Require the watchdog as sole command authority before accepting keys."""

        cmd_publishers = {
            (endpoint.node_name, endpoint.node_namespace)
            for endpoint in self.get_publishers_info_by_topic(COMMAND_TOPIC)
        }
        if cmd_publishers != {WATCHDOG_NODE}:
            raise RuntimeError(
                f"expected only {WATCHDOG_NODE} on {COMMAND_TOPIC}, got "
                f"{sorted(cmd_publishers)}"
            )

        intent_subscribers = {
            (endpoint.node_name, endpoint.node_namespace)
            for endpoint in self.get_subscriptions_info_by_topic(INTENT_TOPIC)
        }
        if WATCHDOG_NODE not in intent_subscribers:
            raise RuntimeError("watchdog intent subscription is not visible")
        unexpected = intent_subscribers - ALLOWED_INTENT_SUBSCRIBERS
        if unexpected:
            raise RuntimeError(f"unexpected intent subscribers: {sorted(unexpected)}")

    def select_key(self, key: str) -> None:
        self._command = command_for_key(
            key,
            linear_mps=self.linear_mps,
            angular_radps=self.angular_radps,
        )

    def publish_current(self) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.twist.linear.x = self._command.linear_mps
        message.twist.angular.z = self._command.angular_radps
        self._publisher.publish(message)

    def publish_final_zero(self) -> None:
        self._command = ZERO_COMMAND
        for _ in range(5):
            self.publish_current()
            time.sleep(0.05)


def _print_banner(node: KeyboardIntentPublisher) -> None:
    print(
        "\nLiViFuser latched keyboard intent (fresh-stamped at "
        f"{node.rate_hz:.1f} Hz)\n"
        "  u  i  o     forward-left / forward / forward-right\n"
        "  j  k  l     rotate-left / STOP / rotate-right\n"
        "  m  ,  .     reverse-left / reverse / reverse-right\n\n"
        f"Limits: {node.linear_mps:.2f} m/s, +/-{node.angular_radps:.2f} rad/s\n"
        "Commands latch until another key; k is the explicit stop. Ctrl-C exits.\n"
    )


def main() -> None:
    rclpy.init()
    node = KeyboardIntentPublisher()
    old_settings = termios.tcgetattr(sys.stdin)
    started = time.monotonic()
    period_s = 1.0 / node.rate_hz
    next_publish = started

    try:
        # Give DDS discovery one bounded opportunity before accepting input.
        deadline = time.monotonic() + 5.0
        while True:
            try:
                node.verify_graph()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                rclpy.spin_once(node, timeout_sec=0.1)

        tty.setcbreak(sys.stdin.fileno())
        _print_banner(node)
        node.publish_current()

        while rclpy.ok():
            now = time.monotonic()
            if now - started >= node.max_runtime_s:
                print("Maximum runtime reached; publishing final zero.")
                break

            timeout_s = max(0.0, min(0.05, next_publish - now))
            readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
            if readable:
                key = sys.stdin.read(1)
                if not key or key == "\x03":
                    break
                node.select_key(key)

            now = time.monotonic()
            if now >= next_publish:
                node.publish_current()
                next_publish += period_s
                if next_publish < now - period_s:
                    next_publish = now + period_s
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_final_zero()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
