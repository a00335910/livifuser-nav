"""Release-aware WSL GUI controller for protocol-clean acquisition.

The window receives actual key-down and key-up events, unlike a terminal. It
publishes raw operator intent only; the robot-local episode manager applies the
recording permit, local deadline, and freshness gate before the command watchdog
can see the intent. Losing focus, closing the window, or losing episode-state
updates produces zero.
"""

from __future__ import annotations

import math
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from livifuser_interfaces.msg import EpisodeState
from rclpy.node import Node
from std_msgs.msg import Empty

from .keyboard_policy import ZERO_COMMAND, ReleaseAwareKeyboard

RAW_INTENT_TOPIC = "/livifuser/operator_intent_stamped"
OPERATOR_STOP_TOPIC = "/livifuser/operator_stop"
EPISODE_STATE_TOPIC = "/livifuser/episode_state"
COMMAND_TOPIC = "/cmd_vel"
WATCHDOG_NODE = ("livifuser_command_watchdog", "/")
EPISODE_MANAGER_NODE = ("livifuser_episode_manager", "/")


class ReleaseKeyboardPublisher(Node):
    """Publish fresh raw intent while both key state and episode permit agree."""

    def __init__(self) -> None:
        super().__init__("livifuser_release_keyboard")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("linear_mps", 0.08)
        self.declare_parameter("angular_radps", 0.40)
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("max_runtime_s", 90.0)
        self.declare_parameter("graph_ready_timeout_s", 45.0)
        self.declare_parameter("episode_state_timeout_ms", 350.0)
        self.declare_parameter("episode_id", "")

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.max_runtime_s = float(self.get_parameter("max_runtime_s").value)
        self.graph_ready_timeout_s = float(self.get_parameter("graph_ready_timeout_s").value)
        self.state_timeout_s = float(self.get_parameter("episode_state_timeout_ms").value) / 1000.0
        self.episode_id = str(self.get_parameter("episode_id").value)
        if not math.isfinite(self.rate_hz) or not 5.0 <= self.rate_hz <= 20.0:
            raise ValueError("rate_hz must be finite and in [5, 20]")
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        if not math.isfinite(self.max_runtime_s) or self.max_runtime_s <= 0.0:
            raise ValueError("max_runtime_s must be finite and positive")
        if not math.isfinite(self.graph_ready_timeout_s) or self.graph_ready_timeout_s <= 0.0:
            raise ValueError("graph_ready_timeout_s must be finite and positive")
        if not math.isfinite(self.state_timeout_s) or self.state_timeout_s <= 0.0:
            raise ValueError("episode_state_timeout_ms must be finite and positive")

        self._keyboard = ReleaseAwareKeyboard(
            linear_mps=float(self.get_parameter("linear_mps").value),
            angular_radps=float(self.get_parameter("angular_radps").value),
        )
        self._state_arrival: float | None = None
        self._motion_permitted = False
        self._phase = "waiting"
        self._reason = "waiting_for_episode_state"
        self._recording_elapsed_s = 0.0
        self._recording_remaining_s = 0.0
        self._episode_done = False
        self._publisher = self.create_publisher(TwistStamped, RAW_INTENT_TOPIC, 10)
        self._stop_publisher = self.create_publisher(Empty, OPERATOR_STOP_TOPIC, 10)
        self.create_subscription(EpisodeState, EPISODE_STATE_TOPIC, self._on_state, 10)

    def _on_state(self, message: EpisodeState) -> None:
        matches = not self.episode_id or message.episode_id == self.episode_id
        self._motion_permitted = bool(matches and message.motion_permitted)
        self._state_arrival = time.monotonic()
        if matches:
            self._phase = str(message.phase)
            self._reason = str(message.reason)
            self._recording_elapsed_s = float(message.recording_elapsed_s)
            self._recording_remaining_s = float(message.recording_remaining_s)
            self._episode_done = self._phase in {"complete", "failed"}
        if not self._motion_permitted:
            self._keyboard.clear()

    def verify_graph(self) -> None:
        command_publishers = {
            (endpoint.node_name, endpoint.node_namespace)
            for endpoint in self.get_publishers_info_by_topic(COMMAND_TOPIC)
        }
        if command_publishers != {WATCHDOG_NODE}:
            raise RuntimeError(
                f"expected only {WATCHDOG_NODE} on {COMMAND_TOPIC}, got "
                f"{sorted(command_publishers)}"
            )
        raw_subscribers = {
            (endpoint.node_name, endpoint.node_namespace)
            for endpoint in self.get_subscriptions_info_by_topic(RAW_INTENT_TOPIC)
        }
        if EPISODE_MANAGER_NODE not in raw_subscribers:
            raise RuntimeError("robot-local episode-manager subscription is not visible")

    def key_down(self, key: str) -> None:
        self._keyboard.key_down(key)

    def key_up(self, key: str) -> None:
        self._keyboard.key_up(key)

    def clear(self) -> None:
        self._keyboard.clear()

    def request_episode_stop(self) -> None:
        """Request a normal cooldown/finalization instead of dropping the GUI."""

        self.clear()
        self.publish_current()
        self._stop_publisher.publish(Empty())

    def _current_command(self):
        now = time.monotonic()
        state_fresh = (
            self._state_arrival is not None
            and 0.0 <= now - self._state_arrival <= self.state_timeout_s
        )
        if not state_fresh or not self._motion_permitted:
            self._keyboard.clear()
            return ZERO_COMMAND
        return self._keyboard.command

    def publish_current(self) -> None:
        command = self._current_command()
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.twist.linear.x = command.linear_mps
        message.twist.angular.z = command.angular_radps
        self._publisher.publish(message)

    def publish_final_zero(self) -> None:
        self.clear()
        for _ in range(5):
            self.publish_current()
            time.sleep(0.05)


def _event_key(event: object) -> str:
    char = str(getattr(event, "char", ""))
    if char:
        return char.lower()
    return str(getattr(event, "keysym", "")).lower()


def main() -> None:
    # Import lazily so policy tests and ROS package inspection do not require a
    # display or Tk installation.
    import tkinter as tk

    rclpy.init()
    node = ReleaseKeyboardPublisher()
    root = tk.Tk()
    root.title("LiViFuser release-aware teleop")
    root.geometry("620x330")
    root.resizable(False, False)
    started = time.monotonic()
    period_ms = max(1, round(1000.0 / node.rate_hz))
    closing = False
    stop_requested = False
    done_seen_at: float | None = None

    episode_var = tk.StringVar(value=f"Episode: {node.episode_id or '(unbound)'}")
    phase_var = tk.StringVar(value="Phase: waiting for robot")
    timer_var = tk.StringVar(value="--.- s")
    reason_var = tk.StringVar(value="Waiting for episode state")

    def finish_window() -> None:
        nonlocal closing
        if closing:
            return
        closing = True
        node.clear()
        root.quit()

    def request_stop() -> None:
        nonlocal stop_requested
        if stop_requested or node._episode_done:
            return
        stop_requested = True
        node.request_episode_stop()
        reason_var.set("Stop requested; recording zero cooldown and finalizing...")

    def key_down(event: object) -> None:
        key = _event_key(event)
        if key == "escape":
            request_stop()
            return
        node.key_down(key)

    def key_up(event: object) -> None:
        node.key_up(_event_key(event))

    def focus_lost(_event: object) -> None:
        node.clear()
        node.publish_current()

    def tick() -> None:
        nonlocal done_seen_at
        if closing:
            return
        if time.monotonic() - started >= node.max_runtime_s:
            request_stop()
        rclpy.spin_once(node, timeout_sec=0.0)
        node.publish_current()
        phase_var.set(f"Phase: {node._phase}")
        timer_var.set(f"{node._recording_remaining_s:05.1f} s remaining")
        reason_var.set(node._reason if not stop_requested else reason_var.get())
        if node._episode_done:
            if done_seen_at is None:
                done_seen_at = time.monotonic()
                node.clear()
                reason_var.set(f"Episode {node._phase}: {node._reason}")
            elif time.monotonic() - done_seen_at >= 1.0:
                finish_window()
                return
        root.after(period_ms, tick)

    try:
        deadline = time.monotonic() + node.graph_ready_timeout_s
        while True:
            try:
                node.verify_graph()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                rclpy.spin_once(node, timeout_sec=0.1)

        tk.Label(
            root,
            textvariable=episode_var,
            font=("TkDefaultFont", 12, "bold"),
        ).pack(pady=(16, 4))
        tk.Label(
            root,
            textvariable=timer_var,
            font=("TkDefaultFont", 24, "bold"),
        ).pack(pady=4)
        tk.Label(root, textvariable=phase_var).pack()
        tk.Label(
            root,
            text="Hold one key to move; release stops immediately",
            font=("TkDefaultFont", 14, "bold"),
        ).pack(pady=(12, 8))
        tk.Label(root, text="u = forward-left    i = forward    o = forward-right").pack()
        tk.Label(root, text="k / focus loss = motion STOP    Esc = finish episode").pack(pady=6)
        tk.Label(
            root,
            text="Motion is also gated by the robot-local recording window.",
        ).pack()
        tk.Label(root, textvariable=reason_var, wraplength=580).pack(pady=5)
        tk.Button(
            root,
            text="STOP & FINALIZE EPISODE",
            command=request_stop,
            bg="#b71c1c",
            fg="white",
        ).pack(pady=4)
        root.bind_all("<KeyPress>", key_down)
        root.bind_all("<KeyRelease>", key_up)
        root.bind("<FocusOut>", focus_lost)
        root.protocol("WM_DELETE_WINDOW", request_stop)
        root.after(period_ms, tick)
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_final_zero()
        root.destroy()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
