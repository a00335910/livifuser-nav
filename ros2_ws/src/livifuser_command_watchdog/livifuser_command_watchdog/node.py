"""ROS 2 wrapper for the fail-safe Stage 1 command watchdog.

The critical path is deliberately small. `_publish_command` copies an intent
snapshot, folds in a cached graph picture, publishes `/cmd_vel`, and returns.
It performs no discovery query, no logging, and no diagnostic publication,
because a block anywhere in that callback delays the *next* tick as well as the
current one. The 2026-07-30 keyboard bag recorded three such stalls of 1.17 s,
2.28 s and 3.60 s, each beginning immediately after a reason transition — that
is, immediately after the logger ran inside the command callback.

Diagnostics are handed to a bounded queue and published by an independent
timer. A dropped diagnostic does **not** merely cost a dataset row: the
exporter's 150 ms zero-order hold can carry the previous action across one
missing 10 Hz sample without rejecting the tick, so the loss is silent label
corruption. Any run with `diagnostic_drops > 0` must be rejected outright.
"""

from __future__ import annotations

import math
import queue
import signal
import threading
import time
from dataclasses import dataclass
from types import FrameType

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from livifuser_interfaces.msg import CommandWatchdogStatus, CommandWatchdogTiming
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions

from .policy import (
    CommandDecision,
    DecisionReason,
    GraphCache,
    VelocityIntent,
    WatchdogLimits,
    apply_graph_cache,
    count_external_publishers,
    decide_command,
    intent_from_message_fields,
)
from .shutdown_policy import FirstSignalGate

DIAGNOSTIC_QUEUE_DEPTH = 64
NOT_APPLICABLE_S = -1.0
FINAL_ZERO_BURST_COUNT = 5
FINAL_ZERO_BURST_PERIOD_S = 0.02


@dataclass(frozen=True)
class DiagnosticRecord:
    """One command tick, carried to the diagnostic path with its own stamp."""

    stamp_sec: int
    stamp_nanosec: int
    decision: CommandDecision
    external_publishers: int
    graph_cache_age_s: float
    tick_interval_s: float
    command_publish_duration_s: float


class CommandWatchdog(Node):
    def __init__(self) -> None:
        super().__init__("livifuser_command_watchdog")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("intent_timeout_ms", 250.0)
        self.declare_parameter("max_abs_linear_mps", 0.10)
        self.declare_parameter("max_abs_angular_radps", 0.50)
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("graph_probe_hz", 1.0)
        self.declare_parameter("graph_max_age_ms", 3000.0)

        rate_hz = float(self.get_parameter("rate_hz").value)
        if not math.isfinite(rate_hz) or not 5.0 <= rate_hz <= 20.0:
            raise ValueError("rate_hz must be finite and in [5, 20]")
        graph_probe_hz = float(self.get_parameter("graph_probe_hz").value)
        if not math.isfinite(graph_probe_hz) or not 0.2 <= graph_probe_hz <= 5.0:
            raise ValueError("graph_probe_hz must be finite and in [0.2, 5]")
        self._graph_max_age_s = float(self.get_parameter("graph_max_age_ms").value) / 1000.0
        if not math.isfinite(self._graph_max_age_s) or self._graph_max_age_s <= 0.0:
            raise ValueError("graph_max_age_ms must be finite and positive")

        self._limits = WatchdogLimits(
            timeout_s=float(self.get_parameter("intent_timeout_ms").value) / 1000.0,
            max_abs_linear_mps=float(self.get_parameter("max_abs_linear_mps").value),
            max_abs_angular_radps=float(self.get_parameter("max_abs_angular_radps").value),
        )
        self._frame_id = str(self.get_parameter("frame_id").value)
        if not self._frame_id:
            raise ValueError("frame_id must not be empty")

        self._command_period_s = 1.0 / rate_hz
        self._tick_overrun_s = 1.5 * self._command_period_s
        self._publish_overrun_s = 0.25 * self._command_period_s

        # Four independent groups so a stalled discovery query, a stalled log
        # write, or a burst of intent traffic cannot delay the command timer.
        self._command_group = MutuallyExclusiveCallbackGroup()
        self._intent_group = MutuallyExclusiveCallbackGroup()
        self._diagnostic_group = MutuallyExclusiveCallbackGroup()
        self._graph_group = MutuallyExclusiveCallbackGroup()

        intent_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._intent_subscription = self.create_subscription(
            TwistStamped,
            "/livifuser/teleop_intent_stamped",
            self._receive_intent,
            intent_qos,
            callback_group=self._intent_group,
        )
        self._command_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self._stamped_publisher = self.create_publisher(
            TwistStamped,
            "/livifuser/cmd_vel_stamped",
            10,
        )
        self._status_publisher = self.create_publisher(
            CommandWatchdogStatus,
            "/livifuser/command_watchdog_status",
            10,
        )
        self._timing_publisher = self.create_publisher(
            CommandWatchdogTiming,
            "/livifuser/command_watchdog_timing",
            10,
        )

        self._intent_lock = threading.Lock()
        self._intent: VelocityIntent | None = None
        self._intent_received_monotonic_s: float | None = None

        self._graph_lock = threading.Lock()
        self._graph_cache = GraphCache()

        self._counter_lock = threading.Lock()
        self._diagnostic_drops = 0

        self._diagnostics: queue.Queue = queue.Queue(maxsize=DIAGNOSTIC_QUEUE_DEPTH)
        self._last_reason: DecisionReason | None = None
        self._last_tick_monotonic_s: float | None = None

        self.create_timer(
            self._command_period_s,
            self._publish_command,
            callback_group=self._command_group,
        )
        # Drain faster than production so a single late drain cannot orphan rows.
        self.create_timer(
            self._command_period_s / 2.0,
            self._publish_diagnostics,
            callback_group=self._diagnostic_group,
        )
        self.create_timer(
            1.0 / graph_probe_hz,
            self._probe_graph,
            callback_group=self._graph_group,
        )
        self.get_logger().info(
            "Stage 1 command watchdog ready at "
            f"{rate_hz:.1f} Hz, timeout={self._limits.timeout_s * 1000:.0f} ms, "
            f"limits=({self._limits.max_abs_linear_mps:.3f} m/s, "
            f"{self._limits.max_abs_angular_radps:.3f} rad/s); "
            f"graph probe {graph_probe_hz:.1f} Hz, "
            f"max age {self._graph_max_age_s * 1000:.0f} ms. "
            "Output is zero until the first successful probe."
        )

    # ------------------------------------------------------------------
    # Intent path
    # ------------------------------------------------------------------
    def _receive_intent(self, message: TwistStamped) -> None:
        twist = message.twist
        stamp_is_set = message.header.stamp.sec != 0 or message.header.stamp.nanosec != 0
        intent = intent_from_message_fields(
            linear_x=twist.linear.x,
            linear_y=twist.linear.y,
            linear_z=twist.linear.z,
            angular_x=twist.angular.x,
            angular_y=twist.angular.y,
            angular_z=twist.angular.z,
            frame_id=message.header.frame_id,
            stamp_is_set=stamp_is_set,
            expected_frame_id=self._frame_id,
        )
        received_s = time.monotonic()
        with self._intent_lock:
            self._intent = intent
            self._intent_received_monotonic_s = received_s

    # ------------------------------------------------------------------
    # Critical path — no discovery, no logging, no diagnostic publication
    # ------------------------------------------------------------------
    def _publish_command(self) -> None:
        entered_s = time.monotonic()
        with self._intent_lock:
            intent = self._intent
            received_s = self._intent_received_monotonic_s
        with self._graph_lock:
            cache = self._graph_cache

        decision = decide_command(intent, received_s, entered_s, self._limits)
        decision = apply_graph_cache(decision, cache, entered_s, self._graph_max_age_s)

        stamp = self.get_clock().now().to_msg()
        command = Twist()
        command.linear.x = decision.output.linear_mps
        command.angular.z = decision.output.angular_radps

        publish_started_s = time.monotonic()
        self._command_publisher.publish(command)
        publish_finished_s = time.monotonic()

        interval_s = (
            NOT_APPLICABLE_S
            if self._last_tick_monotonic_s is None
            else entered_s - self._last_tick_monotonic_s
        )
        self._last_tick_monotonic_s = entered_s

        cache_age_s = (
            NOT_APPLICABLE_S
            if cache.probed_monotonic_s is None
            else entered_s - cache.probed_monotonic_s
        )
        self._enqueue_diagnostic(
            DiagnosticRecord(
                stamp_sec=stamp.sec,
                stamp_nanosec=stamp.nanosec,
                decision=decision,
                external_publishers=cache.external_publishers or 0,
                graph_cache_age_s=cache_age_s,
                tick_interval_s=interval_s,
                command_publish_duration_s=publish_finished_s - publish_started_s,
            )
        )

    def _enqueue_diagnostic(self, record: DiagnosticRecord) -> None:
        """Hand off without blocking; a full queue counts a drop and moves on."""

        try:
            self._diagnostics.put_nowait(record)
        except queue.Full:
            with self._counter_lock:
                self._diagnostic_drops += 1

    # ------------------------------------------------------------------
    # Diagnostic path — may block without endangering the robot
    # ------------------------------------------------------------------
    def _publish_diagnostics(self) -> None:
        while True:
            try:
                record = self._diagnostics.get_nowait()
            except queue.Empty:
                return
            self._emit_diagnostic(record)

    def _emit_diagnostic(
        self,
        record: DiagnosticRecord,
        *,
        log_events: bool = True,
    ) -> None:
        decision = record.decision
        command = Twist()
        command.linear.x = decision.output.linear_mps
        command.angular.z = decision.output.angular_radps

        stamped = TwistStamped()
        stamped.header.stamp.sec = record.stamp_sec
        stamped.header.stamp.nanosec = record.stamp_nanosec
        stamped.header.frame_id = self._frame_id
        stamped.twist = command
        self._stamped_publisher.publish(stamped)

        status = CommandWatchdogStatus()
        status.header.stamp.sec = record.stamp_sec
        status.header.stamp.nanosec = record.stamp_nanosec
        status.header.frame_id = self._frame_id
        status.reason = decision.reason.value
        status.intent_present = decision.intent_present
        status.clamped = decision.clamped
        status.intent_age_s = decision.intent_age_s
        status.requested_linear_mps = decision.requested.linear_mps
        status.requested_angular_radps = decision.requested.angular_radps
        status.output_linear_mps = decision.output.linear_mps
        status.output_angular_radps = decision.output.angular_radps
        status.external_cmd_vel_publishers = record.external_publishers
        self._status_publisher.publish(status)

        with self._counter_lock:
            drops = self._diagnostic_drops

        timing = CommandWatchdogTiming()
        timing.header.stamp.sec = record.stamp_sec
        timing.header.stamp.nanosec = record.stamp_nanosec
        timing.header.frame_id = self._frame_id
        timing.tick_interval_s = record.tick_interval_s
        timing.command_publish_duration_s = record.command_publish_duration_s
        timing.graph_cache_age_s = record.graph_cache_age_s
        timing.diagnostic_drops = drops
        self._timing_publisher.publish(timing)

        # Final safety publication must not enter the filesystem-backed logger.
        # The command, stamped command, status, and timing samples above remain
        # aligned and observable; transition/overrun logs are optional during
        # process teardown.
        if not log_events:
            return

        if record.tick_interval_s > self._tick_overrun_s:
            self.get_logger().warning(
                f"command tick interval {record.tick_interval_s * 1000:.0f} ms "
                f"exceeded {self._tick_overrun_s * 1000:.0f} ms"
            )
        if record.command_publish_duration_s > self._publish_overrun_s:
            self.get_logger().warning(
                f"/cmd_vel publish blocked for "
                f"{record.command_publish_duration_s * 1000:.0f} ms"
            )

        if decision.reason != self._last_reason:
            message = f"Command watchdog state: {decision.reason.value}"
            # Humble's rcutils logger binds severity to a Python call site. Do
            # not route info/warning through one stored method and call line.
            if decision.reason is DecisionReason.FRESH:
                self.get_logger().info(message)
            else:
                self.get_logger().warning(message)
            self._last_reason = decision.reason

    # ------------------------------------------------------------------
    # Graph probe — the only discovery query, off the critical path
    # ------------------------------------------------------------------
    def _probe_graph(self) -> None:
        try:
            endpoints = self.get_publishers_info_by_topic("/cmd_vel")
        except Exception as error:  # noqa: BLE001 - a failed probe must not kill the node
            # Deliberately do not refresh the cache: it ages out and forces zero.
            self.get_logger().warning(f"/cmd_vel graph probe failed: {error}")
            return
        identities = [(endpoint.node_name, endpoint.node_namespace) for endpoint in endpoints]
        count = count_external_publishers(
            identities,
            (self.get_name(), self.get_namespace()),
        )
        with self._graph_lock:
            self._graph_cache = GraphCache(count, time.monotonic())

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def publish_final_zero(self) -> None:
        """Publish a zero command without touching discovery."""

        command = Twist()
        self._command_publisher.publish(command)

        stamp = self.get_clock().now().to_msg()
        with self._graph_lock:
            cache = self._graph_cache
        decision = decide_command(None, None, time.monotonic(), self._limits)
        self._emit_diagnostic(
            DiagnosticRecord(
                stamp_sec=stamp.sec,
                stamp_nanosec=stamp.nanosec,
                decision=decision,
                external_publishers=cache.external_publishers or 0,
                graph_cache_age_s=NOT_APPLICABLE_S,
                tick_interval_s=NOT_APPLICABLE_S,
                command_publish_duration_s=NOT_APPLICABLE_S,
            ),
            log_events=False,
        )


def main(args: list[str] | None = None) -> None:
    signal_gate = FirstSignalGate()

    def request_shutdown(
        _signal_number: int,
        _frame: FrameType | None,
    ) -> None:
        """Keep ROS alive and ignore repeated signals during final cleanup."""

        if signal_gate.accept():
            raise KeyboardInterrupt

    previous_sigint = signal.signal(signal.SIGINT, request_shutdown)
    previous_sigterm = signal.signal(signal.SIGTERM, request_shutdown)
    node: CommandWatchdog | None = None
    executor: MultiThreadedExecutor | None = None
    try:
        # Humble's default SIGTERM handler shuts the context before `spin`
        # returns. That made `rclpy.ok()` false and skipped the final safety
        # publish under `systemctl stop`. Own both handlers so termination first
        # unwinds `spin` while publishers are still valid.
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        node = CommandWatchdog()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop worker callbacks before the final burst; otherwise a concurrent
        # timer callback could publish a stale nonzero command after our zero.
        if executor is not None:
            executor.shutdown()
        if node is not None and rclpy.ok():
            for index in range(FINAL_ZERO_BURST_COUNT):
                node.publish_final_zero()
                if index + 1 < FINAL_ZERO_BURST_COUNT:
                    time.sleep(FINAL_ZERO_BURST_PERIOD_S)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
