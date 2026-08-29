"""Executor-isolation tests for the command watchdog ROS node.

These build the real `CommandWatchdog` and assert that a blocking discovery
probe, a blocking diagnostic path, or heavy intent traffic cannot delay the
`/cmd_vel` timer. The 2026-07-30 keyboard bag recorded 1.17 s, 2.28 s and
3.60 s command outages, each starting at a reason transition; these tests are
the regression barrier for that class of defect.

Patch targets matter here. `create_timer` binds the callback object at node
construction, so replacing `node._probe_graph` afterwards changes nothing the
timer consults. The tests therefore patch what the callback *body* looks up at
call time: `get_publishers_info_by_topic`, `_emit_diagnostic`, and the
`/cmd_vel` publisher.

SAFETY: this module publishes real `/cmd_vel` messages. `setUp` forces an
isolated ROS domain and localhost-only discovery immediately before
`rclpy.init`, then asserts containment rather than assuming it, so traffic never
leaves this machine. Do not run it with the robot powered on.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from pathlib import Path

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "livifuser_command_watchdog"
)
sys.path.insert(0, str(PACKAGE_ROOT))

# The middleware reads these when a context is created, not when rclpy is
# imported. Applying them only at module scope is order-dependent: `unittest
# discover` imports every test module before running any, so a module sorting
# later (`test_watchdog_sigterm`) overwrites ROS_DOMAIN_ID and these tests then
# run on its domain instead of 77. `setUp` reapplies them before `rclpy.init`.
ISOLATION_ENVIRONMENT = {
    "ROS_DOMAIN_ID": "77",
    "ROS_LOCALHOST_ONLY": "1",
    "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
}
os.environ.update(ISOLATION_ENVIRONMENT)

# One domain per test. Context shutdown does not retire the previous test's
# `/cmd_vel` publisher from discovery before the next watchdog probes, which put
# three of these tests into `cmd_vel_publisher_conflict` and had them measuring
# the forced-zero path instead of the nominal one. 79 belongs to
# `test_watchdog_sigterm`.
ISOLATED_DOMAIN_IDS = ("77", "78", "80", "81", "82", "83")
PUBLISHER_CONFLICT_REASON = "cmd_vel_publisher_conflict"
FRESH_REASON = "fresh"

try:
    import rclpy
    from geometry_msgs.msg import Twist, TwistStamped
    from livifuser_command_watchdog.node import CommandWatchdog
    from livifuser_interfaces.msg import CommandWatchdogStatus, CommandWatchdogTiming
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    ROS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only off the ROS host
    # `CommandListener` subclasses `Node` at module scope, which is evaluated on
    # import even when every test is skipped. Give it a base to inherit from.
    Node = object  # type: ignore[assignment, misc]
    ROS_AVAILABLE = False


COMMAND_PERIOD_S = 0.1
# Three periods. Generous enough to absorb scheduler noise on a loaded host,
# far below the 1.17 s minimum outage this guards against.
MAX_ACCEPTABLE_GAP_S = 0.3


class CommandListener(Node):
    """Records `/cmd_vel` arrival times plus the status and timing streams."""

    def __init__(self) -> None:
        super().__init__("command_watchdog_test_listener")
        self.command_times: list[float] = []
        self.command_values: list[tuple[float, float]] = []
        self.statuses: list[CommandWatchdogStatus] = []
        self.timings: list[CommandWatchdogTiming] = []
        self.create_subscription(Twist, "/cmd_vel", self._on_command, 10)
        self.create_subscription(
            CommandWatchdogStatus,
            "/livifuser/command_watchdog_status",
            self.statuses.append,
            10,
        )
        self.create_subscription(
            CommandWatchdogTiming,
            "/livifuser/command_watchdog_timing",
            self.timings.append,
            10,
        )

    def _on_command(self, message: Twist) -> None:
        self.command_times.append(time.monotonic())
        self.command_values.append((message.linear.x, message.angular.z))


@unittest.skipUnless(ROS_AVAILABLE, "rclpy/livifuser_interfaces unavailable")
class CommandWatchdogIsolationTests(unittest.TestCase):
    _domain_index = 0

    def setUp(self) -> None:
        os.environ.update(ISOLATION_ENVIRONMENT)
        index = CommandWatchdogIsolationTests._domain_index
        CommandWatchdogIsolationTests._domain_index += 1
        self.assertLess(
            index,
            len(ISOLATED_DOMAIN_IDS),
            "watchdog isolation test added without reserving another ROS domain",
        )
        os.environ["ROS_DOMAIN_ID"] = ISOLATED_DOMAIN_IDS[index]
        # Containment is asserted, not assumed: this module publishes /cmd_vel.
        self.assertEqual(os.environ["ROS_LOCALHOST_ONLY"], "1")
        self.assertIn(os.environ["ROS_DOMAIN_ID"], ISOLATED_DOMAIN_IDS)
        rclpy.init()
        self.watchdog = CommandWatchdog()
        self.listener = CommandListener()
        self.executor = MultiThreadedExecutor(num_threads=6)
        self.executor.add_node(self.watchdog)
        self.executor.add_node(self.listener)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def tearDown(self) -> None:
        self._stop.set()
        if self._thread.ident is not None:  # None until the thread is started
            self._thread.join(timeout=5.0)
        # Drain pending executor tasks, then let context shutdown reclaim the
        # nodes. Calling destroy_node() here races callbacks still in flight in
        # the thread pool, which rclpy surfaces as an unretrieved Future
        # exception ("cannot use Destroyable because destruction was requested").
        self.executor.shutdown(timeout_sec=5.0)
        # A thread-pool worker can already be inside a timer callback when
        # shutdown returns. Let it finish publishing before the context is
        # invalidated; otherwise rclpy reports "publisher's context is invalid".
        time.sleep(0.25)
        rclpy.shutdown()

    def _spin(self) -> None:
        while not self._stop.is_set():
            try:
                self.executor.spin_once(timeout_sec=0.05)
            except Exception:  # noqa: BLE001 - teardown races are not test failures
                return

    def _run_for(self, seconds: float) -> None:
        self._thread.start()
        time.sleep(seconds)
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._assert_no_publisher_conflict()

    def _assert_no_publisher_conflict(self) -> None:
        """Prove that a cadence assertion exercised the intended graph state."""

        self.assertGreater(
            len(self.listener.statuses),
            0,
            "watchdog produced no status samples, so graph isolation was not observed",
        )
        conflicts = [
            status.external_cmd_vel_publishers
            for status in self.listener.statuses
            if status.reason == PUBLISHER_CONFLICT_REASON
        ]
        self.assertEqual(
            conflicts,
            [],
            "test entered cmd_vel_publisher_conflict instead of its intended path; "
            f"external publisher counts: {conflicts}",
        )

    def _largest_gap(self) -> float:
        times = self.listener.command_times
        self.assertGreater(len(times), 5, "watchdog produced almost no commands")
        return max(b - a for a, b in zip(times, times[1:], strict=False))

    def test_blocking_graph_probe_does_not_interrupt_commands(self) -> None:
        """A discovery query that hangs must not stop the safety output.

        Patches the query itself, not `_probe_graph`: the timer holds a bound
        reference to the latter from construction.
        """

        entered = threading.Event()

        def hanging_query(topic: str) -> list[object]:
            entered.set()
            # Waits on the stop event rather than sleeping outright, so teardown
            # is not left joining a callback that is still deliberately blocked.
            self._stop.wait(1.5)
            return []

        self.watchdog.get_publishers_info_by_topic = hanging_query
        self._run_for(3.0)
        self.assertTrue(entered.is_set(), "the patched discovery query was never called")
        self.assertLess(self._largest_gap(), MAX_ACCEPTABLE_GAP_S)

    def test_blocking_diagnostic_path_does_not_interrupt_commands(self) -> None:
        """Logging or diagnostic publication stalling must not stop commands."""

        original = self.watchdog._emit_diagnostic

        def slow_emit(record: object) -> None:
            self._stop.wait(0.4)
            original(record)

        self.watchdog._emit_diagnostic = slow_emit
        self._run_for(3.0)
        self.assertLess(self._largest_gap(), MAX_ACCEPTABLE_GAP_S)

    def test_intent_flood_does_not_starve_the_command_timer(self) -> None:
        """Intent traffic well above the command rate must not delay commands."""

        publisher = self.listener.create_publisher(
            TwistStamped, "/livifuser/teleop_intent_stamped", 10
        )

        def flood() -> None:
            while not self._stop.is_set():
                try:
                    message = TwistStamped()
                    message.header.stamp = self.listener.get_clock().now().to_msg()
                    message.header.frame_id = "base_link"
                    message.twist.linear.x = 0.08
                    publisher.publish(message)
                except Exception:  # noqa: BLE001 - teardown race, not a failure
                    return
                time.sleep(0.002)

        flooder = threading.Thread(target=flood, daemon=True)
        self._thread.start()
        flooder.start()
        time.sleep(3.0)
        self._stop.set()
        flooder.join(timeout=2.0)
        self._thread.join(timeout=5.0)
        self._assert_no_publisher_conflict()
        self.assertLess(self._largest_gap(), MAX_ACCEPTABLE_GAP_S)
        self.assertTrue(
            any(
                status.reason == FRESH_REASON and status.output_linear_mps > 0.0
                for status in self.listener.statuses
            ),
            "intent flood never reached a fresh nonzero watchdog output",
        )
        self.assertTrue(
            any(linear > 0.0 for linear, _angular in self.listener.command_values),
            "listener never received a nonzero /cmd_vel command",
        )

    def test_shutdown_zero_performs_no_graph_query(self) -> None:
        """The final safety action must not depend on discovery."""

        calls: list[str] = []

        def record_query(topic: str) -> list[object]:
            calls.append(topic)
            return []

        self.watchdog.get_publishers_info_by_topic = record_query
        self.watchdog.publish_final_zero()
        self.assertEqual(calls, [])

    def test_command_timer_stall_is_reported_in_band(self) -> None:
        """An induced stall must surface in the timing topic, not just as a gap."""

        original_publish = self.watchdog._command_publisher.publish
        stalled = threading.Event()

        def stalling_publish(message: Twist) -> None:
            if not stalled.is_set():
                stalled.set()
                self._stop.wait(0.5)
            original_publish(message)

        self.watchdog._command_publisher.publish = stalling_publish
        self._run_for(3.0)
        intervals = [timing.tick_interval_s for timing in self.listener.timings]
        self.assertTrue(
            any(interval > 0.3 for interval in intervals),
            f"stall never appeared in tick_interval_s: {intervals}",
        )
        durations = [timing.command_publish_duration_s for timing in self.listener.timings]
        self.assertTrue(
            any(duration > 0.3 for duration in durations),
            f"blocked publish never appeared in command_publish_duration_s: {durations}",
        )

    def test_status_and_timing_share_a_stamp(self) -> None:
        """Timing joins to status by stamp, so the pair must be emitted together."""

        self._run_for(1.5)
        self.assertGreater(len(self.listener.timings), 5)
        status_stamps = {
            (status.header.stamp.sec, status.header.stamp.nanosec)
            for status in self.listener.statuses
        }
        timing_stamps = {
            (timing.header.stamp.sec, timing.header.stamp.nanosec)
            for timing in self.listener.timings
        }
        self.assertTrue(timing_stamps.issubset(status_stamps))


if __name__ == "__main__":
    unittest.main()
