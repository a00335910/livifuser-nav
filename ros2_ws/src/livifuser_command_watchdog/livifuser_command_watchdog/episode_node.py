"""Robot-local readiness, intent gate, deadline, and rosbag episode manager."""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from livifuser_interfaces.msg import (
    CommandWatchdogStatus,
    CommandWatchdogTiming,
    EpisodeState,
    RelativeGoal,
)
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Empty
from tf2_msgs.msg import TFMessage

from .episode_policy import (
    EpisodeConfig,
    EpisodeIdentity,
    EpisodeLifecycle,
    EpisodePhase,
    GoalReachTracker,
    OperatorIntent,
    StreamObservation,
    StreamRequirement,
    evaluate_readiness,
    gate_operator_intent,
)
from .keyboard_policy import ZERO_COMMAND, KeyboardCommand
from .recorder_qos import build_record_command, installed_qos_override_path
from .shutdown_policy import FirstSignalGate

RAW_INTENT_TOPIC = "/livifuser/operator_intent_stamped"
OPERATOR_STOP_TOPIC = "/livifuser/operator_stop"
GATED_INTENT_TOPIC = "/livifuser/teleop_intent_stamped"
EPISODE_STATE_TOPIC = "/livifuser/episode_state"
COMMAND_TOPIC = "/cmd_vel"
WATCHDOG_NODE = ("livifuser_command_watchdog", "/")
RELEASE_KEYBOARD_NODE = ("livifuser_release_keyboard", "/")
RECORDER_NODE_NAME = "rosbag2_recorder"

RECORD_TOPICS = (
    "/camera/image_raw",
    "/camera/camera_info",
    "/scan",
    "/cmd_vel",
    "/livifuser/cmd_vel_stamped",
    GATED_INTENT_TOPIC,
    RAW_INTENT_TOPIC,
    OPERATOR_STOP_TOPIC,
    "/livifuser/command_watchdog_status",
    "/livifuser/command_watchdog_timing",
    EPISODE_STATE_TOPIC,
    "/odom",
    "/livifuser/goal_relative",
    "/tf",
    "/tf_static",
)

READINESS_REQUIREMENTS = (
    StreamRequirement("/camera/image_raw", 3, 0.20),
    StreamRequirement("/camera/camera_info", 1, 1.00),
    StreamRequirement("/scan", 3, 0.25),
    StreamRequirement("/cmd_vel", 3, 0.25),
    StreamRequirement("/livifuser/cmd_vel_stamped", 3, 0.25),
    StreamRequirement("/livifuser/command_watchdog_status", 3, 0.25),
    StreamRequirement("/livifuser/command_watchdog_timing", 3, 0.25),
    StreamRequirement(RAW_INTENT_TOPIC, 3, 0.25),
    StreamRequirement("/odom", 3, 0.20),
    StreamRequirement("/livifuser/goal_relative", 3, 0.25),
    StreamRequirement("/tf", 1, 0.50),
    StreamRequirement("/tf_static", 1, None),
)


class ProtocolEpisodeManager(Node):
    """Gate host intent and own the complete robot-local recording lifecycle."""

    def __init__(self) -> None:
        super().__init__("livifuser_episode_manager")
        self.declare_parameter("episode_id", "")
        self.declare_parameter("output_path", "")
        self.declare_parameter("environment_id", "")
        self.declare_parameter("split", "")
        self.declare_parameter("route_id", "")
        self.declare_parameter("layout_id", "")
        self.declare_parameter("code_revision", "")
        self.declare_parameter("duration_s", 45.0)
        self.declare_parameter("zero_warmup_s", 2.0)
        self.declare_parameter("zero_cooldown_s", 2.0)
        self.declare_parameter("operator_timeout_ms", 250.0)
        self.declare_parameter("linear_mps", 0.08)
        self.declare_parameter("angular_radps", 0.40)
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("preflight_timeout_s", 30.0)
        self.declare_parameter("recorder_ready_timeout_s", 15.0)
        self.declare_parameter("goal_tolerance_m", 0.25)
        self.declare_parameter("goal_required_samples", 3)
        self.declare_parameter("minimum_free_gib", 1.5)
        self.declare_parameter("storage_id", "mcap")
        self.declare_parameter("qos_overrides_path", "")
        self.declare_parameter("frame_id", "base_link")

        self.episode_id = str(self.get_parameter("episode_id").value)
        self.environment_id = str(self.get_parameter("environment_id").value)
        self.split = str(self.get_parameter("split").value)
        self.route_id = str(self.get_parameter("route_id").value)
        self.layout_id = str(self.get_parameter("layout_id").value)
        self.code_revision = str(self.get_parameter("code_revision").value)
        output_text = str(self.get_parameter("output_path").value)
        self.identity = EpisodeIdentity(
            episode_id=self.episode_id,
            environment_id=self.environment_id,
            split=self.split,
            route_id=self.route_id,
            layout_id=self.layout_id,
            code_revision=self.code_revision,
        )
        if not output_text:
            raise ValueError("output_path is required")
        self.output_path = Path(output_text)
        if not self.output_path.is_absolute():
            raise ValueError("output_path must be absolute on the robot")
        self.identity.validate_output_basename(self.output_path.name)
        self.result_path = self.output_path.with_name(self.output_path.name + ".episode.json")
        if self.output_path.exists() or self.result_path.exists():
            raise FileExistsError("episode output or result sidecar already exists")
        if not self.output_path.parent.is_dir():
            raise FileNotFoundError("episode output parent does not exist")

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.preflight_timeout_s = float(self.get_parameter("preflight_timeout_s").value)
        self.recorder_ready_timeout_s = float(
            self.get_parameter("recorder_ready_timeout_s").value
        )
        self.minimum_free_bytes = int(
            float(self.get_parameter("minimum_free_gib").value) * 1024**3
        )
        self.storage_id = str(self.get_parameter("storage_id").value)
        qos_overrides_text = str(self.get_parameter("qos_overrides_path").value)
        self.qos_overrides_path = (
            Path(qos_overrides_text)
            if qos_overrides_text
            else installed_qos_override_path()
        )
        self.frame_id = str(self.get_parameter("frame_id").value)
        scalar_values = (
            self.rate_hz,
            self.preflight_timeout_s,
            self.recorder_ready_timeout_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in scalar_values):
            raise ValueError("episode-manager rates and timeouts must be finite and positive")
        if not 5.0 <= self.rate_hz <= 20.0:
            raise ValueError("rate_hz must be in [5, 20]")
        if self.minimum_free_bytes <= 0:
            raise ValueError("minimum_free_gib must be positive")
        if not self.storage_id or not self.frame_id:
            raise ValueError("storage_id and frame_id must not be empty")
        if not self.qos_overrides_path.is_file():
            raise FileNotFoundError(
                f"rosbag QoS override does not exist: {self.qos_overrides_path}"
            )

        config = EpisodeConfig(
            duration_s=float(self.get_parameter("duration_s").value),
            zero_warmup_s=float(self.get_parameter("zero_warmup_s").value),
            zero_cooldown_s=float(self.get_parameter("zero_cooldown_s").value),
            operator_timeout_s=(
                float(self.get_parameter("operator_timeout_ms").value) / 1000.0
            ),
            linear_mps=float(self.get_parameter("linear_mps").value),
            angular_radps=float(self.get_parameter("angular_radps").value),
        )
        now = time.monotonic()
        self.lifecycle = EpisodeLifecycle(config, now_monotonic_s=now)
        self._started_monotonic_s = now
        self._started_wall_s = time.time()
        self._latest_operator_intent: OperatorIntent | None = None
        self._observations: dict[str, StreamObservation] = {}
        self._goal_rho_m: float | None = None
        self._goal_reached = GoalReachTracker(
            tolerance_m=float(self.get_parameter("goal_tolerance_m").value),
            required_samples=int(self.get_parameter("goal_required_samples").value),
        )
        self._sequence = 0
        self._done = False
        self._recorder: subprocess.Popen[str] | None = None
        self._recorder_return_code: int | None = None
        self._last_readiness_reasons: tuple[str, ...] = ()
        self._last_gate_reason = "episode_not_recording"

        self._intent_publisher = self.create_publisher(
            TwistStamped, GATED_INTENT_TOPIC, 10
        )
        self._state_publisher = self.create_publisher(
            EpisodeState, EPISODE_STATE_TOPIC, 10
        )
        self.create_subscription(
            TwistStamped, RAW_INTENT_TOPIC, self._on_operator_intent, 10
        )
        self.create_subscription(Empty, OPERATOR_STOP_TOPIC, self._on_operator_stop, 10)
        self._create_readiness_subscriptions()
        self.create_timer(1.0 / self.rate_hz, self._tick)

        self.get_logger().info(
            f"Episode {self.episode_id} waiting for {len(READINESS_REQUIREMENTS)} "
            "required streams; motion remains locally gated to zero."
        )

    @property
    def done(self) -> bool:
        return self._done

    def _create_readiness_subscriptions(self) -> None:
        default = 10
        sensor_qos = qos_profile_sensor_data
        static_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        specifications = (
            ("/camera/image_raw", Image, sensor_qos),
            ("/camera/camera_info", CameraInfo, sensor_qos),
            ("/scan", LaserScan, sensor_qos),
            ("/cmd_vel", Twist, sensor_qos),
            ("/livifuser/cmd_vel_stamped", TwistStamped, sensor_qos),
            (
                "/livifuser/command_watchdog_status",
                CommandWatchdogStatus,
                sensor_qos,
            ),
            (
                "/livifuser/command_watchdog_timing",
                CommandWatchdogTiming,
                sensor_qos,
            ),
            ("/odom", Odometry, sensor_qos),
            ("/tf", TFMessage, sensor_qos),
            ("/tf_static", TFMessage, static_qos),
        )
        for topic, message_type, qos in specifications:
            self.create_subscription(
                message_type,
                topic,
                lambda _message, observed_topic=topic: self._observe(observed_topic),
                qos,
            )
        self.create_subscription(
            RelativeGoal,
            "/livifuser/goal_relative",
            self._on_goal,
            default,
        )

    def _observe(self, topic: str) -> None:
        previous = self._observations.get(topic, StreamObservation(0, None))
        self._observations[topic] = StreamObservation(
            previous.message_count + 1,
            time.monotonic(),
        )

    def _on_goal(self, message: RelativeGoal) -> None:
        self._observe("/livifuser/goal_relative")
        self._goal_rho_m = float(message.rho_m)
        if self.lifecycle.phase is EpisodePhase.RECORDING and self._goal_reached.update(
            self._goal_rho_m
        ):
            self.lifecycle.request_stop(
                "goal_reached",
                now_monotonic_s=time.monotonic(),
            )

    def _on_operator_intent(self, message: TwistStamped) -> None:
        self._observe(RAW_INTENT_TOPIC)
        unsupported = (
            message.twist.linear.y,
            message.twist.linear.z,
            message.twist.angular.x,
            message.twist.angular.y,
        )
        stamp_is_set = bool(message.header.stamp.sec or message.header.stamp.nanosec)
        structurally_valid = (
            message.header.frame_id == self.frame_id
            and stamp_is_set
            and all(math.isfinite(value) and abs(value) <= 1e-9 for value in unsupported)
        )
        self._latest_operator_intent = OperatorIntent(
            KeyboardCommand(
                float(message.twist.linear.x),
                float(message.twist.angular.z),
            ),
            time.monotonic(),
            structurally_valid,
        )

    def _on_operator_stop(self, _message: Empty) -> None:
        """End a valid recording cleanly when the local operator requests it."""

        now = time.monotonic()
        self._latest_operator_intent = OperatorIntent(
            ZERO_COMMAND,
            now,
            True,
        )
        if self.lifecycle.phase is EpisodePhase.RECORDING:
            self.lifecycle.request_stop("operator_stop", now_monotonic_s=now)
        elif self.lifecycle.phase not in (EpisodePhase.COMPLETE, EpisodePhase.FAILED):
            self.abort("operator_cancelled_before_recording")

    def _verify_command_authority(self) -> bool:
        publishers = {
            (endpoint.node_name, endpoint.node_namespace)
            for endpoint in self.get_publishers_info_by_topic(COMMAND_TOPIC)
        }
        return publishers == {WATCHDOG_NODE}

    def _verify_operator_authority(self) -> bool:
        publishers = {
            (endpoint.node_name, endpoint.node_namespace)
            for endpoint in self.get_publishers_info_by_topic(RAW_INTENT_TOPIC)
        }
        return publishers == {RELEASE_KEYBOARD_NODE}

    def _recorder_subscriptions_ready(self) -> bool:
        for topic in RECORD_TOPICS:
            endpoints = self.get_subscriptions_info_by_topic(topic)
            if not any(endpoint.node_name == RECORDER_NODE_NAME for endpoint in endpoints):
                return False
        return True

    def _start_recorder(self) -> None:
        free_bytes = shutil.disk_usage(self.output_path.parent).free
        if free_bytes < self.minimum_free_bytes:
            raise RuntimeError(
                f"disk_free_below_floor:{free_bytes}/{self.minimum_free_bytes}"
            )
        command = build_record_command(
            storage_id=self.storage_id,
            output_path=self.output_path,
            topics=RECORD_TOPICS,
            qos_override_path=self.qos_overrides_path,
        )
        self._recorder = subprocess.Popen(command, text=True, start_new_session=True)

    def _tick(self) -> None:
        try:
            self._tick_inner()
        except Exception as error:  # noqa: BLE001 - convert runtime faults to a rejected run
            self.get_logger().error(f"episode manager failure: {error}")
            self.abort(f"internal_error:{type(error).__name__}:{error}")

    def _tick_inner(self) -> None:
        now = time.monotonic()
        self.lifecycle.advance(now_monotonic_s=now)
        phase = self.lifecycle.phase

        if phase is EpisodePhase.PREFLIGHT:
            readiness = evaluate_readiness(
                READINESS_REQUIREMENTS,
                self._observations,
                now_monotonic_s=now,
            )
            reasons = list(readiness.reasons)
            if not self._verify_command_authority():
                reasons.append("command_authority_not_watchdog_only")
            if not self._verify_operator_authority():
                reasons.append("operator_authority_not_release_keyboard_only")
            if self._goal_rho_m is None or not math.isfinite(self._goal_rho_m):
                reasons.append("goal_range_invalid")
            elif self._goal_rho_m <= self._goal_reached.tolerance_m:
                reasons.append("goal_already_reached")
            self._last_readiness_reasons = tuple(reasons)
            if not reasons:
                self._start_recorder()
                self.lifecycle.begin_recorder(now_monotonic_s=now)
            elif now - self._started_monotonic_s >= self.preflight_timeout_s:
                self.abort("preflight_timeout:" + "|".join(reasons))

        elif phase is EpisodePhase.RECORDER_STARTING:
            if self._recorder is None:
                self.abort("recorder_not_started")
            elif self._recorder.poll() is not None:
                self._recorder_return_code = self._recorder.returncode
                self.abort(f"recorder_exited:{self._recorder_return_code}")
            elif self._recorder_subscriptions_ready():
                self.lifecycle.recorder_ready(now_monotonic_s=now)
            elif now - self.lifecycle.phase_started_s >= self.recorder_ready_timeout_s:
                self.abort("recorder_readiness_timeout")

        elif phase in (
            EpisodePhase.ZERO_WARMUP,
            EpisodePhase.RECORDING,
            EpisodePhase.ZERO_COOLDOWN,
        ):
            if self._recorder is None or self._recorder.poll() is not None:
                if self._recorder is not None:
                    self._recorder_return_code = self._recorder.returncode
                self.abort(f"recorder_exited:{self._recorder_return_code}")
            runtime_readiness = evaluate_readiness(
                READINESS_REQUIREMENTS,
                self._observations,
                now_monotonic_s=now,
            )
            if not runtime_readiness.ready:
                self.abort("stream_lost:" + "|".join(runtime_readiness.reasons))

        self._publish_gated_intent(now)
        self._publish_state(now)
        if self.lifecycle.phase in (EpisodePhase.COMPLETE, EpisodePhase.FAILED):
            self._done = True

    def _publish_gated_intent(self, now_monotonic_s: float) -> None:
        decision = gate_operator_intent(
            self.lifecycle,
            self._latest_operator_intent,
            now_monotonic_s=now_monotonic_s,
        )
        self._last_gate_reason = decision.reason
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.twist.linear.x = decision.command.linear_mps
        message.twist.angular.z = decision.command.angular_radps
        self._intent_publisher.publish(message)

    def _publish_state(self, now_monotonic_s: float) -> None:
        snapshot = self.lifecycle.snapshot(now_monotonic_s=now_monotonic_s)
        message = EpisodeState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.episode_id = self.episode_id
        message.phase = snapshot.phase.value
        message.reason = snapshot.reason
        message.motion_permitted = snapshot.motion_permitted
        message.recording_elapsed_s = snapshot.recording_elapsed_s
        message.recording_remaining_s = snapshot.recording_remaining_s
        message.sequence = self._sequence
        self._sequence += 1
        self._state_publisher.publish(message)

    def abort(self, reason: str) -> None:
        if self.lifecycle.phase in (EpisodePhase.COMPLETE, EpisodePhase.FAILED):
            self._done = True
            return
        now = time.monotonic()
        self.lifecycle.fail(reason, now_monotonic_s=now)
        self._publish_gated_intent(now)
        self._publish_state(now)
        self._done = True

    def _publish_final_zero_burst(self) -> None:
        self._latest_operator_intent = OperatorIntent(
            ZERO_COMMAND,
            time.monotonic(),
            True,
        )
        for _ in range(5):
            message = TwistStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self.frame_id
            self._intent_publisher.publish(message)
            time.sleep(0.05)

    def _stop_recorder(self) -> None:
        if self._recorder is None:
            return
        if self._recorder.poll() is None:
            os.killpg(self._recorder.pid, signal.SIGINT)
            try:
                self._recorder.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                os.killpg(self._recorder.pid, signal.SIGTERM)
                try:
                    self._recorder.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(self._recorder.pid, signal.SIGKILL)
                    self._recorder.wait(timeout=5.0)
        self._recorder_return_code = self._recorder.returncode

    def _write_result(self) -> None:
        snapshot = self.lifecycle.snapshot(now_monotonic_s=time.monotonic())
        document = {
            "schema_version": "1.0.0",
            "episode_id": self.episode_id,
            "environment_id": self.environment_id,
            "split": self.split,
            "route_id": self.route_id,
            "layout_id": self.layout_id,
            "acquisition_code_revision": self.code_revision,
            "episode_manager_completed": snapshot.phase is EpisodePhase.COMPLETE
            and self._recorder_return_code == 0,
            "requires_offline_validation": True,
            "phase": snapshot.phase.value,
            "reason": snapshot.reason,
            "last_gate_reason": self._last_gate_reason,
            "output_path": str(self.output_path),
            "started_wall_unix_s": self._started_wall_s,
            "manager_runtime_s": time.monotonic() - self._started_monotonic_s,
            "recording_elapsed_s": snapshot.recording_elapsed_s,
            "recorder_return_code": self._recorder_return_code,
            "readiness_reasons_at_last_preflight": list(self._last_readiness_reasons),
            "stream_message_counts": {
                topic: observation.message_count
                for topic, observation in sorted(self._observations.items())
            },
            "configuration": {
                "duration_s": self.lifecycle.config.duration_s,
                "zero_warmup_s": self.lifecycle.config.zero_warmup_s,
                "zero_cooldown_s": self.lifecycle.config.zero_cooldown_s,
                "operator_timeout_s": self.lifecycle.config.operator_timeout_s,
                "linear_mps": self.lifecycle.config.linear_mps,
                "angular_radps": self.lifecycle.config.angular_radps,
                "goal_tolerance_m": self._goal_reached.tolerance_m,
                "goal_required_samples": self._goal_reached.required_samples,
                "minimum_free_bytes": self.minimum_free_bytes,
                "storage_id": self.storage_id,
                "qos_overrides_path": str(self.qos_overrides_path),
                "record_topics": list(RECORD_TOPICS),
            },
        }
        with self.result_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def finalize(self) -> None:
        self._publish_final_zero_burst()
        self._stop_recorder()
        self._write_result()


def main(args: list[str] | None = None) -> None:
    signal_gate = FirstSignalGate()

    def request_shutdown(_signum: int, _frame: object) -> None:
        if signal_gate.accept():
            raise KeyboardInterrupt

    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ProtocolEpisodeManager()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    previous_sigint = signal.signal(signal.SIGINT, request_shutdown)
    previous_sigterm = signal.signal(signal.SIGTERM, request_shutdown)
    try:
        while rclpy.ok() and not node.done:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        node.abort("manager_signal")
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.finalize()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
