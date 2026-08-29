"""ROS wrapper for the exact frozen four-variant live policy runtime."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMessage
from livifuser_interfaces.msg import PolicyProposal, RelativeGoal
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Empty
from tf2_msgs.msg import TFMessage

from livifuser_nav.live_association import LiveAssociator
from livifuser_nav.live_runtime import (
    LiveObservation,
    construct_exact_runtime,
    extract_verified_backbone,
    load_policy_material,
)

PROPOSAL_TOPIC = "/livifuser/eval/policy_proposal"
RESET_TOPIC = "/livifuser/eval/runtime_reset"


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def time_message(value_ns: int) -> TimeMessage:
    message = TimeMessage()
    message.sec, message.nanosec = divmod(int(value_ns), 1_000_000_000)
    return message


class LivePolicyRunnerNode(Node):
    """Accept exactly one immutable identity and publish proposals, never commands."""

    def __init__(self) -> None:
        super().__init__("livifuser_live_policy_runner")
        for name, default in (
            ("variant", ""),
            ("seed", 0),
            ("device", "cuda:0"),
            ("backbone_bundle", ""),
            ("policy_bundle", ""),
            ("backbone_extract_root", ""),
            ("sensor_contract", ""),
        ):
            self.declare_parameter(name, default)
        variant = str(self.get_parameter("variant").value)
        seed = int(self.get_parameter("seed").value)
        device = str(self.get_parameter("device").value)
        backbone_bundle = Path(str(self.get_parameter("backbone_bundle").value))
        policy_bundle = Path(str(self.get_parameter("policy_bundle").value))
        extract_root = Path(str(self.get_parameter("backbone_extract_root").value))
        sensor_contract_path = Path(str(self.get_parameter("sensor_contract").value))
        for label, path in (
            ("backbone_bundle", backbone_bundle),
            ("policy_bundle", policy_bundle),
            ("sensor_contract", sensor_contract_path),
        ):
            if not path.is_file():
                raise ValueError(f"{label} is not a file: {path}")
        if not str(extract_root):
            raise ValueError("backbone_extract_root is required")

        # Complete nested verification precedes model construction and every
        # subscription. A process can therefore expose only one verified ID.
        material = load_policy_material(policy_bundle, variant, seed)
        snapshot = extract_verified_backbone(backbone_bundle, extract_root)
        self._runtime = construct_exact_runtime(
            backbone_snapshot=snapshot,
            policy_material=material,
            sensor_contract_path=sensor_contract_path,
            device=device,
        )
        self._warmup_loaded_models()
        self._variant = variant
        self._seed = seed
        self._contract = self._runtime.sensor_contract
        self._associator = LiveAssociator()
        self._context_sequence = 0
        self._camera_info_verified = False
        self._tf_verified: set[str] = set()
        self._logged_calibration_wait = False
        self._control_timer = None
        self._last_tick_ns: int | None = None
        self._streams_seen: set[str] = set()

        self._proposal_publisher = self.create_publisher(PolicyProposal, PROPOSAL_TOPIC, 10)
        self.create_subscription(Image, "/camera/image_raw", self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self._on_odometry, qos_profile_sensor_data)
        self.create_subscription(
            RelativeGoal, "/livifuser/goal_relative", self._on_goal, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, "/camera/camera_info", self._on_camera_info, qos_profile_sensor_data
        )
        # Three independent static_transform_publisher nodes each latch one
        # TransformStamped on /tf_static. TRANSIENT_LOCAL with depth 1 keeps a
        # single sample, so two of the three required frames are dropped and
        # calibration never completes. Depth must cover every publisher.
        static_qos = QoSProfile(depth=10)
        static_qos.reliability = ReliabilityPolicy.RELIABLE
        static_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(TFMessage, "/tf_static", self._on_tf_static, static_qos)
        self.create_subscription(Empty, RESET_TOPIC, self._on_external_reset, 10)
        # Do not start the 10 Hz sim-time loop until camera_info and all three
        # latched static transforms have been accepted. A tick that runs in
        # the same executor slice as subscription creation publishes
        # startup_calibration_not_verified forever (a7) or leaves a >250 ms
        # hole that trips the supervisor watchdog (a9).
        self.create_timer(
            0.05,
            self._on_wait_for_calibration,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        self.get_logger().info(
            f"verified exact live policy identity {variant}/seed_{seed} on {device}"
        )

    def _warmup_loaded_models(self) -> None:
        rgb = np.zeros((3, 224, 224), dtype=np.float32)
        pixels = self._runtime.torch.from_numpy(rgb).unsqueeze(0).to(self._runtime.device)
        self._runtime._sync()
        with self._runtime.torch.inference_mode():
            self._runtime.backbone(pixel_values=pixels, return_dict=False)
        self._runtime._sync()

    def _claim_stamp(self, stamp_ns: int) -> bool:
        if stamp_ns <= 0:
            return False
        if self._last_tick_ns is not None and stamp_ns <= self._last_tick_ns:
            return False
        self._last_tick_ns = stamp_ns
        return True

    def _on_wait_for_calibration(self) -> None:
        if self._control_timer is not None:
            return
        if self.get_clock().now().nanoseconds <= 0:
            return
        if not self._calibration_ready():
            return
        self.get_logger().info(
            "calibration verified; sim clock "
            f"{self.get_clock().now().nanoseconds}; starting 10 Hz control loop"
        )
        self._control_timer = self.create_timer(
            0.1, self._on_control_tick, clock=self.get_clock()
        )

    def _reset(self, reason: str, *, publish: bool = True) -> None:
        self._associator.reset()
        self._runtime.clear_history()
        if publish:
            self._publish_invalid(reason, self.get_clock().now().nanoseconds)

    def _push(self, stream: str, header_stamp_ns: int, payload) -> None:
        result = self._associator.push(
            stream,
            stamp_ns=header_stamp_ns,
            arrival_monotonic_ns=time.monotonic_ns(),
            payload=payload,
        )
        if result.accepted:
            self._streams_seen.add(stream)
            return
        # Sensor-stream integrity failures reset association state. They must
        # not publish a proposal: those headers use whatever /clock is now and
        # interleave with the 10 Hz control stamps, which the supervisor
        # treats as proposal-stamp regression.
        self._runtime.clear_history()

    def _on_rgb(self, message: Image) -> None:
        contract = self._contract["image_contract"]
        if (
            message.width != contract["width"]
            or message.height != contract["height"]
            or message.encoding != contract["encoding"]
            or message.header.frame_id != contract["frame_id"]
            or message.step < message.width * 3
            or len(message.data) != message.step * message.height
        ):
            self._reset("rgb_contract_invalid")
            return
        raw = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        rgb = np.ascontiguousarray(raw[:, : message.width * 3].reshape(240, 320, 3))
        self._push("rgb", stamp_ns(message.header.stamp), rgb)

    def _on_scan(self, message: LaserScan) -> None:
        contract = self._contract["scan_contract"]
        ranges = np.asarray(message.ranges, dtype=np.float32)
        valid = (
            message.header.frame_id == contract["frame_id"]
            and ranges.ndim == 1
            and ranges.size >= contract["minimum_beam_count"]
            and math.isclose(message.angle_min, contract["angle_min_rad"], abs_tol=1e-6)
            and math.isclose(message.angle_max, contract["angle_max_rad"], abs_tol=1e-5)
            and math.isclose(message.range_min, contract["range_min_m"], abs_tol=1e-6)
            and math.isclose(message.range_max, contract["range_max_m"], abs_tol=1e-6)
            and math.isfinite(message.angle_increment)
            and message.angle_increment > 0.0
            and not np.any(np.isinf(ranges))
        )
        # NaNs are the frozen missing-return representation and are tokenized
        # as invalid beams; infinities or malformed geometry are integrity loss.
        if not valid:
            self._reset("scan_contract_invalid")
            return
        self._push(
            "scan",
            stamp_ns(message.header.stamp),
            (ranges, int(ranges.size), float(message.angle_increment)),
        )

    def _on_odometry(self, message: Odometry) -> None:
        state = np.asarray(
            [message.twist.twist.linear.x, message.twist.twist.angular.z], dtype=np.float32
        )
        if (
            message.header.frame_id != "odom"
            or message.child_frame_id != "base_link"
            or not np.all(np.isfinite(state))
        ):
            self._reset("odometry_contract_invalid")
            return
        self._push("odometry", stamp_ns(message.header.stamp), state)

    def _on_goal(self, message: RelativeGoal) -> None:
        goal = np.asarray([message.rho_m, message.sin_alpha, message.cos_alpha], dtype=np.float32)
        unit = float(goal[1] ** 2 + goal[2] ** 2)
        if (
            message.header.frame_id != "base_link"
            or not np.all(np.isfinite(goal))
            or not math.isclose(unit, 1.0, abs_tol=1e-3)
        ):
            self._reset("goal_contract_invalid")
            return
        self._push("goal", stamp_ns(message.header.stamp), goal)

    def _on_camera_info(self, message: CameraInfo) -> None:
        expected = self._contract["calibration"]["recorded_camera_info"]
        tolerances = self._contract["camera_info_contract"]
        self._camera_info_verified = bool(
            message.width == expected["width"]
            and message.height == expected["height"]
            and message.distortion_model == expected["distortion_model"]
            and message.header.frame_id == expected["frame_id"]
            and np.allclose(
                message.k, expected["k"], rtol=0.0, atol=tolerances["k_absolute_tolerance"]
            )
            and np.allclose(
                message.d, expected["d"], rtol=0.0, atol=tolerances["d_absolute_tolerance"]
            )
        )
        if self._camera_info_verified:
            return
        self.get_logger().warn(
            "camera_info mismatch: "
            f"size={message.width}x{message.height} model={message.distortion_model!r} "
            f"frame={message.header.frame_id!r}"
        )
        self._reset("camera_info_identity_invalid")

    def _on_tf_static(self, message: TFMessage) -> None:
        expected = self._contract["calibration"]["static_transforms"]
        for transform in message.transforms:
            key = f"{transform.header.frame_id}->{transform.child_frame_id}"
            if key not in expected:
                continue
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            observed_t = np.asarray([translation.x, translation.y, translation.z])
            observed_q = np.asarray([rotation.x, rotation.y, rotation.z, rotation.w])
            expected_t = np.asarray(expected[key]["translation"])
            expected_q = np.asarray(expected[key]["quaternion_xyzw"])
            quaternion_match = np.allclose(
                observed_q, expected_q, rtol=0.0, atol=1e-7
            ) or np.allclose(observed_q, -expected_q, rtol=0.0, atol=1e-7)
            if np.allclose(observed_t, expected_t, rtol=0.0, atol=1e-7) and quaternion_match:
                if key not in self._tf_verified:
                    self.get_logger().info(f"accepted static transform {key}")
                self._tf_verified.add(key)
            else:
                self._reset(f"static_transform_identity_invalid:{key}")

    def _on_external_reset(self, _message: Empty) -> None:
        self._reset("supervisor_runtime_reset", publish=False)

    def _calibration_ready(self) -> bool:
        required = {
            "base_link->base_scan",
            "base_scan->camera",
            "camera->camera_optical_frame",
        }
        ready = self._camera_info_verified and required.issubset(self._tf_verified)
        if ready:
            return True
        if not self._logged_calibration_wait:
            self.get_logger().warn(
                "calibration not verified: camera_info="
                f"{self._camera_info_verified} missing_static_tf="
                f"{sorted(required - self._tf_verified)}"
            )
            self._logged_calibration_wait = True
        return False

    def _on_control_tick(self) -> None:
        tick_ns = self.get_clock().now().nanoseconds
        # Sim-time timers catch up in bursts. A second callback at the same
        # stamp is a no-op: publishing it trips proposal-stamp regression, and
        # feeding it to select() wipes the K=8 history as a duplicate tick.
        if tick_ns <= 0 or (
            self._last_tick_ns is not None and tick_ns <= self._last_tick_ns
        ):
            return
        if not self._calibration_ready():
            self._reset("startup_calibration_not_verified")
            return
        selected = self._associator.select(tick_ns)
        if not selected.accepted or selected.context is None:
            if selected.reason not in {
                "clock_regression_or_duplicate_tick",
                "rgb_missing",
            }:
                self._runtime.clear_history()
            self._publish_invalid(selected.reason, tick_ns)
            return
        if not self._claim_stamp(tick_ns):
            return
        context = selected.context
        ranges, beam_count, increment = context.scan.payload
        observation = LiveObservation(
            rgb=context.rgb.payload,
            scan_ranges=ranges,
            scan_beam_count=beam_count,
            scan_angle_increment_rad=increment,
            goal=context.goal.payload,
            robot_state=context.odometry.payload,
        )
        try:
            decision = self._runtime.accept(observation)
        except Exception as exc:
            self.get_logger().error(f"live inference integrity failure: {exc}")
            self._reset(f"inference_integrity_failure:{type(exc).__name__}")
            return
        self._context_sequence += 1
        proposal = self._base_proposal(decision.status, tick_ns)
        proposal.valid = True
        proposal.inference_ready = decision.ready
        proposal.context_sequence = self._context_sequence
        proposal.rgb_stamp_ns = context.rgb.stamp_ns
        proposal.scan_stamp_ns = context.scan.stamp_ns
        proposal.odometry_stamp_ns = context.odometry.stamp_ns
        proposal.goal_stamp_ns = context.goal.stamp_ns
        proposal.mean_h8 = decision.mean_h8.reshape(-1).tolist()
        proposal.log_variance_h8 = decision.log_variance_h8.reshape(-1).tolist()
        proposal.proposed_linear_x = float(decision.proposed_action[0])
        proposal.proposed_angular_z = float(decision.proposed_action[1])
        for name in ("aleatoric", "mahalanobis", "z_aleatoric", "z_mahalanobis", "combined"):
            setattr(proposal, name, float(getattr(decision, name)))
        thresholds = self._runtime.material.thresholds
        proposal.aleatoric_threshold = thresholds["aleatoric"]
        proposal.mahalanobis_threshold = thresholds["mahalanobis"]
        proposal.combined_threshold = thresholds["combined"]
        proposal.aleatoric_flag = decision.aleatoric_flag
        proposal.mahalanobis_flag = decision.mahalanobis_flag
        proposal.combined_intervention = decision.combined_intervention
        for name in (
            "rgb_preprocess",
            "splus_forward_and_pool",
            "lidar_tokenize",
            "policy_stack_and_forward",
            "uncertainty",
            "complete_path",
        ):
            setattr(proposal, f"{name}_ms", decision.stage_ms.get(name, 0.0))
        if self._context_sequence in (1, 8) or self._context_sequence % 200 == 0:
            self.get_logger().info(
                f"tick seq={self._context_sequence} ready={decision.ready} "
                f"status={decision.status} hist={len(self._runtime.history)} "
                f"stamp_ns={tick_ns}"
            )
        self._proposal_publisher.publish(proposal)

    def _base_proposal(self, status: str, header_stamp_ns: int) -> PolicyProposal:
        proposal = PolicyProposal()
        proposal.header.stamp = time_message(header_stamp_ns)
        proposal.header.frame_id = "base_link"
        proposal.variant = self._variant
        proposal.seed = self._seed
        proposal.status = status
        proposal.published_monotonic_ns = time.monotonic_ns()
        return proposal

    def _publish_invalid(self, reason: str, header_stamp_ns: int) -> None:
        if not hasattr(self, "_proposal_publisher"):
            return
        if not self._claim_stamp(header_stamp_ns):
            return
        proposal = self._base_proposal(reason, header_stamp_ns)
        proposal.context_sequence = self._context_sequence
        proposal.valid = False
        proposal.inference_ready = False
        self._proposal_publisher.publish(proposal)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LivePolicyRunnerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
