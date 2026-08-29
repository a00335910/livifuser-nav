"""Independent simulation-only owner of the Gazebo velocity command."""

from __future__ import annotations

import math
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from livifuser_interfaces.msg import PolicyProposal, SimulationSupervisorStatus
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Empty

from livifuser_nav.live_runtime import CONSTANT_ARM_NAME
from livifuser_nav.simulation_supervision import (
    PrivilegedState,
    ProposalInput,
    SimulationSupervisor,
)

from .world_layers import LAYER_EXPERT, load_world, point_clearance

PROPOSAL_TOPIC = "/livifuser/eval/policy_proposal"
STATUS_TOPIC = "/livifuser/eval/supervisor_status"
RESET_TOPIC = "/livifuser/eval/runtime_reset"
SIM_COMMAND_TOPIC = "/livifuser/sim_cmd_vel"
RECORDED_COMMAND_TOPIC = "/livifuser/cmd_vel_stamped"
GROUND_TRUTH_TOPIC = "/livifuser/sim/ground_truth/odom"


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class SimulationSupervisorNode(Node):
    """The only process allowed to publish the isolated simulator command."""

    def __init__(self) -> None:
        super().__init__("livifuser_simulation_supervisor")
        self.declare_parameter("world_json", "")
        self.declare_parameter("scientific_deadline_sec", 120.0)
        self.declare_parameter("expected_variant", "")
        self.declare_parameter("expected_seed", 0)
        # A hung-policy detector, expressed in wall time. The default suits a
        # simulator at real-time factor 1.0; running below real time stretches
        # the wall gap between simulated ticks and the launch file scales this
        # accordingly. It never gates motion, and it never shortens the frozen
        # simulated scientific deadline.
        self.declare_parameter("stale_proposal_wall_timeout_ms", 250.0)
        world_path = Path(str(self.get_parameter("world_json").value))
        if not world_path.is_file():
            raise ValueError(f"world_json is not a file: {world_path}")
        self._world = load_world(world_path)
        if self._world.goal_xy_m is None:
            raise ValueError("world has no goal")
        self._variant = str(self.get_parameter("expected_variant").value)
        self._seed = int(self.get_parameter("expected_seed").value)
        # The constant reference arm has no checkpoint and therefore no training
        # seed; it is pinned to the reserved identity seed 0. Every learned
        # identity must still carry a real non-zero seed, so the guard below is
        # narrowed rather than relaxed.
        if self._variant == CONSTANT_ARM_NAME:
            if self._seed != 0:
                raise ValueError("the constant reference arm must use reserved seed 0")
        elif not self._variant or not self._seed:
            raise ValueError("expected_variant and expected_seed are required")
        self._supervisor = SimulationSupervisor(
            scientific_deadline_sec=float(
                self.get_parameter("scientific_deadline_sec").value
            )
        )
        self._ground_truth: Odometry | None = None
        self._last_proposal_receipt_ns: int | None = None
        self._last_context_sequence = 0
        self._emergency_stop = False
        # 250 ms is a hung-policy detector, not a real-time-factor floor.
        # Arm it only after the first inference-ready proposal so startup,
        # K=8 warmup, and a slow first ogre1 frame cannot look like a stall.
        self._stale_watchdog_armed = False
        self._stale_timeout_ns = int(
            float(self.get_parameter("stale_proposal_wall_timeout_ms").value) * 1e6
        )

        self._command_publisher = self.create_publisher(Twist, SIM_COMMAND_TOPIC, 10)
        self._recorded_publisher = self.create_publisher(
            TwistStamped, RECORDED_COMMAND_TOPIC, 10
        )
        self._status_publisher = self.create_publisher(
            SimulationSupervisorStatus, STATUS_TOPIC, 10
        )
        self._reset_publisher = self.create_publisher(Empty, RESET_TOPIC, 10)
        self.create_subscription(PolicyProposal, PROPOSAL_TOPIC, self._on_proposal, 10)
        self.create_subscription(
            Odometry, GROUND_TRUTH_TOPIC, self._on_ground_truth, qos_profile_sensor_data
        )
        self.create_subscription(
            Bool, "/livifuser/eval/emergency_stop", self._on_emergency_stop, 10
        )
        self.create_timer(
            0.05, self._on_wall_watchdog, clock=Clock(clock_type=ClockType.STEADY_TIME)
        )

    def _on_ground_truth(self, message: Odometry) -> None:
        pose = message.pose.pose
        values = (
            pose.position.x,
            pose.position.y,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if not all(math.isfinite(float(value)) for value in values):
            self._terminal_stop("operational_failure_ground_truth_invalid")
            return
        self._ground_truth = message

    def _on_emergency_stop(self, message: Bool) -> None:
        self._emergency_stop = bool(message.data)
        if self._emergency_stop:
            self._terminal_stop("emergency_operator_stop")

    def _privileged_state(self) -> PrivilegedState:
        if self._ground_truth is None:
            return PrivilegedState(False, False, math.inf, math.nan)
        position = self._ground_truth.pose.pose.position
        clearance = point_clearance(
            self._world.layer(LAYER_EXPERT), float(position.x), float(position.y)
        )
        goal_distance = math.hypot(
            self._world.goal_xy_m[0] - float(position.x),
            self._world.goal_xy_m[1] - float(position.y),
        )
        return PrivilegedState(
            available=True,
            collision=clearance < 0.105,
            goal_distance_m=goal_distance,
            clearance_m=clearance,
        )

    def _on_proposal(self, message: PolicyProposal) -> None:
        receipt_ns = time.monotonic_ns()
        if message.variant != self._variant or int(message.seed) != self._seed:
            self._terminal_stop("operational_failure_policy_identity")
            return
        if (
            message.valid
            and message.inference_ready
            and int(message.context_sequence) <= self._last_context_sequence
        ):
            self._terminal_stop("operational_failure_context_sequence")
            return
        self._last_context_sequence = max(
            self._last_context_sequence, int(message.context_sequence)
        )
        self._last_proposal_receipt_ns = receipt_ns
        if message.inference_ready:
            self._stale_watchdog_armed = True
        privileged = self._privileged_state()
        decision = self._supervisor.step(
            ProposalInput(
                stamp_ns=stamp_ns(message.header.stamp),
                linear_x=float(message.proposed_linear_x),
                angular_z=float(message.proposed_angular_z),
                valid=bool(message.valid),
                inference_ready=bool(message.inference_ready),
                status=str(message.status),
                combined_intervention=bool(message.combined_intervention),
            ),
            privileged,
            emergency_stop=self._emergency_stop,
        )
        self._publish_command(decision.executed, message.header.stamp)
        status = SimulationSupervisorStatus()
        status.header = message.header
        status.variant = message.variant
        status.seed = message.seed
        status.context_sequence = message.context_sequence
        status.proposal_receipt_monotonic_ns = receipt_ns
        status.raw_linear_x, status.raw_angular_z = decision.raw
        status.clipped_linear_x, status.clipped_angular_z = decision.clipped
        status.executed_linear_x, status.executed_angular_z = decision.executed
        status.proposal_valid = message.valid
        status.inference_ready = message.inference_ready
        status.stale_timeout = decision.reason == "watchdog_timeout"
        status.collision = privileged.collision
        status.uncertainty_intervention = message.combined_intervention
        status.success = decision.terminal_reason == "success"
        status.terminal = bool(decision.terminal_reason)
        status.priority_reason = decision.reason
        status.terminal_reason = decision.terminal_reason
        status.ground_truth_clearance_m = privileged.clearance_m
        status.ground_truth_goal_distance_m = privileged.goal_distance_m
        status.consecutive_success_samples = decision.success_samples
        if message.published_monotonic_ns and receipt_ns >= message.published_monotonic_ns:
            status.proposal_age_ms = (receipt_ns - message.published_monotonic_ns) / 1e6
        else:
            status.proposal_age_ms = math.nan
        status.control_interval_ms = decision.control_interval_ms
        status.stretched_interval_count = self._supervisor.stretched_interval_count
        self._status_publisher.publish(status)

    def _on_wall_watchdog(self) -> None:
        if (
            not self._stale_watchdog_armed
            or self._supervisor.terminal_reason
            or self._last_proposal_receipt_ns is None
        ):
            return
        if time.monotonic_ns() - self._last_proposal_receipt_ns > self._stale_timeout_ns:
            self._reset_publisher.publish(Empty())
            self._terminal_stop("watchdog_timeout")

    def _terminal_stop(self, reason: str) -> None:
        self._supervisor.force_terminal(reason)
        stamp = self.get_clock().now().to_msg()
        self._publish_command((0.0, 0.0), stamp)
        status = SimulationSupervisorStatus()
        status.header.stamp = stamp
        status.header.frame_id = "base_link"
        status.variant = self._variant
        status.seed = self._seed
        status.context_sequence = self._last_context_sequence
        status.proposal_receipt_monotonic_ns = time.monotonic_ns()
        status.terminal = True
        status.stale_timeout = reason == "watchdog_timeout"
        status.priority_reason = reason
        status.terminal_reason = self._supervisor.terminal_reason
        status.ground_truth_clearance_m = math.nan
        status.ground_truth_goal_distance_m = math.nan
        status.proposal_age_ms = math.nan
        self._status_publisher.publish(status)

    def _publish_command(self, values: tuple[float, float], stamp) -> None:
        command = Twist()
        command.linear.x = float(values[0])
        command.angular.z = float(values[1])
        self._command_publisher.publish(command)
        recorded = TwistStamped()
        recorded.header.stamp = stamp
        recorded.header.frame_id = "base_link"
        recorded.twist = command
        self._recorded_publisher.publish(recorded)

    def destroy_node(self):
        if hasattr(self, "_command_publisher"):
            self._publish_command((0.0, 0.0), self.get_clock().now().to_msg())
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SimulationSupervisorNode()
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
