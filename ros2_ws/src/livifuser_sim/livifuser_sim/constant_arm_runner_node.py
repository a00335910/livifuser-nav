"""ROS wrapper for the open-loop constant-training-mean reference arm.

Closed-loop execution amendment section 1.1. This is deliberately a separate
node from `live_policy_runner_node`, not a branch inside it. The arm's entire
evidential value rests on being sensor-blind, and a node that never creates a
sensor subscription cannot lose that property to a later edit.

It publishes the same `PolicyProposal` on the same topic as the learned runner,
so the evaluation supervisor, its priority order, its limits, its termination
rules, and the recorded evidence are all unchanged.
"""

from __future__ import annotations

import math
import time

import rclpy
from builtin_interfaces.msg import Time as TimeMessage
from livifuser_interfaces.msg import PolicyProposal
from rclpy.node import Node
from std_msgs.msg import Empty

from livifuser_nav.live_runtime import (
    CONSTANT_ARM_NAME,
    ConstantActionRuntime,
)

PROPOSAL_TOPIC = "/livifuser/eval/policy_proposal"
RESET_TOPIC = "/livifuser/eval/runtime_reset"

# The arm produces no horizon prediction and no uncertainty score. Every field
# that would carry one is filled with NaN rather than zero: a zero is
# indistinguishable from a real score that did not exceed its threshold, and
# the amendment forbids recording an inapplicable gate as a gate that did not
# fire.
NOT_APPLICABLE = float("nan")


def time_message(value_ns: int) -> TimeMessage:
    message = TimeMessage()
    message.sec = int(value_ns // 1_000_000_000)
    message.nanosec = int(value_ns % 1_000_000_000)
    return message


class ConstantArmRunnerNode(Node):
    """Publish the preregistered constant proposal; never publish a command."""

    def __init__(self) -> None:
        super().__init__("livifuser_constant_arm_runner")
        self._runtime = ConstantActionRuntime()
        self._context_sequence = 0
        self._proposal_publisher = self.create_publisher(PolicyProposal, PROPOSAL_TOPIC, 10)
        self.create_subscription(Empty, RESET_TOPIC, self._on_external_reset, 10)
        self.create_timer(0.1, self._on_control_tick, clock=self.get_clock())
        self.get_logger().info(
            f"constant reference arm active: {CONSTANT_ARM_NAME}; "
            "no camera, scan, odometry, goal, backbone, or checkpoint is used"
        )

    def _on_external_reset(self, _message: Empty) -> None:
        self._runtime.clear_history()

    def _on_control_tick(self) -> None:
        tick_ns = self.get_clock().now().nanoseconds
        decision = self._runtime.accept()
        self._context_sequence += 1

        proposal = PolicyProposal()
        proposal.header.stamp = time_message(tick_ns)
        proposal.header.frame_id = "base_link"
        proposal.variant = CONSTANT_ARM_NAME
        proposal.seed = 0
        proposal.status = decision.status
        proposal.published_monotonic_ns = time.monotonic_ns()
        proposal.context_sequence = self._context_sequence
        proposal.valid = True
        proposal.inference_ready = decision.ready

        # No sensor is read, so there is no associated input stamp to report.
        proposal.rgb_stamp_ns = 0
        proposal.scan_stamp_ns = 0
        proposal.odometry_stamp_ns = 0
        proposal.goal_stamp_ns = 0

        proposal.mean_h8 = [NOT_APPLICABLE] * 16
        proposal.log_variance_h8 = [NOT_APPLICABLE] * 16
        # The proposal is the exact float64 constant; this frozen message
        # transports it in float32 exactly as it transports learned proposals.
        proposal.proposed_linear_x = float(decision.proposed_action[0])
        proposal.proposed_angular_z = float(decision.proposed_action[1])

        for name in (
            "aleatoric",
            "mahalanobis",
            "z_aleatoric",
            "z_mahalanobis",
            "combined",
            "aleatoric_threshold",
            "mahalanobis_threshold",
            "combined_threshold",
        ):
            setattr(proposal, name, NOT_APPLICABLE)
        proposal.aleatoric_flag = False
        proposal.mahalanobis_flag = False
        proposal.combined_intervention = False

        for name in (
            "rgb_preprocess",
            "splus_forward_and_pool",
            "lidar_tokenize",
            "policy_stack_and_forward",
            "uncertainty",
        ):
            setattr(proposal, f"{name}_ms", NOT_APPLICABLE)
        proposal.complete_path_ms = (time.monotonic_ns() - proposal.published_monotonic_ns) / 1e6

        self._proposal_publisher.publish(proposal)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ConstantArmRunnerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


assert math.isnan(NOT_APPLICABLE), "the not-applicable sentinel must never be a real value"


if __name__ == "__main__":
    main()
