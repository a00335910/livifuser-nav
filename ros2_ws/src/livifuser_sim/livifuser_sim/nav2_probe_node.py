"""Isolated command relay and goal client for the bounded Nav2 probe."""

from __future__ import annotations

import json
import math
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist, TwistStamped
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node

from .world_layers import load_world


def bounded(value: float, limit: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(-limit, min(limit, value))


class Nav2ProbeNode(Node):
    """Send one map-frame goal and relay only bounded commands to Gazebo."""

    def __init__(self) -> None:
        super().__init__("livifuser_nav2_probe")
        self.declare_parameter("geometry_path", "")
        self.declare_parameter("status_path", "")
        self.declare_parameter("condition", "C0")
        self.declare_parameter("max_linear_mps", 0.08)
        self.declare_parameter("max_angular_radps", 0.40)
        geometry_path = Path(str(self.get_parameter("geometry_path").value))
        self._status_path = Path(str(self.get_parameter("status_path").value))
        self._condition = str(self.get_parameter("condition").value)
        self._max_linear = float(self.get_parameter("max_linear_mps").value)
        self._max_angular = float(self.get_parameter("max_angular_radps").value)
        if not geometry_path.is_file():
            raise ValueError("geometry_path must name a world JSON")
        if not self._status_path.parent.is_dir():
            raise ValueError("status_path parent must exist")
        if self._max_linear <= 0.0 or self._max_angular <= 0.0:
            raise ValueError("command bounds must be positive")
        world = load_world(geometry_path)
        if world.start_pose_xy_yaw is None or world.goal_xy_m is None:
            raise ValueError("world must declare a start pose and goal")
        self._world = world
        self._clamped_command_count = 0
        self._invalid_command_count = 0
        self._command_count = 0
        self._last_command = Twist()
        self._goal_sent = False
        self._goal_accepted = False
        self._result_status: int | None = None
        self._navigator_state: int | None = None
        self._state_future = None

        self._sim_publisher = self.create_publisher(
            Twist, "/livifuser/sim_cmd_vel", 10
        )
        self._audit_publisher = self.create_publisher(
            TwistStamped, "/livifuser/cmd_vel_stamped", 10
        )
        self.create_subscription(
            Twist, "/livifuser/nav2_cmd_vel", self._on_command, 10
        )
        self._action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self._state_client = self.create_client(GetState, "bt_navigator/get_state")
        self._goal_timer = self.create_timer(0.5, self._try_send_goal)
        self._write_status("waiting_for_nav2_active")

    def _publish(self, command: Twist) -> None:
        self._sim_publisher.publish(command)
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = "base_link"
        stamped.twist = command
        self._audit_publisher.publish(stamped)

    def _publish_zero(self) -> None:
        self._last_command = Twist()
        self._publish(self._last_command)

    def _on_command(self, message: Twist) -> None:
        self._command_count += 1
        incoming = (message.linear.x, message.angular.z)
        if not all(math.isfinite(value) for value in incoming):
            self._invalid_command_count += 1
        linear = bounded(incoming[0], self._max_linear)
        angular = bounded(incoming[1], self._max_angular)
        if linear != incoming[0] or angular != incoming[1]:
            self._clamped_command_count += 1
        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        self._last_command = command
        self._publish(command)

    def _try_send_goal(self) -> None:
        if self._goal_sent:
            return
        if not self._action_client.server_is_ready() or not self._state_client.service_is_ready():
            return
        if self._state_future is None:
            self._state_future = self._state_client.call_async(GetState.Request())
            return
        if not self._state_future.done():
            return
        response = self._state_future.result()
        self._state_future = None
        self._navigator_state = int(response.current_state.id)
        if self._navigator_state != State.PRIMARY_STATE_ACTIVE:
            self._write_status("waiting_for_nav2_active")
            return
        self._goal_sent = True
        self._goal_timer.cancel()
        start_x, start_y, _ = self._world.start_pose_xy_yaw
        goal_x, goal_y = self._world.goal_xy_m
        yaw = math.atan2(goal_y - start_y, goal_x - start_x)
        goal = NavigateToPose.Goal()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = "map"
        goal.pose.pose.position.x = goal_x
        goal.pose.pose.position.y = goal_y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)
        self._write_status("goal_sent")

    def _on_goal_response(self, future) -> None:
        goal_handle = future.result()
        self._goal_accepted = bool(goal_handle.accepted)
        if not goal_handle.accepted:
            self._publish_zero()
            self._write_status("goal_rejected")
            return
        self._write_status("goal_accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        self._result_status = int(future.result().status)
        self._publish_zero()
        label = {
            GoalStatus.STATUS_SUCCEEDED: "succeeded",
            GoalStatus.STATUS_ABORTED: "aborted",
            GoalStatus.STATUS_CANCELED: "canceled",
        }.get(self._result_status, "finished_unknown")
        self._write_status(label)

    def _write_status(self, phase: str) -> None:
        start = self._world.start_pose_xy_yaw
        payload = {
            "schema_version": 1,
            "phase": phase,
            "condition": self._condition,
            "world": self._world.name,
            "start_pose_xy_yaw": list(start) if start is not None else None,
            "goal_xy_m": list(self._world.goal_xy_m or ()),
            "goal_sent": self._goal_sent,
            "goal_accepted": self._goal_accepted,
            "result_status": self._result_status,
            "bt_navigator_state": self._navigator_state,
            "command_bounds": {
                "linear_mps": self._max_linear,
                "angular_radps": self._max_angular,
            },
            "clamped_command_count": self._clamped_command_count,
            "invalid_command_count": self._invalid_command_count,
            "command_count": self._command_count,
        }
        temporary = self._status_path.with_suffix(self._status_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self._status_path)

    def destroy_node(self) -> bool:
        self._publish_zero()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Nav2ProbeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
