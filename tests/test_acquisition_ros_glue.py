"""ROS glue checks for acquisition nodes that never publish `/cmd_vel`."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMAND_PACKAGE = REPO_ROOT / "ros2_ws" / "src" / "livifuser_command_watchdog"
GOAL_PACKAGE = REPO_ROOT / "ros2_ws" / "src" / "livifuser_goal_publisher"
sys.path.insert(0, str(COMMAND_PACKAGE))
sys.path.insert(0, str(GOAL_PACKAGE))

ISOLATION_ENVIRONMENT = {
    "ROS_LOCALHOST_ONLY": "1",
    "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
}
os.environ.update(ISOLATION_ENVIRONMENT)
ISOLATED_DOMAIN_IDS = ("84", "85", "86", "87")

try:
    import rclpy
    from geometry_msgs.msg import TwistStamped
    from livifuser_command_watchdog.episode_node import ProtocolEpisodeManager
    from livifuser_command_watchdog.release_keyboard_node import ReleaseKeyboardPublisher
    from livifuser_goal_publisher.odom_node import OdomWaypointGoalPublisher
    from livifuser_interfaces.msg import EpisodeState, RelativeGoal
    from nav_msgs.msg import Odometry
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import ReliabilityPolicy
    from std_msgs.msg import Empty

    ROS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on Windows
    Node = object  # type: ignore[assignment, misc]
    ROS_AVAILABLE = False


class AcquisitionListener(Node):
    def __init__(self) -> None:
        super().__init__("acquisition_glue_test_listener")
        self.gated_intents: list[TwistStamped] = []
        self.goals: list[RelativeGoal] = []
        self.create_subscription(
            TwistStamped,
            "/livifuser/teleop_intent_stamped",
            self.gated_intents.append,
            10,
        )
        self.create_subscription(
            RelativeGoal,
            "/livifuser/goal_relative",
            self.goals.append,
            10,
        )


@unittest.skipUnless(ROS_AVAILABLE, "ROS acquisition interfaces unavailable")
class AcquisitionRosGlueTests(unittest.TestCase):
    _domain_index = 0

    def setUp(self) -> None:
        os.environ.update(ISOLATION_ENVIRONMENT)
        index = AcquisitionRosGlueTests._domain_index
        AcquisitionRosGlueTests._domain_index += 1
        self.assertLess(index, len(ISOLATED_DOMAIN_IDS))
        os.environ["ROS_DOMAIN_ID"] = ISOLATED_DOMAIN_IDS[index]
        self.assertEqual(os.environ["ROS_LOCALHOST_ONLY"], "1")
        self.temporary = tempfile.TemporaryDirectory()
        self.nodes: list[Node] = []

    def tearDown(self) -> None:
        for node in reversed(self.nodes):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self.temporary.cleanup()

    def _spin(self, nodes: list[Node], duration_s: float = 0.5) -> None:
        executor = SingleThreadedExecutor()
        for node in nodes:
            executor.add_node(node)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        for node in nodes:
            executor.remove_node(node)
        executor.shutdown()

    def test_episode_preflight_cannot_forward_nonzero_intent(self) -> None:
        episode_id = "development_room_a_001"
        output = Path(self.temporary.name) / episode_id
        rclpy.init(
            args=[
                "--ros-args",
                "-p",
                f"episode_id:={episode_id}",
                "-p",
                f"output_path:={output}",
                "-p",
                "environment_id:=room_a",
                "-p",
                "split:=development",
                "-p",
                "route_id:=route_3m",
                "-p",
                "layout_id:=no_obstacle",
                "-p",
                "code_revision:=e281fbf",
            ]
        )
        manager = ProtocolEpisodeManager()
        listener = AcquisitionListener()
        self.nodes.extend((manager, listener))

        sensor_topics = {"/camera/image_raw", "/camera/camera_info", "/scan", "/odom"}
        sensor_subscriptions = {
            subscription.topic_name: subscription
            for subscription in manager.subscriptions
            if subscription.topic_name in sensor_topics
        }
        self.assertEqual(set(sensor_subscriptions), sensor_topics)
        self.assertTrue(
            all(
                subscription.qos_profile.reliability
                is ReliabilityPolicy.BEST_EFFORT
                for subscription in sensor_subscriptions.values()
            )
        )

        raw = TwistStamped()
        raw.header.stamp.sec = 1
        raw.header.frame_id = "base_link"
        raw.twist.linear.x = 0.08
        manager._on_operator_intent(raw)
        self._spin([manager, listener])

        self.assertGreater(len(listener.gated_intents), 2)
        self.assertTrue(
            all(message.twist.linear.x == 0.0 for message in listener.gated_intents)
        )
        manager.abort("test_complete")
        manager.finalize()
        result = json.loads(manager.result_path.read_text(encoding="utf-8"))
        self.assertFalse(result["episode_manager_completed"])
        self.assertTrue(result["requires_offline_validation"])
        self.assertEqual(result["phase"], "failed")

    def test_release_gui_requires_fresh_matching_episode_permission(self) -> None:
        rclpy.init(
            args=[
                "--ros-args",
                "-p",
                "episode_id:=development_room_a_002",
            ]
        )
        publisher = ReleaseKeyboardPublisher()
        self.nodes.append(publisher)
        publisher.key_down("i")
        self.assertEqual(publisher._current_command().linear_mps, 0.0)

        state = EpisodeState()
        state.episode_id = "development_room_a_002"
        state.motion_permitted = True
        publisher._on_state(state)
        publisher.key_down("i")
        self.assertEqual(publisher._current_command().linear_mps, 0.08)

        state.episode_id = "different_episode"
        publisher._on_state(state)
        self.assertEqual(publisher._current_command().linear_mps, 0.0)

    def test_operator_stop_before_recording_cancels_and_writes_zero_sidecar(self) -> None:
        episode_id = "development_operator_cancel_001"
        output = Path(self.temporary.name) / episode_id
        rclpy.init(
            args=[
                "--ros-args",
                "-p",
                f"episode_id:={episode_id}",
                "-p",
                f"output_path:={output}",
                "-p",
                "environment_id:=room_a",
                "-p",
                "split:=development",
                "-p",
                "route_id:=route_3m",
                "-p",
                "layout_id:=box_left",
                "-p",
                "code_revision:=e281fbf",
            ]
        )
        manager = ProtocolEpisodeManager()
        self.nodes.append(manager)
        self.assertIn(
            "/livifuser/operator_stop",
            {subscription.topic_name for subscription in manager.subscriptions},
        )
        manager._on_operator_stop(Empty())
        self.assertTrue(manager.done)
        manager.finalize()
        result = json.loads(manager.result_path.read_text(encoding="utf-8"))
        self.assertEqual(result["phase"], "failed")
        self.assertEqual(result["reason"], "operator_cancelled_before_recording")

    def test_odom_waypoint_goal_updates_after_robot_motion(self) -> None:
        rclpy.init()
        publisher = OdomWaypointGoalPublisher()
        listener = AcquisitionListener()
        self.nodes.extend((publisher, listener))
        odom_subscription = next(
            subscription
            for subscription in publisher.subscriptions
            if subscription.topic_name == "/odom"
        )
        self.assertIs(
            odom_subscription.qos_profile.reliability,
            ReliabilityPolicy.BEST_EFFORT,
        )

        odometry = Odometry()
        odometry.header.frame_id = "odom"
        odometry.pose.pose.orientation.w = 1.0
        publisher._on_odom(odometry)
        self._spin([publisher, listener], 0.3)
        self.assertTrue(listener.goals)
        self.assertAlmostEqual(listener.goals[-1].rho_m, 3.0, places=5)

        odometry.pose.pose.position.x = 1.0
        publisher._on_odom(odometry)
        self._spin([publisher, listener], 0.3)
        self.assertAlmostEqual(listener.goals[-1].rho_m, 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
