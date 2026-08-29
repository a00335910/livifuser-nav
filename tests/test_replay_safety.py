import unittest

from livifuser_nav.replay_safety import (
    EXEMPT_PUBLISHERS,
    REFERENCE_COMMAND_TOPICS,
    REPLAY_ALLOWLIST,
    SafetyAuditError,
    assert_publisher_names_safe,
    build_topic_map,
    evaluate_graph,
    is_forbidden_publisher,
)

PILOT_BAG_TOPICS = {
    "/camera/image_raw": "sensor_msgs/msg/Image",
    "/camera/camera_info": "sensor_msgs/msg/CameraInfo",
    "/scan": "sensor_msgs/msg/LaserScan",
    "/odom": "nav_msgs/msg/Odometry",
    "/cmd_vel": "geometry_msgs/msg/Twist",
    "/livifuser/goal_relative": "livifuser_interfaces/msg/RelativeGoal",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/tf_static": "tf2_msgs/msg/TFMessage",
}


class AllowlistTests(unittest.TestCase):
    def test_command_topic_is_never_published_by_default(self) -> None:
        mapping = build_topic_map(PILOT_BAG_TOPICS)
        self.assertNotIn("/cmd_vel", mapping)
        self.assertNotIn("/cmd_vel", mapping.values())

    def test_allowlist_contains_no_command_topic(self) -> None:
        for source, target in REPLAY_ALLOWLIST.items():
            self.assertFalse(is_forbidden_publisher(target), f"{source} -> {target}")

    def test_only_allowlisted_topics_are_published(self) -> None:
        mapping = build_topic_map(PILOT_BAG_TOPICS)
        self.assertEqual(set(mapping), set(REPLAY_ALLOWLIST))

    def test_unknown_bag_topics_are_ignored(self) -> None:
        mapping = build_topic_map({**PILOT_BAG_TOPICS, "/battery_state": "x"})
        self.assertNotIn("/battery_state", mapping)

    def test_absent_allowlist_topics_are_not_invented(self) -> None:
        mapping = build_topic_map({"/scan": "sensor_msgs/msg/LaserScan"})
        self.assertEqual(mapping, {"/scan": "/scan"})


class ReferenceCommandTests(unittest.TestCase):
    def test_recorded_command_is_remapped_to_an_inert_topic(self) -> None:
        mapping = build_topic_map(PILOT_BAG_TOPICS, include_reference_commands=True)
        self.assertEqual(mapping["/cmd_vel"], "/livifuser/replay/reference_cmd_vel")

    def test_no_reference_target_is_a_command_topic(self) -> None:
        for source, target in REFERENCE_COMMAND_TOPICS.items():
            self.assertTrue(target.startswith("/livifuser/replay/"), source)
            self.assertFalse(is_forbidden_publisher(target), source)

    def test_reference_targets_survive_the_audit(self) -> None:
        assert_publisher_names_safe(list(REFERENCE_COMMAND_TOPICS.values()))

    def test_every_reference_target_is_explicitly_exempted(self) -> None:
        self.assertEqual(set(REFERENCE_COMMAND_TOPICS.values()), set(EXEMPT_PUBLISHERS))

    def test_exemption_is_exact_match_not_a_namespace_prefix(self) -> None:
        # A prefix exemption would excuse anything under /livifuser/replay/,
        # including a literal command topic. These must still be refused.
        for name in (
            "/livifuser/replay/cmd_vel",
            "/livifuser/replay/cmd_vel_stamped",
            "/livifuser/replay/motor_left",
            "/livifuser/replay/nested/cmd_vel",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_forbidden_publisher(name))
                with self.assertRaises(SafetyAuditError):
                    assert_publisher_names_safe([name])


class ForbiddenNameTests(unittest.TestCase):
    def test_command_names_are_refused(self) -> None:
        for name in (
            "/cmd_vel",
            "cmd_vel",
            "/robot/cmd_vel",
            "/cmd_vel/managed",
            "/cmd_vel_stamped",
            "/motor_command",
            "/wheel_cmd",
            "/wheelcmd",
            "/joint_trajectory",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_forbidden_publisher(name))

    def test_sensor_names_are_permitted(self) -> None:
        for name in ("/scan", "/odom", "/camera/image_raw", "/tf_static"):
            with self.subTest(name=name):
                self.assertFalse(is_forbidden_publisher(name))

    def test_audit_raises_and_names_every_offender(self) -> None:
        with self.assertRaises(SafetyAuditError) as caught:
            assert_publisher_names_safe(["/scan", "/cmd_vel", "/motor_left"])
        message = str(caught.exception)
        self.assertIn("/cmd_vel", message)
        self.assertIn("/motor_left", message)
        self.assertNotIn("/scan", message)

    def test_a_regressed_allowlist_cannot_pass_the_audit(self) -> None:
        # Second-layer protection: if a future edit added a command topic to the
        # allowlist, building the topic map must still refuse it.
        REPLAY_ALLOWLIST["/cmd_vel"] = "/cmd_vel"
        try:
            with self.assertRaises(SafetyAuditError):
                build_topic_map(PILOT_BAG_TOPICS)
        finally:
            REPLAY_ALLOWLIST.pop("/cmd_vel", None)
        self.assertNotIn("/cmd_vel", build_topic_map(PILOT_BAG_TOPICS))


class GraphProbeTests(unittest.TestCase):
    def test_clean_graph_is_safe(self) -> None:
        probe = evaluate_graph(["livifuser_replay", "rviz"], {"/cmd_vel": 0})
        self.assertTrue(probe.is_safe)

    def test_turtlebot_node_is_detected(self) -> None:
        probe = evaluate_graph(["turtlebot3_node"], {})
        self.assertFalse(probe.is_safe)
        self.assertEqual(probe.robot_nodes, ("turtlebot3_node",))

    def test_detection_is_case_insensitive(self) -> None:
        self.assertFalse(evaluate_graph(["TurtleBot3_Node"], {}).is_safe)

    def test_command_subscriber_is_detected(self) -> None:
        probe = evaluate_graph(["rviz"], {"/cmd_vel": 1})
        self.assertFalse(probe.is_safe)
        self.assertEqual(probe.command_subscribers, (("/cmd_vel", 1),))

    def test_zero_count_subscribers_are_not_flagged(self) -> None:
        self.assertTrue(evaluate_graph([], {"/cmd_vel": 0, "/cmd_vel_stamped": 0}).is_safe)

    def test_lidar_and_base_drivers_are_detected(self) -> None:
        for name in ("hlds_laser_publisher", "coin_d4_driver", "diff_drive_controller"):
            with self.subTest(name=name):
                self.assertFalse(evaluate_graph([name], {}).is_safe)


if __name__ == "__main__":
    unittest.main()
