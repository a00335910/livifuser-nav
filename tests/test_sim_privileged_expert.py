"""Privileged expert and geometry-layer tests.

The central test here is preregistration Gate B, label invariance: an episode
generated with a corrupted sensor layer must produce byte-identical expert
labels to the matched clean episode. If that fails, every sensor-failure
condition in the study is training on corrupt labels and the study is void.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_sim.analytic_lidar import (  # noqa: E402
    AnalyticLidarGeometry,
    Pose2D,
    raycast_distance,
    simulate_ranges,
)
from livifuser_sim.privileged_expert import (  # noqa: E402
    MAX_ANGULAR_RADPS,
    MAX_LINEAR_MPS,
    ExpertLimits,
    PlannerSpecification,
    build_clearance_field,
    follow_path,
    plan_path,
    privileged_command,
    simulate_expert_episode,
)
from livifuser_sim.world_layers import (  # noqa: E402
    LAYER_CAMERA,
    LAYER_COLLISION,
    LAYER_EXPERT,
    LAYER_LIDAR,
    geometry_for_layer,
    load_world,
    parse_world,
    point_clearance,
)

CONTROL_PERIOD_S = 0.1


def all_layers(**overrides: bool) -> dict:
    layers = {layer: True for layer in (LAYER_COLLISION, LAYER_EXPERT, LAYER_CAMERA, LAYER_LIDAR)}
    layers.update(overrides)
    return layers


def corridor_payload() -> dict:
    """A straight corridor with one low obstacle on the centre line.

    The low obstacle is the C4 case: physically real, camera-visible, known to
    the privileged expert, and absent from the planar LiDAR.
    """

    return {
        "schema_version": 2,
        "name": "dev_corridor_low_obstacle",
        "source": "unit test fixture; not a confirmatory world",
        "laser": {
            "beam_count": 400,
            "angle_min_rad": 0.0,
            "angle_max_rad": 2.0 * math.pi,
            "scan_time_sec": 0.1,
            "range_min_m": 0.12,
            "range_max_m": 8.0,
            "frame_id": "base_scan",
        },
        "bounds_min_xy_m": [0.0, -2.0],
        "bounds_max_xy_m": [6.0, 2.0],
        "obstacles": [
            {
                "name": "left_wall",
                "type": "box",
                "center_xy_m": [3.0, 1.2],
                "size_xy_m": [6.0, 0.1],
                "yaw_rad": 0.0,
                "layers": all_layers(),
            },
            {
                "name": "right_wall",
                "type": "box",
                "center_xy_m": [3.0, -1.2],
                "size_xy_m": [6.0, 0.1],
                "yaw_rad": 0.0,
                "layers": all_layers(),
            },
            {
                "name": "low_box",
                "type": "circle",
                "center_xy_m": [2.5, 0.0],
                "radius_m": 0.20,
                "layers": all_layers(lidar=False),
            },
        ],
    }


# At the locked 0.08 m/s, 4.5 m of corridor needs at least 531 ticks at 10 Hz
# before any detour or turn-rate speed scaling. 900 leaves headroom without
# letting a stuck episode run forever.
def roll_episode(world, *, start=(0.5, 0.0, 0.0), goal=(5.0, 0.0), ticks=900):
    """Roll an episode through the production path.

    Delegates to ``simulate_expert_episode`` deliberately. Gate B must exercise
    the code that actually generates episodes; a parallel integrator in the test
    file could stay invariant while the real one did not.
    """

    rollout = simulate_expert_episode(
        world,
        start,
        goal,
        control_period_s=CONTROL_PERIOD_S,
        max_ticks=ticks,
    )
    return list(rollout.labels), list(rollout.poses)


class TestLabelInvariance(unittest.TestCase):
    """Preregistration Gate B."""

    def test_expert_signature_admits_no_sensor_input(self):
        # Structural guarantee: corruption cannot reach the labels because there
        # is no parameter to carry it. This pins the property against a future
        # edit that quietly adds a scan argument "just for the clearance check".
        forbidden = {"scan", "ranges", "angles", "laser_scan", "observation", "image"}
        for function in (privileged_command, follow_path, plan_path):
            parameters = set(inspect.signature(function).parameters)
            self.assertEqual(
                parameters & forbidden,
                set(),
                f"{function.__name__} must not accept sensor input",
            )

    def test_labels_identical_when_lidar_layer_is_corrupted(self):
        clean = parse_world(corridor_payload())
        corrupted_payload = copy.deepcopy(corridor_payload())
        for item in corrupted_payload["obstacles"]:
            if item["name"] == "low_box":
                item["layers"][LAYER_LIDAR] = True
            else:
                item["layers"][LAYER_LIDAR] = False
        corrupted = parse_world(corrupted_payload)

        clean_labels, clean_poses = roll_episode(clean)
        corrupted_labels, corrupted_poses = roll_episode(corrupted)

        self.assertEqual(clean_labels, corrupted_labels)
        self.assertEqual(clean_poses, corrupted_poses)

    def test_labels_identical_when_camera_layer_is_corrupted(self):
        clean = parse_world(corridor_payload())
        corrupted_payload = copy.deepcopy(corridor_payload())
        for item in corrupted_payload["obstacles"]:
            item["layers"][LAYER_CAMERA] = False
        corrupted = parse_world(corrupted_payload)

        self.assertEqual(roll_episode(clean)[0], roll_episode(corrupted)[0])

    def test_scan_content_differs_while_labels_do_not(self):
        """The corruption must be real, or invariance is vacuous."""

        clean = parse_world(corridor_payload())
        visible_payload = copy.deepcopy(corridor_payload())
        for item in visible_payload["obstacles"]:
            if item["name"] == "low_box":
                item["layers"][LAYER_LIDAR] = True
        visible = parse_world(visible_payload)

        pose = Pose2D(1.9, 0.0, 0.0)
        forward = 0.0
        clean_range = raycast_distance(
            _lidar_geometry(clean), pose, forward
        )
        visible_range = raycast_distance(
            _lidar_geometry(visible), pose, forward
        )
        # The low obstacle is 0.4 m ahead and invisible to the clean scan.
        self.assertAlmostEqual(visible_range, 0.4, places=6)
        self.assertGreater(clean_range, 3.0)
        self.assertEqual(roll_episode(clean)[0], roll_episode(visible)[0])

    def test_repeated_generation_is_deterministic(self):
        world = parse_world(corridor_payload())
        self.assertEqual(roll_episode(world)[0], roll_episode(world)[0])

    def test_replanning_every_tick_is_also_invariant(self):
        clean = parse_world(corridor_payload())
        corrupted_payload = copy.deepcopy(corridor_payload())
        for item in corrupted_payload["obstacles"]:
            if item["name"] == "low_box":
                item["layers"][LAYER_LIDAR] = True
        corrupted = parse_world(corrupted_payload)

        for pose in ((0.5, 0.0, 0.0), (1.8, 0.2, 0.3), (2.4, -0.6, -0.2)):
            first = privileged_command(clean, *pose, (5.0, 0.0))
            second = privileged_command(corrupted, *pose, (5.0, 0.0))
            self.assertEqual(first, second)


def _lidar_geometry(world):
    """Adapt a layered world to the analytic ray caster's LiDAR view."""

    return geometry_for_layer(world, LAYER_LIDAR)


class TestLidarLayerLeak(unittest.TestCase):
    """A LiDAR-invisible obstacle must not influence a single beam.

    This is the C4 analogue of label invariance. The failure it guards against
    is worse than a crash: if a LiDAR-invisible obstacle leaks into the scan,
    C4 still runs, still terminates normally, and still produces clean numbers
    — reporting that fusion confers no benefit at C4. That reads as a null
    result on the primary thesis (§1.1a) when it is really a plumbing bug.
    """

    def _deleted_variant(self, payload, name):
        """The same world with one obstacle removed outright."""

        reduced = copy.deepcopy(payload)
        reduced["obstacles"] = [
            item for item in reduced["obstacles"] if item["name"] != name
        ]
        return parse_world(reduced)

    def test_hidden_obstacle_is_indistinguishable_from_a_deleted_one(self):
        payload = corridor_payload()
        hidden = parse_world(payload)
        deleted = self._deleted_variant(payload, "low_box")

        hidden_geometry = geometry_for_layer(hidden, LAYER_LIDAR)
        deleted_geometry = geometry_for_layer(deleted, LAYER_LIDAR)

        for pose in (
            Pose2D(1.9, 0.0, 0.0),
            Pose2D(2.3, 0.15, 0.4),
            Pose2D(2.5, -0.5, -1.1),
            Pose2D(3.2, 0.0, math.pi),
        ):
            self.assertEqual(
                simulate_ranges(hidden_geometry, pose),
                simulate_ranges(deleted_geometry, pose),
                f"LiDAR-invisible obstacle influenced a beam at {pose}",
            )

    def test_full_obstacle_set_would_have_leaked(self):
        """Proves the guard above is not vacuous."""

        world = parse_world(corridor_payload())
        leaked = AnalyticLidarGeometry(
            schema_version=1,
            source="deliberate leak",
            laser=world.laser,
            obstacles=tuple(entry.obstacle for entry in world.obstacles),
        )
        correct = geometry_for_layer(world, LAYER_LIDAR)
        pose = Pose2D(1.9, 0.0, 0.0)
        self.assertNotEqual(
            simulate_ranges(leaked, pose), simulate_ranges(correct, pose)
        )

    def test_expert_still_sees_the_obstacle_the_scan_cannot(self):
        """The layers must diverge, or C4 is not being constructed at all."""

        world = parse_world(corridor_payload())
        self.assertIn(
            "low_box",
            [obstacle.name for obstacle in world.layer(LAYER_EXPERT)],
        )
        self.assertNotIn(
            "low_box",
            [obstacle.name for obstacle in world.layer(LAYER_LIDAR)],
        )

    def test_geometry_for_layer_rejects_an_unknown_layer(self):
        world = parse_world(corridor_payload())
        with self.assertRaises(ValueError):
            geometry_for_layer(world, "policy")

    def test_tracked_lab_world_matches_the_v1_geometry_contract(self):
        """v2 adds layers; it must not silently re-author the geometry.

        The v1 contract is pinned to the SDF by
        ``test_sim_analytic_lidar.test_config_obstacles_are_pinned_to_world_geometry``,
        so pinning v2 to v1 carries that parity forward transitively.
        """

        package_config = PACKAGE_ROOT / "config"
        version_one = json.loads(
            (package_config / "livifuser_lab_geometry_v1.json").read_text(
                encoding="utf-8"
            )
        )
        version_two = json.loads(
            (package_config / "livifuser_lab_world_v2.json").read_text(encoding="utf-8")
        )
        self.assertEqual(version_one["laser"], version_two["laser"])

        def comparable(items):
            return [
                {key: value for key, value in item.items() if key != "layers"}
                for item in items
            ]

        self.assertEqual(
            comparable(version_one["obstacles"]), comparable(version_two["obstacles"])
        )

    def test_tracked_lab_world_hides_nothing_from_the_scan(self):
        world = load_world(PACKAGE_ROOT / "config" / "livifuser_lab_world_v2.json")
        self.assertEqual(world.lidar_invisible, ())
        self.assertEqual(len(world.layer(LAYER_LIDAR)), len(world.obstacles))


class TestPrivilegedExpertBehaviour(unittest.TestCase):
    def test_reaches_goal_in_the_corridor(self):
        world = parse_world(corridor_payload())
        labels, poses = roll_episode(world)
        self.assertEqual(labels[-1][2], "goal_reached")
        final_x, final_y, _ = poses[-1]
        self.assertLessEqual(math.hypot(5.0 - final_x, 0.0 - final_y), 0.25)

    def test_avoids_the_lidar_invisible_obstacle(self):
        """C4: the expert must route around an obstacle the scan cannot see."""

        world = parse_world(corridor_payload())
        _, poses = roll_episode(world)
        closest = min(math.hypot(x - 2.5, y - 0.0) for x, y, _ in poses)
        # Obstacle radius 0.20 plus the robot footprint; never intersect it.
        self.assertGreater(closest, 0.20 + 0.105)

    def test_commands_stay_within_turtlebot_limits(self):
        world = parse_world(corridor_payload())
        labels, _ = roll_episode(world)
        for linear, angular, _ in labels:
            self.assertGreaterEqual(linear, 0.0)
            self.assertLessEqual(linear, MAX_LINEAR_MPS + 1e-12)
            self.assertLessEqual(abs(angular), MAX_ANGULAR_RADPS + 1e-12)

    def test_unreachable_goal_yields_stop_not_motion(self):
        payload = corridor_payload()
        payload["obstacles"].append(
            {
                "name": "full_barrier",
                "type": "box",
                "center_xy_m": [3.5, 0.0],
                "size_xy_m": [0.2, 2.4],
                "yaw_rad": 0.0,
                "layers": all_layers(),
            }
        )
        world = parse_world(payload)
        command = privileged_command(world, 0.5, 0.0, 0.0, (5.0, 0.0))
        self.assertEqual(command.reason, "no_path")
        self.assertEqual(command.linear_mps, 0.0)
        self.assertEqual(command.angular_radps, 0.0)

    def test_goal_tolerance_reports_reached(self):
        world = parse_world(corridor_payload())
        command = privileged_command(world, 4.9, 0.0, 0.0, (5.0, 0.0))
        self.assertEqual(command.reason, "goal_reached")
        self.assertEqual(command.linear_mps, 0.0)

    def test_speed_falls_as_clearance_falls(self):
        world = parse_world(corridor_payload())
        field = build_clearance_field(world)
        path = plan_path(field, (0.5, 0.0), (5.0, 0.0))
        limits = ExpertLimits()
        open_command = follow_path(path, 0.5, 0.0, 0.0, (5.0, 0.0), 1.0, limits)
        tight_command = follow_path(path, 0.5, 0.0, 0.0, (5.0, 0.0), 0.20, limits)
        self.assertLess(tight_command.linear_mps, open_command.linear_mps)
        self.assertEqual(tight_command.reason, "cautious_advance")

    def test_clearance_below_stop_threshold_halts(self):
        world = parse_world(corridor_payload())
        field = build_clearance_field(world)
        path = plan_path(field, (0.5, 0.0), (5.0, 0.0))
        command = follow_path(path, 0.5, 0.0, 0.0, (5.0, 0.0), 0.10)
        self.assertEqual(command.reason, "clearance_stop")
        self.assertEqual(command.linear_mps, 0.0)


class TestGeometryLayerContract(unittest.TestCase):
    def test_collision_and_expert_membership_must_agree(self):
        payload = corridor_payload()
        payload["obstacles"][2]["layers"][LAYER_EXPERT] = False
        with self.assertRaises(ValueError) as raised:
            parse_world(payload)
        self.assertIn("collision and expert", str(raised.exception))

    def test_every_layer_must_be_declared_explicitly(self):
        payload = corridor_payload()
        del payload["obstacles"][0]["layers"][LAYER_CAMERA]
        with self.assertRaises(ValueError) as raised:
            parse_world(payload)
        self.assertIn("missing", str(raised.exception))

    def test_unknown_layer_is_rejected(self):
        payload = corridor_payload()
        payload["obstacles"][0]["layers"]["policy"] = True
        with self.assertRaises(ValueError):
            parse_world(payload)

    def test_schema_version_one_is_rejected(self):
        payload = corridor_payload()
        payload["schema_version"] = 1
        with self.assertRaises(ValueError):
            parse_world(payload)

    def test_lidar_invisible_set_reports_the_c4_obstacle(self):
        world = parse_world(corridor_payload())
        self.assertEqual(
            [obstacle.name for obstacle in world.lidar_invisible], ["low_box"]
        )

    def test_layer_views_differ_between_lidar_and_camera(self):
        world = parse_world(corridor_payload())
        self.assertEqual(len(world.layer(LAYER_CAMERA)), 3)
        self.assertEqual(len(world.layer(LAYER_LIDAR)), 2)
        self.assertEqual(len(world.layer(LAYER_COLLISION)), 3)

    def test_world_without_colliding_obstacle_is_rejected(self):
        payload = corridor_payload()
        for item in payload["obstacles"]:
            item["layers"][LAYER_COLLISION] = False
            item["layers"][LAYER_EXPERT] = False
        with self.assertRaises(ValueError):
            parse_world(payload)


class TestClearance(unittest.TestCase):
    def test_point_clearance_outside_circle(self):
        world = parse_world(corridor_payload())
        obstacles = world.layer(LAYER_EXPERT)
        self.assertAlmostEqual(point_clearance(obstacles, 2.0, 0.0), 0.30, places=6)

    def test_point_clearance_is_zero_inside_an_obstacle(self):
        world = parse_world(corridor_payload())
        obstacles = world.layer(LAYER_EXPERT)
        self.assertEqual(point_clearance(obstacles, 2.5, 0.0), 0.0)

    def test_clearance_field_matches_direct_evaluation(self):
        world = parse_world(corridor_payload())
        specification = PlannerSpecification()
        field = build_clearance_field(world, specification)
        obstacles = world.layer(LAYER_EXPERT)
        for x_m, y_m in ((1.0, 0.5), (3.0, -0.4), (4.5, 0.0)):
            column, row = field.cell_of(x_m, y_m)
            centre_x, centre_y = field.cell_centre(column, row)
            self.assertAlmostEqual(
                field.clearance_at(x_m, y_m),
                point_clearance(obstacles, centre_x, centre_y),
                places=9,
            )

    def test_cells_inside_inflation_are_not_traversable(self):
        world = parse_world(corridor_payload())
        field = build_clearance_field(world)
        self.assertFalse(field.traversable(*field.cell_of(2.5, 0.0)))
        self.assertTrue(field.traversable(*field.cell_of(0.5, 0.0)))


if __name__ == "__main__":
    unittest.main()
