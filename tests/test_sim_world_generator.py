"""Procedural world generator tests.

Covers the three properties the generator exists to guarantee — determinism,
seed disjointness, validated feasibility — plus the C4 derivation, which must
change the LiDAR layer and nothing else (preregistration §8).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_sim.privileged_expert import simulate_expert_episode  # noqa: E402
from livifuser_sim.world_generator import (  # noqa: E402
    ARCHETYPES,
    DERIVED_GROUPS,
    GROUP_SEED_BLOCKS,
    MAX_BLIND_ROUTE_SWITCHABLE_CLEARANCE_M,
    MAX_EPISODE_SECONDS,
    MAX_SWITCHABLE_PATH_CLEARANCE_M,
    MIN_GOAL_DISTANCE_M,
    MIN_PATH_EXCESS_RATIO,
    derive_condition,
    generate_world,
    seed_for,
    validate_world,
)
from livifuser_sim.world_layers import (  # noqa: E402
    DECLARABLE_LAYERS,
    LAYER_CAMERA,
    LAYER_COLLISION,
    LAYER_EXPERT,
    LAYER_LIDAR,
    parse_world,
)

GROUP_COUNTS = {"dev": 2, "train": 6, "val_id": 2, "test_id": 2}


class TestSeedAllocation(unittest.TestCase):
    def test_group_blocks_are_disjoint_across_plausible_counts(self):
        seen: dict[int, str] = {}
        for group in GROUP_SEED_BLOCKS:
            for index in range(500):
                seed = seed_for(group, index)
                self.assertNotIn(
                    seed, seen, f"{group}[{index}] collides with {seen.get(seed)}"
                )
                seen[seed] = f"{group}[{index}]"

    def test_derived_groups_cannot_be_generated(self):
        for group in DERIVED_GROUPS:
            with self.assertRaises(ValueError) as raised:
                seed_for(group, 0)
            self.assertIn("derived", str(raised.exception))

    def test_unknown_group_is_rejected(self):
        with self.assertRaises(ValueError):
            seed_for("holdout", 0)

    def test_negative_index_is_rejected(self):
        with self.assertRaises(ValueError):
            seed_for("dev", -1)

    def test_seed_is_recorded_in_the_payload(self):
        payload = generate_world("dev", 0)
        self.assertEqual(payload["seed"], seed_for("dev", 0))


class TestDeterminism(unittest.TestCase):
    def test_same_group_and_index_reproduce_the_payload(self):
        self.assertEqual(generate_world("train", 2), generate_world("train", 2))

    def test_different_indices_give_different_geometry(self):
        first = generate_world("train", 0)
        second = generate_world("val_id", 0)
        self.assertNotEqual(first["obstacles"], second["obstacles"])

    def test_redraws_are_reproducible(self):
        """A world that needed retries must still regenerate identically."""

        payloads = [generate_world("dev", 0) for _ in range(3)]
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[1], payloads[2])


class TestGeneratedWorldsAreValid(unittest.TestCase):
    def test_every_configured_world_generates_and_validates(self):
        for group, count in GROUP_COUNTS.items():
            for index in range(count):
                with self.subTest(group=group, index=index):
                    payload = generate_world(group, index)
                    world = parse_world(payload)
                    report = validate_world(world)
                    self.assertTrue(report.accepted, report.reason)
                    self.assertGreaterEqual(
                        report.goal_distance_m, MIN_GOAL_DISTANCE_M
                    )
                    self.assertGreaterEqual(
                        report.path_excess_ratio, MIN_PATH_EXCESS_RATIO
                    )
                    self.assertLessEqual(report.episode_duration_s, MAX_EPISODE_SECONDS)
                    self.assertLessEqual(
                        report.switchable_path_clearance_m,
                        MAX_SWITCHABLE_PATH_CLEARANCE_M,
                    )
                    self.assertLessEqual(
                        report.blind_route_switchable_clearance_m,
                        MAX_BLIND_ROUTE_SWITCHABLE_CLEARANCE_M,
                    )
                    self.assertEqual(
                        payload["validation"][
                            "blind_route_switchable_clearance_m"
                        ],
                        round(report.blind_route_switchable_clearance_m, 4),
                    )

    def test_train_group_covers_every_archetype(self):
        archetypes = {
            generate_world("train", index)["archetype"]
            for index in range(len(ARCHETYPES))
        }
        self.assertEqual(archetypes, set(ARCHETYPES))

    def test_expert_reaches_the_goal_in_every_world(self):
        for group, count in GROUP_COUNTS.items():
            for index in range(count):
                with self.subTest(group=group, index=index):
                    world = parse_world(generate_world(group, index))
                    rollout = simulate_expert_episode(
                        world, world.start_pose_xy_yaw, world.goal_xy_m
                    )
                    self.assertTrue(rollout.reached)

    def test_every_world_carries_a_switchable_obstacle(self):
        for group, count in GROUP_COUNTS.items():
            for index in range(count):
                with self.subTest(group=group, index=index):
                    world = parse_world(generate_world(group, index))
                    self.assertTrue(world.profile_switchable)

    def test_every_switchable_obstacle_has_a_low_render_profile(self):
        for group, count in GROUP_COUNTS.items():
            for index in range(count):
                payload = generate_world(group, index)
                for item in payload["obstacles"]:
                    self.assertIn("render", item)
                    if item.get("profile_switchable"):
                        self.assertLess(item["render"]["height_m"], 0.172)

    def test_matched_id_world_hides_nothing_from_the_scan(self):
        world = parse_world(generate_world("test_id", 0))
        self.assertEqual(world.lidar_invisible, ())


class TestConditionDerivation(unittest.TestCase):
    def test_world_condition_is_explicit_after_parsing(self):
        payload = generate_world("test_id", 0)
        self.assertEqual(parse_world(payload).condition, "C0")
        self.assertEqual(
            parse_world(derive_condition(payload, "C1")).condition, "C1"
        )
        self.assertEqual(
            parse_world(derive_condition(payload, "C4")).condition, "C4"
        )

    def test_c4_changes_only_the_lidar_layer(self):
        payload = generate_world("test_id", 0)
        variant = derive_condition(payload, "C4")
        self.assertEqual(len(payload["obstacles"]), len(variant["obstacles"]))
        for before, after in zip(
            payload["obstacles"], variant["obstacles"], strict=True
        ):
            for key in ("name", "type", "center_xy_m"):
                self.assertEqual(before[key], after[key])
            for layer in (LAYER_COLLISION, LAYER_EXPERT, LAYER_CAMERA):
                self.assertEqual(before["layers"][layer], after["layers"][layer])
            if before.get("profile_switchable"):
                self.assertFalse(after["layers"][LAYER_LIDAR])
            else:
                self.assertEqual(
                    before["layers"][LAYER_LIDAR], after["layers"][LAYER_LIDAR]
                )

    def test_c4_preserves_start_goal_and_seed(self):
        payload = generate_world("test_id", 1)
        variant = derive_condition(payload, "C4")
        for key in ("start_pose_xy_yaw", "goal_xy_m", "seed", "archetype"):
            self.assertEqual(payload[key], variant[key])

    def test_c1_changes_only_the_camera_condition_descriptor(self):
        payload = generate_world("test_id", 0)
        variant = derive_condition(payload, "C1")
        self.assertEqual(variant["condition"], "C1")
        self.assertEqual(
            variant["camera_condition"]["name"], "C1_WARM_LOW_LIGHT_V1"
        )
        unchanged = {
            key: value
            for key, value in variant.items()
            if key not in {"condition", "camera_condition"}
        }
        self.assertEqual(unchanged, payload)

    def test_c1_preserves_privileged_labels(self):
        payload = generate_world("test_id", 1)
        matched = parse_world(payload)
        variant = parse_world(derive_condition(payload, "C1"))
        first = simulate_expert_episode(
            matched, matched.start_pose_xy_yaw, matched.goal_xy_m
        )
        second = simulate_expert_episode(
            variant, variant.start_pose_xy_yaw, variant.goal_xy_m
        )
        self.assertEqual(first.labels, second.labels)
        self.assertEqual(first.poses, second.poses)

    def test_c4_labels_are_identical_to_the_matched_id_world(self):
        for index in range(GROUP_COUNTS["test_id"]):
            with self.subTest(index=index):
                payload = generate_world("test_id", index)
                matched = parse_world(payload)
                variant = parse_world(derive_condition(payload, "C4"))
                first = simulate_expert_episode(
                    matched, matched.start_pose_xy_yaw, matched.goal_xy_m
                )
                second = simulate_expert_episode(
                    variant, variant.start_pose_xy_yaw, variant.goal_xy_m
                )
                self.assertEqual(first.labels, second.labels)
                self.assertEqual(first.poses, second.poses)

    def test_c4_actually_hides_something_from_the_scan(self):
        payload = generate_world("test_id", 0)
        variant = parse_world(derive_condition(payload, "C4"))
        self.assertTrue(variant.lidar_invisible)

    def test_c0_returns_the_world_unchanged(self):
        payload = generate_world("test_id", 0)
        self.assertEqual(derive_condition(payload, "C0"), payload)

    def test_observation_conditions_are_rejected(self):
        payload = generate_world("test_id", 0)
        for condition in ("C3", "C3a", "C3b"):
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                derive_condition(payload, condition)

    def test_conditions_cannot_be_composed(self):
        payload = generate_world("test_id", 0)
        for first, second in (("C1", "C1"), ("C1", "C4"), ("C4", "C1")):
            with self.subTest(first=first, second=second), self.assertRaises(ValueError):
                derive_condition(derive_condition(payload, first), second)

    def test_deriving_c4_twice_does_not_hide_more(self):
        payload = generate_world("test_id", 0)
        once = derive_condition(payload, "C4")
        with self.assertRaises(ValueError):
            derive_condition(once, "C4")

    def test_switchable_obstacle_must_start_in_every_layer(self):
        payload = generate_world("test_id", 0)
        for item in payload["obstacles"]:
            if item.get("profile_switchable"):
                item["layers"][LAYER_LIDAR] = False
        with self.assertRaises(ValueError) as raised:
            parse_world(payload)
        self.assertIn("every layer", str(raised.exception))

    def test_switchable_obstacles_are_in_all_layers_when_generated(self):
        payload = generate_world("train", 0)
        for item in payload["obstacles"]:
            if item.get("profile_switchable"):
                for layer in DECLARABLE_LAYERS:
                    self.assertTrue(item["layers"][layer])


if __name__ == "__main__":
    unittest.main()
