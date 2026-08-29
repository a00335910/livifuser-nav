"""Static-map leakage checks for the bounded Nav2 structural probe."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_sim.nav2_map import FREE, OCCUPIED, rasterize_nav2_map  # noqa: E402
from livifuser_sim.world_generator import derive_condition, generate_world  # noqa: E402
from livifuser_sim.world_layers import parse_world  # noqa: E402


class TestNav2Map(unittest.TestCase):
    def test_permanent_obstacles_are_occupied(self) -> None:
        world = parse_world(generate_world("dev", 1))
        nav_map = rasterize_nav2_map(world)
        permanent = next(
            entry
            for entry in world.obstacles
            if entry.in_layer("collision") and not entry.profile_switchable
        )
        self.assertEqual(
            nav_map.pixel_at_world(
                permanent.obstacle.center_x_m, permanent.obstacle.center_y_m
            ),
            OCCUPIED,
        )

    def test_switchable_obstacles_are_excluded(self) -> None:
        world = parse_world(generate_world("dev", 1))
        nav_map = rasterize_nav2_map(world)
        for entry in world.obstacles:
            if entry.profile_switchable:
                self.assertEqual(
                    nav_map.pixel_at_world(
                        entry.obstacle.center_x_m, entry.obstacle.center_y_m
                    ),
                    FREE,
                )
                self.assertIn(entry.name, nav_map.excluded_switchable_obstacles)

    def test_c0_and_c4_maps_are_bitwise_identical(self) -> None:
        c0_payload = generate_world("dev", 0)
        c4_payload = derive_condition(c0_payload, "C4")
        c0 = rasterize_nav2_map(parse_world(c0_payload))
        c4 = rasterize_nav2_map(parse_world(c4_payload))
        self.assertEqual(c0.pgm_bytes(), c4.pgm_bytes())
        self.assertEqual(c0.yaml_text("map.pgm"), c4.yaml_text("map.pgm"))

    def test_start_and_goal_are_free(self) -> None:
        world = parse_world(generate_world("dev", 1))
        nav_map = rasterize_nav2_map(world)
        assert world.start_pose_xy_yaw is not None
        assert world.goal_xy_m is not None
        self.assertEqual(nav_map.pixel_at_world(*world.start_pose_xy_yaw[:2]), FREE)
        self.assertEqual(nav_map.pixel_at_world(*world.goal_xy_m), FREE)

    def test_pgm_has_expected_size(self) -> None:
        nav_map = rasterize_nav2_map(parse_world(generate_world("dev", 0)))
        header, dimensions, maximum, pixels = nav_map.pgm_bytes().split(b"\n", 3)
        self.assertEqual(header, b"P5")
        self.assertEqual(dimensions, f"{nav_map.width} {nav_map.height}".encode())
        self.assertEqual(maximum, b"255")
        self.assertEqual(len(pixels), nav_map.width * nav_map.height)


if __name__ == "__main__":
    unittest.main()
