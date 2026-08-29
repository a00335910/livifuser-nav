import json
import math
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_sim.analytic_lidar import (  # noqa: E402
    AnalyticLidarGeometry,
    BoxObstacle,
    CircleObstacle,
    LaserSpecification,
    Pose2D,
    load_geometry,
    raycast_distance,
    simulate_ranges,
)

CONFIG_PATH = PACKAGE_ROOT / "config" / "livifuser_lab_geometry_v1.json"
WORLD_PATH = PACKAGE_ROOT / "worlds" / "livifuser_lab.sdf"


def geometry_with(*obstacles):
    return AnalyticLidarGeometry(
        schema_version=1,
        source="test",
        laser=LaserSpecification(400, 0.0, 2.0 * math.pi, 0.1, 0.12, 8.0, "base_scan"),
        obstacles=tuple(obstacles),
    )


class TestAnalyticLidar(unittest.TestCase):
    def test_circle_intersection_is_exact(self):
        geometry = geometry_with(CircleObstacle("circle", 2.0, 0.0, 0.5))
        distance = raycast_distance(geometry, Pose2D(0.0, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(distance, 1.5)

    def test_rotated_box_intersection_is_exact(self):
        geometry = geometry_with(
            BoxObstacle("box", 2.0, 0.0, 1.0, 1.0, math.pi / 4.0)
        )
        distance = raycast_distance(geometry, Pose2D(0.0, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(distance, 2.0 - math.sqrt(0.5))

    def test_robot_yaw_rotates_the_local_scan(self):
        geometry = geometry_with(CircleObstacle("circle", 2.0, 0.0, 0.5))
        pose = Pose2D(0.0, 0.0, math.pi)
        self.assertTrue(math.isinf(raycast_distance(geometry, pose, 0.0)))
        self.assertAlmostEqual(raycast_distance(geometry, pose, math.pi), 1.5)

    def test_lab_geometry_has_correct_corridor_clearance(self):
        geometry = load_geometry(CONFIG_PATH)
        pose = Pose2D(0.0, 0.0, 0.0)
        self.assertAlmostEqual(raycast_distance(geometry, pose, math.pi / 2.0), 1.5)
        self.assertAlmostEqual(raycast_distance(geometry, pose, -math.pi / 2.0), 1.5)

    def test_scan_matches_observed_lds03_angular_contract(self):
        geometry = load_geometry(CONFIG_PATH)
        ranges = simulate_ranges(geometry, Pose2D(0.0, 0.0, 0.0))
        self.assertEqual(len(ranges), 400)
        self.assertAlmostEqual(geometry.laser.angle_min_rad, 0.0)
        self.assertAlmostEqual(geometry.laser.angle_max_rad, 2.0 * math.pi)
        self.assertAlmostEqual(geometry.laser.angle_increment_rad, 2.0 * math.pi / 401)
        finite = [value for value in ranges if math.isfinite(value)]
        self.assertGreater(len(finite), 100)
        self.assertGreater(max(finite) - min(finite), 0.5)

    def test_config_obstacles_are_pinned_to_world_geometry(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        root = ET.parse(WORLD_PATH).getroot()
        models = {model.attrib["name"]: model for model in root.findall("world/model")}
        for obstacle in config["obstacles"]:
            model = models[obstacle["name"]]
            pose = [float(value) for value in model.findtext("pose").split()]
            self.assertEqual(pose[:2], obstacle["center_xy_m"])
            if obstacle["type"] == "box":
                size = [
                    float(value)
                    for value in model.findtext("link/collision/geometry/box/size").split()
                ]
                self.assertEqual(size[:2], obstacle["size_xy_m"])
                self.assertAlmostEqual(pose[5], obstacle["yaw_rad"])
            else:
                radius = float(
                    model.findtext("link/collision/geometry/cylinder/radius")
                )
                self.assertAlmostEqual(radius, obstacle["radius_m"])


if __name__ == "__main__":
    unittest.main()
