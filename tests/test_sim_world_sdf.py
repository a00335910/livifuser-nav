"""Generated-world to Gazebo SDF materialization tests."""

from __future__ import annotations

import copy
import math
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_sim.world_generator import (  # noqa: E402
    LIDAR_SCAN_HEIGHT_M,
    LOW_OBSTACLE_HEIGHT_M,
    derive_condition,
    generate_world,
)
from livifuser_sim.world_sdf import render_world_sdf  # noqa: E402

TEMPLATE = PACKAGE_ROOT / "worlds" / "livifuser_lab.sdf"


def root_for(payload: dict) -> ET.Element:
    return ET.fromstring(render_world_sdf(payload, TEMPLATE))


def generated_models(root: ET.Element) -> dict[str, ET.Element]:
    world = root.find("world")
    assert world is not None
    return {
        model.get("name", ""): model
        for model in world.findall("model")
        if model.get("name", "").startswith("generated_")
    }


def normalized_visual_geometry(model: ET.Element) -> list[str]:
    return [
        ET.tostring(node, encoding="unicode").replace("_c1.", "_c0.")
        for node in model.findall("link/visual/geometry")
    ]


class TestWorldSdfMaterialization(unittest.TestCase):
    def test_robot_dynamics_match_official_burger_fixed_joint_lump(self):
        root = root_for(generate_world("dev", 0))
        robot = root.find("world/model[@name='livifuser_burger']")
        self.assertIsNotNone(robot)
        base = robot.find("link[@name='base_link']")
        self.assertIsNotNone(base)

        inertial_pose = [float(value) for value in base.findtext("inertial/pose").split()]
        self.assertEqual(inertial_pose[:3], [-0.00429009175, 0.0, 0.0307338557])
        self.assertAlmostEqual(float(base.findtext("inertial/mass")), 0.94473504)
        self.assertAlmostEqual(
            float(base.findtext("inertial/inertia/ixz")), 0.000576740468
        )

        body = base.find("collision[@name='body_collision']")
        caster = base.find("collision[@name='caster_back_collision']")
        self.assertIsNotNone(body)
        self.assertIsNotNone(caster)
        self.assertEqual(body.findtext("geometry/box/size"), "0.140 0.140 0.143")
        self.assertEqual(caster.findtext("geometry/box/size"), "0.030 0.009 0.020")
        self.assertEqual(caster.findtext("surface/friction/ode/mu"), "0.01")
        self.assertIsNone(robot.find("link[@name='caster']"))
        self.assertIsNone(robot.find("joint[@name='caster_joint']"))

        self.assertAlmostEqual(
            float(robot.findtext("link[@name='left_wheel']/inertial/mass")),
            0.02849894,
        )
        for side, lateral in (("left", 0.080), ("right", -0.080)):
            wheel_pose = [
                float(value)
                for value in robot.findtext(f"link[@name='{side}_wheel']/pose").split()
            ]
            wheel_pose_element = robot.find(f"link[@name='{side}_wheel']/pose")
            joint = robot.find(f"joint[@name='{side}_wheel_joint']")
            joint_pose_element = joint.find("pose")
            joint_pose = [float(value) for value in joint_pose_element.text.split()]
            self.assertEqual(wheel_pose, [0.0] * 6)
            self.assertEqual(
                wheel_pose_element.get("relative_to"), f"{side}_wheel_joint"
            )
            self.assertEqual(joint_pose[:3], [0.0, lateral, 0.033])
            self.assertAlmostEqual(joint_pose[3], math.pi / 2.0)
            self.assertEqual(joint_pose_element.get("relative_to"), "base_link")
            self.assertEqual(joint.findtext("axis/xyz"), "0 0 -1")

        drive = robot.find("plugin[@name='gz::sim::systems::DiffDrive']")
        self.assertEqual(drive.findtext("max_linear_acceleration"), "0.08")
        self.assertEqual(drive.findtext("min_linear_acceleration"), "-0.08")
        self.assertEqual(drive.findtext("max_angular_acceleration"), "0.40")
        self.assertEqual(drive.findtext("min_angular_acceleration"), "-0.40")

    def test_every_obstacle_is_materialized(self):
        payload = generate_world("dev", 0)
        models = generated_models(root_for(payload))
        self.assertEqual(len(models), len(payload["obstacles"]))
        for obstacle in payload["obstacles"]:
            self.assertIn(f"generated_{obstacle['name']}", models)

    def test_robot_starts_at_generated_pose(self):
        payload = generate_world("dev", 1)
        root = root_for(payload)
        pose = root.find("world/model[@name='livifuser_burger']/pose")
        self.assertIsNotNone(pose)
        values = [float(value) for value in pose.text.split()]
        self.assertEqual(values[:2], payload["start_pose_xy_yaw"][:2])
        self.assertAlmostEqual(values[5], payload["start_pose_xy_yaw"][2])

    def test_switchable_obstacle_is_physically_below_scan_plane(self):
        payload = generate_world("dev", 0)
        root = root_for(payload)
        models = generated_models(root)
        for obstacle in payload["obstacles"]:
            if not obstacle.get("profile_switchable"):
                continue
            model = models[f"generated_{obstacle['name']}"]
            pose = [float(value) for value in model.find("pose").text.split()]
            self.assertAlmostEqual(pose[2] * 2.0, LOW_OBSTACLE_HEIGHT_M)
            self.assertLess(LOW_OBSTACLE_HEIGHT_M, LIDAR_SCAN_HEIGHT_M)

    def test_c0_and_c4_render_and_collision_geometry_are_identical(self):
        payload = generate_world("dev", 0)
        c0_models = generated_models(root_for(payload))
        c4_models = generated_models(root_for(derive_condition(payload, "C4")))
        self.assertEqual(c0_models.keys(), c4_models.keys())
        for name in c0_models:
            self.assertEqual(
                ET.tostring(c0_models[name], encoding="unicode"),
                ET.tostring(c4_models[name], encoding="unicode"),
            )

    def test_c1_changes_scene_and_materials_but_not_geometry(self):
        payload = generate_world("dev", 0)
        c0_root = root_for(payload)
        c1_root = root_for(derive_condition(payload, "C1"))
        c0_world = c0_root.find("world")
        c1_world = c1_root.find("world")
        self.assertIsNotNone(c0_world)
        self.assertIsNotNone(c1_world)

        self.assertEqual(c1_world.findtext("scene/ambient"), "0.25 0.18 0.12 1")
        self.assertEqual(
            c1_world.findtext("light[@name='sun']/diffuse"), "0.55 0.38 0.22 1"
        )
        self.assertNotEqual(
            c0_world.findtext("scene/ambient"), c1_world.findtext("scene/ambient")
        )

        c0_models = generated_models(c0_root)
        c1_models = generated_models(c1_root)
        self.assertEqual(c0_models.keys(), c1_models.keys())
        material_changed = False
        for name in c0_models:
            before = c0_models[name]
            after = c1_models[name]
            self.assertEqual(before.findtext("pose"), after.findtext("pose"))
            self.assertEqual(
                ET.tostring(before.find("link/collision"), encoding="unicode"),
                ET.tostring(after.find("link/collision"), encoding="unicode"),
            )
            self.assertEqual(
                normalized_visual_geometry(before),
                normalized_visual_geometry(after),
            )
            material_changed = material_changed or (
                ET.tostring(before.find("link/visual/material"), encoding="unicode")
                != ET.tostring(after.find("link/visual/material"), encoding="unicode")
            )
        self.assertTrue(material_changed)

        self.assertEqual(
            ET.tostring(
                c0_world.find("model[@name='livifuser_burger']/link/sensor"),
                encoding="unicode",
            ),
            ET.tostring(
                c1_world.find("model[@name='livifuser_burger']/link/sensor"),
                encoding="unicode",
            ),
        )

    def test_c1_material_palette_permutation_is_exact(self):
        payload = generate_world("dev", 0)
        c0_models = generated_models(root_for(payload))
        c1_models = generated_models(root_for(derive_condition(payload, "C1")))
        name = next(iter(c0_models))
        before = [
            float(value)
            for value in c0_models[name].findtext("link/visual/material/diffuse").split()
        ]
        after = [
            float(value)
            for value in c1_models[name].findtext("link/visual/material/diffuse").split()
        ]
        self.assertEqual(after, [before[2], before[0], before[1], before[3]])

    def test_c1_rejects_drifted_descriptor(self):
        payload = derive_condition(generate_world("dev", 0), "C1")
        payload["camera_condition"] = dict(payload["camera_condition"])
        payload["camera_condition"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen visual contract"):
            render_world_sdf(payload, TEMPLATE)

    def test_invalid_tall_c4_obstacle_is_rejected(self):
        payload = generate_world("dev", 0)
        for obstacle in payload["obstacles"]:
            if obstacle.get("profile_switchable"):
                obstacle["render"]["height_m"] = LIDAR_SCAN_HEIGHT_M
        with self.assertRaises(ValueError) as raised:
            render_world_sdf(payload, TEMPLATE)
        self.assertIn("scan plane", str(raised.exception))

    def test_missing_render_contract_is_rejected(self):
        payload = generate_world("dev", 0)
        del payload["obstacles"][0]["render"]
        with self.assertRaises(ValueError):
            render_world_sdf(payload, TEMPLATE)

    def test_camera_and_collision_membership_control_sdf_parts(self):
        payload = generate_world("dev", 0)
        candidate = copy.deepcopy(payload)
        item = candidate["obstacles"][0]
        item["layers"]["camera"] = False
        model = generated_models(root_for(candidate))[f"generated_{item['name']}"]
        self.assertIsNotNone(model.find("link/collision"))
        self.assertIsNone(model.find("link/visual"))


if __name__ == "__main__":
    unittest.main()
