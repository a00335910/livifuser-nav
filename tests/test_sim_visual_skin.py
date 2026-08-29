"""Geometry and appearance gates for the controlled Small House skin."""

from __future__ import annotations

import hashlib
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"
MODEL_ROOT = PACKAGE_ROOT / "models" / "livifuser_visual_skin_v1"
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_sim.visual_skin import (  # noqa: E402
    VISUAL_SKIN_CONFIG_SHA256,
    load_visual_skin_contract,
    verify_installed_visual_assets,
    visual_skin_descriptor,
)
from livifuser_sim.world_generator import derive_condition, generate_world  # noqa: E402
from livifuser_sim.world_sdf import render_world_sdf  # noqa: E402

TEMPLATE = PACKAGE_ROOT / "worlds" / "livifuser_lab.sdf"


def _numbers(text: str | None) -> list[float]:
    assert text is not None
    return [float(value) for value in text.split()]


def _root(payload: dict) -> ET.Element:
    return ET.fromstring(render_world_sdf(payload, TEMPLATE))


class TestVisualSkinContract(unittest.TestCase):
    def test_configuration_and_every_asset_match_pinned_hashes(self) -> None:
        config_path = PACKAGE_ROOT / "config" / "visual_skin_v1.json"
        observed = hashlib.sha256(config_path.read_bytes()).hexdigest().upper()
        self.assertEqual(observed, VISUAL_SKIN_CONFIG_SHA256)
        report = verify_installed_visual_assets()
        self.assertTrue(report["valid"], report["issues"])
        self.assertGreaterEqual(report["asset_count"], 30)

    def test_groups_have_declared_held_out_style_families(self) -> None:
        contract = load_visual_skin_contract()
        self.assertEqual(
            len(
                {
                    contract["group_styles"]["train"],
                    contract["group_styles"]["val_id"],
                    contract["group_styles"]["test_id"],
                }
            ),
            3,
        )
        for group in ("dev", "train", "val_id", "test_id"):
            self.assertEqual(
                generate_world(group, 0)["visual_skin"],
                visual_skin_descriptor(group),
            )

    def test_every_c1_texture_is_the_exact_frozen_channel_permutation(self) -> None:
        texture_root = MODEL_ROOT / "materials" / "textures"
        c0_paths = sorted(texture_root.glob("*_c0.png"))
        self.assertEqual(len(c0_paths), 15)
        for c0_path in c0_paths:
            with self.subTest(texture=c0_path.name):
                c1_path = c0_path.with_name(c0_path.name.replace("_c0.png", "_c1.png"))
                with Image.open(c0_path) as source, Image.open(c1_path) as shifted:
                    c0 = np.asarray(source.convert("RGB"))
                    c1 = np.asarray(shifted.convert("RGB"))
                np.testing.assert_array_equal(c1, c0[:, :, [2, 0, 1]])

    def test_c4_hazard_texture_passes_frozen_visibility_floor(self) -> None:
        gate = load_visual_skin_contract()["c4_visibility_gate"]
        self.assertGreaterEqual(
            gate["observed_minimum_mean_rgb_distance"],
            gate["minimum_hazard_to_floor_mean_rgb_distance"],
        )
        self.assertGreaterEqual(
            gate["observed_minimum_hazard_luminance_std"],
            gate["minimum_hazard_luminance_std"],
        )

    def test_c0_and_c1_mesh_files_have_identical_normalized_geometry(self) -> None:
        hashes = load_visual_skin_contract()["normalized_mesh_geometry_sha256"]
        c0_files = sorted(name for name in hashes if "_c0." in name)
        self.assertGreaterEqual(len(c0_files), 18)
        for c0_name in c0_files:
            with self.subTest(mesh=c0_name):
                c1_name = c0_name.replace("_c0.", "_c1.")
                self.assertIn(c1_name, hashes)
                self.assertEqual(hashes[c0_name], hashes[c1_name])

    def test_collada_uses_default_namespace_required_by_fortress(self) -> None:
        dae_paths = sorted((MODEL_ROOT / "meshes").glob("*.dae"))
        self.assertEqual(len(dae_paths), 12)
        for path in dae_paths:
            with self.subTest(mesh=path.name):
                content = path.read_text(encoding="utf-8")
                self.assertIn(
                    '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"',
                    content,
                )
                self.assertNotIn("ns0:COLLADA", content)
                self.assertNotIn(b"\r", path.read_bytes())


class TestVisualSkinGeometry(unittest.TestCase):
    def test_no_imported_mesh_is_ever_used_for_collision(self) -> None:
        for group in ("dev", "train", "val_id", "test_id"):
            root = _root(generate_world(group, 0))
            for model in root.findall("world/model"):
                if not model.get("name", "").startswith("generated_"):
                    continue
                for collision in model.findall("link/collision"):
                    self.assertIsNone(collision.find("geometry/mesh"))
                    self.assertTrue(
                        collision.find("geometry/box") is not None
                        or collision.find("geometry/cylinder") is not None
                    )

    def test_wall_visual_partition_exactly_matches_collision_height(self) -> None:
        root = _root(generate_world("train", 0))
        for model in root.findall("world/model"):
            wall = model.find("link/visual[@name='wall_visual']")
            if wall is None:
                continue
            skirting = model.find("link/visual[@name='skirting_visual']")
            self.assertIsNotNone(skirting)
            collision_size = _numbers(model.findtext("link/collision/geometry/box/size"))
            wall_size = _numbers(wall.findtext("geometry/mesh/scale"))
            skirting_size = _numbers(skirting.findtext("geometry/box/size"))
            wall_z = _numbers(wall.findtext("pose"))[2]
            skirting_z = _numbers(skirting.findtext("pose"))[2]
            self.assertEqual(wall_size[:2], collision_size[:2])
            self.assertEqual(skirting_size[:2], collision_size[:2])
            intervals = sorted(
                [
                    (wall_z - wall_size[2] / 2, wall_z + wall_size[2] / 2),
                    (
                        skirting_z - skirting_size[2] / 2,
                        skirting_z + skirting_size[2] / 2,
                    ),
                ]
            )
            self.assertAlmostEqual(intervals[0][0], -collision_size[2] / 2)
            self.assertAlmostEqual(intervals[0][1], intervals[1][0])
            self.assertAlmostEqual(intervals[1][1], collision_size[2] / 2)

    def test_visual_mesh_envelopes_stay_inside_authoritative_collision(self) -> None:
        contract = load_visual_skin_contract()
        for archetype_index in range(6):
            root = _root(generate_world("train", archetype_index))
            for model in root.findall("world/model"):
                for role in ("furniture", "cylinder"):
                    visual = model.find(f"link/visual[@name='{role}_visual']")
                    if visual is None:
                        continue
                    envelope = contract["mesh_envelopes_m"][role]
                    minimum = envelope["min_xyz"]
                    maximum = envelope["max_xyz"]
                    pose = _numbers(visual.findtext("pose"))
                    scale = _numbers(visual.findtext("geometry/mesh/scale"))
                    low = [pose[i] + minimum[i] * scale[i] for i in range(3)]
                    high = [pose[i] + maximum[i] * scale[i] for i in range(3)]
                    if role == "furniture":
                        size = _numbers(model.findtext("link/collision/geometry/box/size"))
                        for axis in range(3):
                            self.assertGreaterEqual(low[axis], -size[axis] / 2 - 1e-6)
                            self.assertLessEqual(high[axis], size[axis] / 2 + 1e-6)
                    else:
                        radius = float(
                            model.findtext("link/collision/geometry/cylinder/radius")
                        )
                        height = float(
                            model.findtext("link/collision/geometry/cylinder/length")
                        )
                        scaled_radius = envelope["radial_extent_m"] * scale[0]
                        self.assertLessEqual(scaled_radius, radius + 1e-6)
                        self.assertGreaterEqual(low[2], -height / 2 - 1e-6)
                        self.assertLessEqual(high[2], height / 2 + 1e-6)

    def test_c1_changes_texture_materials_without_changing_mesh_geometry(self) -> None:
        c0 = _root(generate_world("test_id", 0))
        c1 = _root(derive_condition(generate_world("test_id", 0), "C1"))
        c0_models = {
            model.get("name"): model
            for model in c0.findall("world/model")
            if model.get("name", "").startswith("generated_")
        }
        c1_models = {
            model.get("name"): model
            for model in c1.findall("world/model")
            if model.get("name", "").startswith("generated_")
        }
        self.assertEqual(c0_models.keys(), c1_models.keys())
        for name in c0_models:
            before = c0_models[name]
            after = c1_models[name]
            self.assertEqual(
                [
                    ET.tostring(node, encoding="unicode").replace("_c1.", "_c0.")
                    for node in before.findall("link/visual/geometry")
                ],
                [
                    ET.tostring(node, encoding="unicode").replace("_c1.", "_c0.")
                    for node in after.findall("link/visual/geometry")
                ],
            )
            before_uris = before.findall("link/visual/geometry/mesh/uri")
            after_uris = after.findall("link/visual/geometry/mesh/uri")
            self.assertEqual(len(before_uris), len(after_uris))
            for source, shifted in zip(before_uris, after_uris, strict=True):
                self.assertEqual(shifted.text, source.text.replace("_c0.", "_c1."))


if __name__ == "__main__":
    unittest.main()
