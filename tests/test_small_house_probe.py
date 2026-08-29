import importlib.util
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_small_house_fortress_probe.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_small_house_probe", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestSmallHouseProbeAdapter(unittest.TestCase):
    def test_set_static_removes_inertial_blocks(self) -> None:
        model = ET.fromstring(
            "<model name='chair'><link name='body'><inertial><mass>1</mass>"
            "</inertial></link></model>"
        )

        MODULE._set_static(model)

        self.assertEqual(model.findtext("static"), "true")
        self.assertIsNone(model.find("link/inertial"))

    def test_flatten_legacy_include_preserves_identity_pose_and_static(self) -> None:
        world = ET.fromstring(
            "<world name='house'><model name='chair_7'>"
            "<pose>1 2 3 0 0 0.5</pose>"
            "<include><uri>model://SmallHouseChair</uri></include>"
            "</model></world>"
        )

        self.assertEqual(MODULE._flatten_legacy_model_includes(world), 1)

        include = world.find("include")
        self.assertIsNotNone(include)
        assert include is not None
        self.assertEqual(include.findtext("name"), "chair_7")
        self.assertEqual(include.findtext("uri"), "model://SmallHouseChair")
        self.assertEqual(include.findtext("pose"), "1 2 3 0 0 0.5")
        self.assertEqual(include.findtext("static"), "true")
        self.assertIsNone(world.find("model"))

    def test_filter_probe_includes_keeps_shell_and_nearby_assets(self) -> None:
        world = ET.fromstring(
            "<world name='house'>"
            "<include><name>near_chair</name><uri>model://chair</uri>"
            "<pose>4 1 0 0 0 0</pose></include>"
            "<include><name>far_chair</name><uri>model://chair</uri>"
            "<pose>20 20 0 0 0 0</pose></include>"
            "<include><name>HouseWallB_1</name><uri>model://wall</uri>"
            "<pose>20 20 0 0 0 0</pose></include>"
            "</world>"
        )

        kept, removed = MODULE._filter_probe_includes(world)

        self.assertEqual((kept, removed), (2, 1))
        self.assertEqual(
            [node.findtext("name") for node in world.findall("include")],
            ["near_chair", "HouseWallB_1"],
        )
