from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "package_sim_heldout_handoff.py"
SPEC = importlib.util.spec_from_file_location("package_sim_heldout_handoff", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestHeldoutHandoff(unittest.TestCase):
    def test_shard_key_separates_split_world_and_condition(self) -> None:
        test_id = SimpleNamespace(
            record={
                "split": "test_id",
                "world_name": "test_id_straight_corridor_000",
                "condition": "C0",
            }
        )
        test_ood = SimpleNamespace(
            record={
                "split": "test_ood",
                "world_name": "test_id_straight_corridor_000",
                "condition": "C3",
            }
        )
        self.assertEqual(MODULE.shard_key(test_id), "test_id_straight_corridor_000_c0")
        self.assertEqual(MODULE.shard_key(test_ood), "test_ood_straight_corridor_000_c3")

    def test_manifest_self_hash_rejects_mutation(self) -> None:
        value = {"name": "heldout"}
        value["manifest_sha256_excludes_self"] = MODULE.manifest_self_hash(
            value, "manifest_sha256_excludes_self"
        )
        self.assertEqual(
            MODULE.manifest_self_hash(value, "manifest_sha256_excludes_self"),
            value["manifest_sha256_excludes_self"],
        )
        value["name"] = "changed"
        self.assertNotEqual(
            MODULE.manifest_self_hash(value, "manifest_sha256_excludes_self"),
            value["manifest_sha256_excludes_self"],
        )


if __name__ == "__main__":
    unittest.main()
