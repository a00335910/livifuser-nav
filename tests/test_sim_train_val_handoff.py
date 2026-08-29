from __future__ import annotations

import importlib.util
import shutil
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "package_sim_train_val_handoff.py"
TEST_ROOT = REPOSITORY_ROOT / ".test-tmp" / "test_sim_train_val_handoff"
SPEC = importlib.util.spec_from_file_location("package_sim_train_val_handoff", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MODULE.json_bytes(value))


class TestTrainValHandoff(unittest.TestCase):
    def setUp(self) -> None:
        if TEST_ROOT.exists():
            shutil.rmtree(TEST_ROOT)
        TEST_ROOT.mkdir(parents=True)
        self.simulation = TEST_ROOT / "simulation"
        self.entries = [
            self.make_entry(0, "train", "train_world_000", 0),
            self.make_entry(1, "train", "train_world_000", 1),
            self.make_entry(2, "val_id", "val_world_000", 0),
        ]
        schedule = {
            "schema_version": "test",
            "recollection_freeze_manifest": "freeze.json",
            "recollection_freeze_manifest_sha256": "FREEZE",
            "episodes": self.entries,
        }
        schedule["schedule_sha256_excludes_self"] = MODULE.sha256_bytes(
            MODULE.canonical_bytes(schedule)
        )
        write_json(self.simulation / "schedule.json", schedule)
        for entry in self.entries:
            self.make_episode(entry)

    def tearDown(self) -> None:
        if TEST_ROOT.exists():
            shutil.rmtree(TEST_ROOT)

    @staticmethod
    def make_entry(ordinal: int, split: str, world: str, episode_index: int) -> dict:
        return {
            "ordinal": ordinal,
            "episode_id": f"{split}_{world}_e{episode_index:03d}",
            "entry_sha256": f"ENTRY{ordinal}",
            "split": split,
            "condition": "C0",
            "world_name": world,
            "world_index": 0,
            "world_seed": 100 + ordinal,
            "episode_index": episode_index,
            "observation_seed": 200 + ordinal,
        }

    def make_episode(self, entry: dict) -> None:
        root = self.simulation / "episodes" / entry["episode_id"]
        export = root / "attempt_001" / "export"
        export.mkdir(parents=True)
        outputs = {}
        for name, payload in {
            "rgb_320x240_rgb8.npy": b"rgb" + bytes([entry["ordinal"]]),
            "scan_ranges.npy": b"scan",
            "vectors.npz": b"vectors",
            "samples.jsonl": b"{}\n",
            "rejections.jsonl": b"",
        }.items():
            path = export / name
            path.write_bytes(payload)
            outputs[name] = {
                "sha256": MODULE.sha256_file(path),
                "size_bytes": len(payload),
            }
        manifest = {
            "export_schema_version": MODULE.EXPECTED_EXPORT_SCHEMA,
            "run_id": entry["episode_id"],
            "environment_id": entry["world_name"],
            "domain": "simulation",
            "view": "policy",
            "code": {"source_tree_sha256": "SOURCE"},
            "effective_configuration": {"lidar_causal": True},
            "effective_configuration_sha256": "CONFIG",
            "counts": {"accepted_samples": 2},
            "contiguity": {"windowable_k8_h8": 1},
            "outputs": outputs,
        }
        manifest["manifest_sha256_excludes_self"] = MODULE.manifest_self_hash(
            manifest, "manifest_sha256_excludes_self"
        )
        write_json(export / "manifest.json", manifest)
        attempt = {
            "status": "accepted",
            "return_code": 0,
            "entry_sha256": entry["entry_sha256"],
            "sha256": {"export/manifest.json": MODULE.sha256_file(export / "manifest.json")},
        }
        write_json(root / "attempt_001" / "ATTEMPT.json", attempt)
        success = {
            "status": "accepted",
            "episode_id": entry["episode_id"],
            "entry_sha256": entry["entry_sha256"],
            "accepted_attempt": "attempt_001",
            "attempt_manifest_sha256": MODULE.sha256_file(
                root / "attempt_001" / "ATTEMPT.json"
            ),
        }
        write_json(root / "SUCCESS.json", success)

    def audit(self):
        return MODULE.audit_handoff(
            self.simulation,
            expected_counts={"train": 2, "val_id": 1},
            expected_schedule_sha256=None,
        )

    def test_audit_binds_success_attempt_and_export(self) -> None:
        audit = self.audit()
        self.assertEqual(audit.summary["episodes"], 3)
        self.assertEqual(audit.summary["by_split"], {"train": 2, "val_id": 1})
        self.assertEqual(audit.summary["windowable_k8_h8"], 3)

    def test_audit_rejects_tampered_output(self) -> None:
        path = (
            self.simulation
            / "episodes"
            / self.entries[0]["episode_id"]
            / "attempt_001"
            / "export"
            / "samples.jsonl"
        )
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "(size|hash) mismatch"):
            self.audit()

    def test_world_shards_are_lossless_and_verifiable(self) -> None:
        audit = self.audit()
        output = TEST_ROOT / "handoff"
        records = [
            MODULE.build_shard(output, audit, world, episodes)
            for world, episodes in MODULE.shard_groups(audit).items()
        ]
        index = MODULE.write_handoff_index(output, audit, records)
        MODULE.verify_archives(output, index, deep=True)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["sha256"] for record in records))


if __name__ == "__main__":
    unittest.main()
