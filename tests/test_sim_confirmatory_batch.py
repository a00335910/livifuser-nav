import importlib.util
import json
import os
import shutil
import sys
import unittest
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "run_confirmatory_sim_batch.py"
TEST_TEMP_ROOT = REPOSITORY_ROOT / ".test-tmp"
TEST_WORK_ROOT = TEST_TEMP_ROOT / "test_sim_confirmatory_batch"
SPEC = importlib.util.spec_from_file_location("run_confirmatory_sim_batch", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestConfirmatoryBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_TEMP_ROOT.mkdir(exist_ok=True)
        cls.config = MODULE.load_json(MODULE.DEFAULT_CONFIG)
        cls.plan = MODULE.build_plan(cls.config)
        cls.entries = cls.plan.schedule["episodes"]

    def setUp(self):
        if TEST_WORK_ROOT.exists():
            shutil.rmtree(TEST_WORK_ROOT)
        TEST_WORK_ROOT.mkdir()

    def tearDown(self):
        if TEST_WORK_ROOT.exists():
            shutil.rmtree(TEST_WORK_ROOT)

    def test_schedule_matches_frozen_counts(self):
        self.assertEqual(self.plan.schedule["schema_version"], "3.0.0")
        self.assertEqual(
            self.config["output_root"], "artifacts/simulation/confirmatory_v3"
        )
        self.assertEqual(
            self.config["forbidden_predecessor_roots"],
            [
                "artifacts/simulation/confirmatory_v1",
                "artifacts/simulation/confirmatory_v2",
            ],
        )
        self.assertEqual(len(self.entries), 260)
        self.assertEqual(
            Counter(entry["split"] for entry in self.entries),
            Counter({"train": 120, "val_id": 30, "test_id": 30, "test_ood": 80}),
        )
        ood = [entry for entry in self.entries if entry["split"] == "test_ood"]
        self.assertEqual(
            Counter(entry["condition"] for entry in ood),
            Counter({"C0": 20, "C1": 20, "C3": 20, "C4": 20}),
        )

    def test_non_ood_collection_is_c0_only(self):
        non_ood = [
            entry for entry in self.entries if entry["split"] != "test_ood"
        ]
        self.assertEqual({entry["condition"] for entry in non_ood}, {"C0"})
        self.assertEqual({entry["world_variant"] for entry in non_ood}, {"C0"})
        self.assertEqual({entry["lidar_condition"] for entry in non_ood}, {"C0"})

    def test_condition_implementations_are_exact(self):
        ood = [entry for entry in self.entries if entry["split"] == "test_ood"]
        observed = {
            entry["condition"]: (
                entry["world_variant"],
                entry["lidar_condition"],
            )
            for entry in ood
        }
        self.assertEqual(
            observed,
            {
                "C0": ("C0", "C0"),
                "C1": ("C1", "C0"),
                "C3": ("C0", "C3b"),
                "C4": ("C4", "C0"),
            },
        )

    def test_ood_conditions_are_matched_on_world_episode_and_seed(self):
        paired = {}
        for entry in self.entries:
            if entry["split"] == "test_ood":
                key = (entry["world_index"], entry["episode_index"])
                paired.setdefault(key, []).append(entry)
        self.assertEqual(len(paired), 20)
        for values in paired.values():
            self.assertEqual({entry["condition"] for entry in values}, {"C0", "C1", "C3", "C4"})
            for field in (
                "source_world_group",
                "world_index",
                "world_name",
                "world_seed",
                "episode_index",
                "observation_seed",
            ):
                self.assertEqual(len({entry[field] for entry in values}), 1)

    def test_seed_blocks_do_not_overlap(self):
        by_split = {
            split: {
                entry["observation_seed"]
                for entry in self.entries
                if entry["split"] == split
            }
            for split in ("train", "val_id", "test_id", "test_ood")
        }
        names = list(by_split)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                self.assertFalse(by_split[left] & by_split[right])

    def test_recollection_requires_world_truth_and_all_three_amendments(self):
        self.assertIn("/livifuser/sim/ground_truth/odom", self.config["topics"])
        self.assertEqual(
            [record["number"] for record in self.config["amendments"]],
            [1, 2, 3, 4],
        )
        self.assertEqual(
            self.plan.schedule["invalidated_predecessor_roots"],
            [
                "artifacts/simulation/confirmatory_v1",
                "artifacts/simulation/confirmatory_v2",
            ],
        )

    def test_plan_is_deterministic(self):
        rebuilt = MODULE.build_plan(self.config)
        self.assertEqual(rebuilt.schedule, self.plan.schedule)
        self.assertEqual(rebuilt.assets, self.plan.assets)

    def test_prepare_is_idempotent_and_rejects_drift(self):
        first = MODULE.prepare(self.plan, TEST_WORK_ROOT)
        second = MODULE.prepare(self.plan, TEST_WORK_ROOT)
        artifact_count = len(self.plan.assets) + 1
        self.assertEqual(first, {"written": artifact_count})
        self.assertEqual(second, {"verified": artifact_count})
        MODULE.validate_prepared(self.plan, TEST_WORK_ROOT)
        asset = TEST_WORK_ROOT / next(iter(self.plan.assets))
        asset.write_bytes(b"drift")
        with self.assertRaisesRegex(ValueError, "drifted"):
            MODULE.validate_prepared(self.plan, TEST_WORK_ROOT)

    def test_success_marker_must_match_schedule_entry(self):
        entry = self.entries[0]
        marker = {
            "status": "accepted",
            "entry_sha256": "wrong",
        }
        (TEST_WORK_ROOT / "SUCCESS.json").write_text(
            json.dumps(marker), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "different schedule"):
            MODULE.accepted_success(entry, TEST_WORK_ROOT)

    @unittest.skipUnless(os.name == "nt", "Windows-to-WSL conversion is Windows-only")
    def test_windows_path_is_converted_for_wsl(self):
        expected = "/mnt/d/LiViFuser/scripts/run_confirmatory_sim_episode.sh"
        self.assertEqual(MODULE.to_wsl_path(MODULE.EPISODE_SCRIPT), expected)

    def test_simulation_cleanup_checks_process_groups_not_only_leaders(self):
        grouped = (
            "record_export_sim_dev_episode.sh",
            "run_confirmatory_sim_episode.sh",
            "run_nav2_sim_dev_episode.sh",
            "run_lidar_policy_sim_dev_episode.sh",
        )
        for name in grouped:
            content = (REPOSITORY_ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn('kill -0 -- "-$group"', content, name)
        isolated = (
            REPOSITORY_ROOT / "scripts" / "run_isolated_sim_dev_episode.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('kill -0 -- "-$launch_group"', isolated)


if __name__ == "__main__":
    unittest.main()
