"""Frozen simulation training handoff and dual-GPU partition tests."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPOSITORY_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load_script("prepare_sim_training_data")
RUNNER = load_script("run_simulation_sweep_kaggle")


class FrozenConfigTests(unittest.TestCase):
    def test_notebook_redirects_bytecode_away_from_read_only_kaggle_input(self) -> None:
        notebook = json.loads(
            (REPOSITORY_ROOT / "notebooks" / "kaggle_t4x2_simulation_sweep_v1.ipynb").read_text()
        )
        source = "".join(notebook["cells"][1]["source"])
        self.assertIn("PYTHONPYCACHEPREFIX", source)
        self.assertIn("livifuser_pycache", source)
        self.assertIn("env=compile_env", source)

    def test_config_is_frozen_at_the_orchestrator_hash(self) -> None:
        path = REPOSITORY_ROOT / "config" / "simulation_sweep_v1.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(PREPARE.sha256_file(path), RUNNER.CONFIG_SHA256)
        self.assertEqual(config["execution_freeze"]["status"], "frozen")
        self.assertEqual(config["seeds"], [20260805, 20260806, 20260807])
        self.assertEqual(config["warmup_steps"], 24750)
        self.assertEqual(config["nll_steps"], 8250)
        self.assertEqual(len(config["runs"]) * len(config["seeds"]), 24)

    def test_gpu_partitions_cover_every_run_once(self) -> None:
        names = [name for partition in RUNNER.RUN_PARTITIONS for name in partition]
        self.assertEqual(len(names), 8)
        self.assertEqual(len(set(names)), 8)
        config = json.loads((REPOSITORY_ROOT / "config" / "simulation_sweep_v1.json").read_text())
        self.assertEqual(set(names), {run["name"] for run in config["runs"]})


class DataPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.handoff = json.loads(
            (
                REPOSITORY_ROOT
                / "artifacts"
                / "simulation"
                / "gpu_handoff_train_val_v1"
                / "handoff_manifest.json"
            ).read_text()
        )

    def test_real_handoff_builds_the_exact_split_plan(self) -> None:
        exports = {
            episode["episode_id"]: Path("exports") / episode["episode_id"]
            for episode in self.handoff["episodes"]
        }
        caches = {
            episode["episode_id"]: Path("caches") / episode["episode_id"]
            for episode in self.handoff["episodes"]
        }
        plan = PREPARE.build_plan(Path("work"), self.handoff, exports, caches)
        RUNNER.validate_data_plan(plan)
        self.assertEqual(plan["train"]["windows_k8_h8"], 41367)
        self.assertEqual(plan["validation"]["windows_k8_h8"], 9459)

    def test_real_handoff_builds_validation_only_plan(self) -> None:
        validation = [
            episode for episode in self.handoff["episodes"] if episode["split"] == "val_id"
        ]
        exports = {
            episode["episode_id"]: Path("exports") / episode["episode_id"] for episode in validation
        }
        caches = {
            episode["episode_id"]: Path("caches") / episode["episode_id"] for episode in validation
        }
        plan = PREPARE.build_validation_plan(Path("work"), self.handoff, exports, caches)
        self.assertNotIn("train", plan)
        self.assertEqual(plan["purpose"], "validation_score_freeze_only")
        self.assertFalse(plan["heldout_attached"])
        self.assertEqual(plan["validation"]["episode_count"], 30)
        self.assertEqual(plan["validation"]["accepted_samples"], 13125)
        self.assertEqual(plan["validation"]["windows_k8_h8"], 9459)

    def test_heldout_cache_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cache_manifest.json").write_text(
                json.dumps({"audit": {"by_split": {"test_id": 1}}})
            )
            with self.assertRaisesRegex(RuntimeError, "held-out"):
                PREPARE.refuse_heldout(root)

    def test_data_plan_rejects_a_heldout_episode(self) -> None:
        exports = {
            episode["episode_id"]: Path("exports") / episode["episode_id"]
            for episode in self.handoff["episodes"]
        }
        caches = {
            episode["episode_id"]: Path("caches") / episode["episode_id"]
            for episode in self.handoff["episodes"]
        }
        plan = PREPARE.build_plan(Path("work"), self.handoff, exports, caches)
        plan["validation"]["episode_ids"][0] = "test_id_forbidden"
        with self.assertRaisesRegex(ValueError, "held-out"):
            RUNNER.validate_data_plan(plan)


if __name__ == "__main__":
    unittest.main()
