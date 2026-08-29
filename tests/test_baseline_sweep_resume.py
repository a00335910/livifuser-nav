import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "run_baseline_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_baseline_sweep", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def result_payload(run, seed):
    return {
        "run": run,
        "seed": seed,
        "training": {"parameter_count": 10, "seconds": 1.25},
        "validation": {
            "macro_episode_nll": -0.5,
            "macro_episode_normalized_mse": 0.125,
        },
    }


class TestBaselineSweepResume(unittest.TestCase):
    def setUp(self):
        self.run = {"name": "full", "variant": "full", "loss": "heteroscedastic"}

    def test_new_or_empty_directory_is_not_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            self.assertIsNone(MODULE.completed_result(root, self.run, 7))
            root.mkdir()
            self.assertIsNone(MODULE.completed_result(root, self.run, 7))

    def test_result_and_checkpoint_are_a_completion_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoint.pt").write_bytes(b"checkpoint")
            (root / "result.json").write_text(
                json.dumps(result_payload(self.run, 7)), encoding="utf-8"
            )
            result = MODULE.completed_result(root, self.run, 7)
            self.assertEqual(MODULE.summary_row(result)["seed"], 7)

    def test_partial_or_wrong_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoint.pt").write_bytes(b"checkpoint")
            with self.assertRaisesRegex(RuntimeError, "partial"):
                MODULE.completed_result(root, self.run, 7)
            (root / "result.json").write_text(
                json.dumps(result_payload(self.run, 8)), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "identity"):
                MODULE.completed_result(root, self.run, 7)

    def test_run_context_must_match_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "context.json"
            MODULE.ensure_run_context(path, {"revision": "abc", "fold": 1})
            MODULE.ensure_run_context(path, {"revision": "abc", "fold": 1})
            with self.assertRaisesRegex(RuntimeError, "context differs"):
                MODULE.ensure_run_context(path, {"revision": "def", "fold": 1})

    def test_proposed_execution_config_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "not frozen"):
            MODULE.require_frozen_execution(
                {"execution_freeze": {"status": "proposed_pending_explicit_user_approval"}}
            )

    def test_frozen_and_legacy_configs_are_accepted(self):
        MODULE.require_frozen_execution({"execution_freeze": {"status": "frozen"}})
        MODULE.require_frozen_execution({"sweep_id": "legacy_pilot"})

    def test_run_partitions_preserve_requested_order_and_reject_drift(self):
        config = {
            "runs": [
                {"name": "full", "variant": "full"},
                {"name": "concat", "variant": "concat"},
            ]
        }
        selected = MODULE.selected_runs(config, ["concat", "full"])
        self.assertEqual([run["name"] for run in selected], ["concat", "full"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            MODULE.selected_runs(config, ["missing"])
        with self.assertRaisesRegex(ValueError, "unique"):
            MODULE.selected_runs(config, ["full", "full"])

    def test_cpu_device_moves_every_training_tensor(self):
        device = MODULE.resolve_device("cpu")
        arrays = {
            key: MODULE.np.zeros((2, 1), dtype=MODULE.np.float32)
            for key in MODULE.TENSOR_KEYS
        }
        arrays["target"] = MODULE.np.zeros((2, 1), dtype=MODULE.np.float32)
        tensors, target = MODULE.batch_tensors(arrays, device)
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in tensors.values()))
        self.assertEqual(target.device.type, "cpu")
        self.assertEqual(MODULE.device_provenance(device)["requested"], "cpu")

    def test_cuda_request_fails_closed_when_unavailable(self):
        original = MODULE.torch.cuda.is_available
        MODULE.torch.cuda.is_available = lambda: False
        try:
            with self.assertRaisesRegex(RuntimeError, "CUDA.*unavailable"):
                MODULE.resolve_device("cuda:0")
        finally:
            MODULE.torch.cuda.is_available = original


if __name__ == "__main__":
    unittest.main()
