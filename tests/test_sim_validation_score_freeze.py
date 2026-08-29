"""Tests for the frozen validation uncertainty-score execution contract."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPOSITORY_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPLAY = load_script("replay_sim_validation_scores")
ORCHESTRATOR = load_script("run_sim_validation_score_freeze_kaggle")
AUDITOR = load_script("audit_sim_validation_score_freeze")


class ScoreContractTests(unittest.TestCase):
    def test_notebook_accepts_kaggle_expanded_code_dataset(self) -> None:
        notebook = json.loads(
            (
                REPOSITORY_ROOT / "notebooks" / "kaggle_t4x2_sim_validation_score_freeze_v1.ipynb"
            ).read_text(encoding="utf-8")
        )
        source = "".join(notebook["cells"][1]["source"])
        manifest_discovery = "manifests = find_validation_manifests(INPUT)"
        archive_fallback = "archives = list(INPUT.rglob('livifuser_sim_validation_code_*.zip'))"
        self.assertIn(manifest_discovery, source)
        self.assertIn("frozen_amendment_sha256", source)
        self.assertIn("if not manifests:", source)
        self.assertIn(archive_fallback, source)
        self.assertLess(source.index(manifest_discovery), source.index(archive_fallback))

    def test_notebook_prefers_kaggle_expanded_result_directory(self) -> None:
        notebook = json.loads(
            (
                REPOSITORY_ROOT / "notebooks" / "kaggle_t4x2_sim_validation_score_freeze_v1.ipynb"
            ).read_text(encoding="utf-8")
        )
        source = "".join(notebook["cells"][3]["source"])
        expanded = "INPUT.rglob('livifuser_simulation_sweep_v1')"
        archive = "INPUT.rglob('livifuser_simulation_sweep_v1_results.zip')"
        self.assertIn(expanded, source)
        self.assertIn("validate_result_source", source)
        self.assertLess(source.index(expanded), source.index(archive))

    def test_expanded_result_source_verifies_exact_member_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / REPLAY.ROOT.rstrip("/")
            root.mkdir()
            member = REPLAY.ROOT + "summary.json"
            payload = b'{"result_count":24}\n'
            (root / "summary.json").write_bytes(payload)
            audit = {
                "archive": {"sha256": REPLAY.RESULT_ARCHIVE_SHA256},
                "member_sha256": {member: REPLAY.sha256_bytes(payload)},
            }
            result = REPLAY.validate_result_source(root, audit)
            self.assertEqual(result["mode"], "kaggle_expanded_directory")
            self.assertEqual(result["file_count"], 1)
            with REPLAY.ResultSource(root) as source:
                self.assertEqual(source.read(member), payload)

    def test_right_continuous_cdf_assigns_ties_the_upper_rank(self) -> None:
        reference = np.asarray([3.0, 1.0, 2.0, 1.0])
        observed = REPLAY.right_continuous_cdf(reference, np.asarray([0.0, 1.0, 2.0, 4.0]))
        np.testing.assert_array_equal(observed, [0.0, 0.5, 0.75, 1.0])

    def test_threshold_is_second_largest_and_strict(self) -> None:
        threshold, false_interventions = REPLAY.operating_threshold(np.arange(30))
        self.assertEqual(threshold, 28.0)
        self.assertEqual(false_interventions, 1)
        tied = np.concatenate((np.arange(28), np.asarray([29, 29])))
        threshold, false_interventions = REPLAY.operating_threshold(tied)
        self.assertEqual(threshold, 29.0)
        self.assertEqual(false_interventions, 0)

    def test_deterministic_npz_is_byte_reproducible_and_pickle_free(self) -> None:
        arrays = {
            "strings": np.asarray(["episode_1", "episode_2"], dtype=np.str_),
            "values": np.asarray([1.0, 2.0], dtype=np.float64),
        }
        first = REPLAY.deterministic_npz(arrays)
        second = REPLAY.deterministic_npz(dict(reversed(list(arrays.items()))))
        self.assertEqual(first, second)
        with np.load(io.BytesIO(first), allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["strings"], arrays["strings"])
            np.testing.assert_array_equal(archive["values"], arrays["values"])

    def test_singleton_npz_reconstruction_preserves_producer_compression(self) -> None:
        arrays = {
            "first": np.arange(16, dtype=np.float64),
            "second": np.asarray(["a", "b"], dtype=np.str_),
        }
        combined = REPLAY.deterministic_npz(arrays)
        for name, array in arrays.items():
            reconstructed = AUDITOR.singleton_npz_from_score(combined, name)
            self.assertEqual(reconstructed, REPLAY.deterministic_npz({name: array}))

    def test_partitions_freeze_exactly_21_heteroscedastic_scores(self) -> None:
        names = [
            name for partition in REPLAY.HETEROSCEDASTIC_PARTITIONS.values() for name in partition
        ]
        self.assertEqual(len(names), 7)
        self.assertEqual(len(set(names)), 7)
        self.assertNotIn("full_mean_only", names)
        self.assertEqual(len(names) * len(REPLAY.SEEDS), 21)
        self.assertEqual(len(REPLAY.CLOSED_LOOP_NAMES) * len(REPLAY.SEEDS), 12)

    def test_manifest_self_hash_excludes_only_its_own_field(self) -> None:
        value = {"schema_version": 1, "members": [{"name": "score.npz"}]}
        field = "manifest_sha256_excludes_self"
        value[field] = ORCHESTRATOR.self_hash(value, field)
        self.assertEqual(value[field], ORCHESTRATOR.self_hash(value, field))
        value["members"][0]["name"] = "drift.npz"
        self.assertNotEqual(value[field], ORCHESTRATOR.self_hash(value, field))


if __name__ == "__main__":
    unittest.main()
