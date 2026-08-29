"""Guard the frozen simulation preregistration and implementation checksums."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "experiments"
    / "PREREGISTRATION_FREEZE_SIM_V1.json"
)
HISTORICAL_V2_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "experiments"
    / "PREREGISTRATION_RECOLLECTION_FREEZE_SIM_V2.json"
)
RECOLLECTION_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "experiments"
    / "PREREGISTRATION_RECOLLECTION_FREEZE_SIM_V3.json"
)


class TestSimulationPreregistrationFreeze(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = json.loads(BASE_MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.historical_v2 = json.loads(
            HISTORICAL_V2_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        cls.recollection = json.loads(
            RECOLLECTION_MANIFEST_PATH.read_text(encoding="utf-8")
        )

    def test_status_approval_and_counts_are_frozen(self):
        self.assertEqual(self.base["status"], "frozen")
        self.assertEqual(
            self.base["approval"]["statement"],
            "I approve and freeze the preregistration.",
        )
        self.assertFalse(
            self.base["approval"]["independent_supervisor_countersignature"]
        )
        episodes = self.base["frozen_design"]["episodes"]
        self.assertEqual(episodes["total_confirmatory"], 260)
        self.assertEqual(
            episodes["train"]
            + episodes["val_id"]
            + episodes["test_id"]
            + episodes["test_ood"],
            episodes["total_confirmatory"],
        )

    def test_recollection_freeze_preserves_design_and_excludes_predecessor(self):
        self.assertEqual(self.recollection["status"], "frozen_pre_recollection")
        self.assertEqual(
            self.recollection["frozen_design"], self.base["frozen_design"]
        )
        self.assertTrue(
            self.recollection["collection_roots"]["predecessor_reuse_forbidden"]
        )
        self.assertEqual(
            self.recollection["collection_roots"]["invalidated_preserved"],
            [
                "artifacts/simulation/confirmatory_v1",
                "artifacts/simulation/confirmatory_v2",
            ],
        )
        self.assertEqual(
            self.recollection["collection_roots"]["replacement"],
            "artifacts/simulation/confirmatory_v3",
        )
        self.assertEqual(
            [item["number"] for item in self.recollection["amendments"]],
            [1, 2, 3, 4],
        )

    def test_base_manifest_is_preserved_by_hash(self):
        expected = self.recollection["base_freeze_manifest"]["sha256"]
        actual = hashlib.sha256(BASE_MANIFEST_PATH.read_bytes()).hexdigest().upper()
        self.assertEqual(actual, expected)

    def test_failed_v2_freeze_is_preserved_by_hash(self):
        expected = self.recollection["historical_v2_manifest"]["sha256"]
        actual = hashlib.sha256(
            HISTORICAL_V2_MANIFEST_PATH.read_bytes()
        ).hexdigest().upper()
        self.assertEqual(actual, expected)
        self.assertEqual(self.historical_v2["status"], "frozen_pre_recollection")

    def test_every_recollection_frozen_file_matches_its_checksum(self):
        for relative, expected in self.recollection["sha256"].items():
            with self.subTest(path=relative):
                path = REPOSITORY_ROOT / relative
                actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
