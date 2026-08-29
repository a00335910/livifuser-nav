from __future__ import annotations

import importlib.util
import shutil
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "repair_confirmatory_exports.py"
TEST_ROOT = REPOSITORY_ROOT / ".test-tmp" / "test_repair_confirmatory_exports"
SPEC = importlib.util.spec_from_file_location("repair_confirmatory_exports", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestRepairConfirmatoryExports(unittest.TestCase):
    def setUp(self) -> None:
        if TEST_ROOT.exists():
            shutil.rmtree(TEST_ROOT)
        TEST_ROOT.mkdir(parents=True)
        for index, name in enumerate(MODULE.OUTPUT_NAMES):
            (TEST_ROOT / name).write_bytes(name.encode("utf-8") + bytes([index]))

    def tearDown(self) -> None:
        if TEST_ROOT.exists():
            shutil.rmtree(TEST_ROOT)

    def test_output_audit_accepts_exact_hashes_and_sizes(self) -> None:
        manifest = {"outputs": MODULE.actual_output_records(TEST_ROOT)}
        self.assertEqual(MODULE.output_mismatches(TEST_ROOT, manifest), [])

    def test_output_audit_reports_post_manifest_change(self) -> None:
        manifest = {"outputs": MODULE.actual_output_records(TEST_ROOT)}
        changed = TEST_ROOT / MODULE.OUTPUT_NAMES[0]
        changed.write_bytes(b"changed-but-valid-file")
        mismatches = MODULE.output_mismatches(TEST_ROOT, manifest)
        self.assertEqual([item["file"] for item in mismatches], [changed.name])


if __name__ == "__main__":
    unittest.main()
