from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import livifuser_nav.backbone_handoff as handoff
from livifuser_nav.backbone_handoff import (
    BUNDLE_ROOT,
    COMPLETE_NAME,
    MANIFEST_NAME,
    MODEL_REVISION,
    ZIP_TIMESTAMP,
    seal_snapshot,
    sha256_file,
    verify_bundle,
)


def _snapshot(root: Path) -> tuple[Path, dict[str, dict[str, object]]]:
    files = {
        "LICENSE.md": b"license\n",
        "README.md": b"readme\n",
        "config.json": json.dumps(
            {
                "patch_size": 16,
                "hidden_size": 384,
                "num_register_tokens": 4,
                "use_gated_mlp": True,
            }
        ).encode(),
        "model.safetensors": b"weights",
        "preprocessor_config.json": b"{}\n",
    }
    expected = {}
    for name, payload in files.items():
        expected[name] = {
            "size_bytes": len(payload),
            "sha256": __import__("hashlib").sha256(payload).hexdigest().upper(),
        }
    root.mkdir()
    for name, payload in files.items():
        (root / name).write_bytes(payload)
    return root, expected


class BackboneHandoffTests(unittest.TestCase):
    def test_seal_and_verify_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            snapshot, expected = _snapshot(root / "snapshot")
            first = root / "first.zip"
            second = root / "second.zip"
            with patch.object(handoff, "EXPECTED_MODEL_FILES", expected):
                first_report = seal_snapshot(snapshot, first)
                second_report = seal_snapshot(snapshot, second)
                self.assertEqual(first_report["model_revision"], MODEL_REVISION)
                self.assertEqual(first_report["bundle_sha256"], second_report["bundle_sha256"])
                self.assertEqual(sha256_file(first), sha256_file(second))
                self.assertEqual(verify_bundle(first)["status"], "verified")
                with zipfile.ZipFile(first) as archive:
                    self.assertEqual(
                        set(archive.namelist()),
                        {
                            *(f"{BUNDLE_ROOT}/{name}" for name in expected),
                            f"{BUNDLE_ROOT}/{MANIFEST_NAME}",
                            f"{BUNDLE_ROOT}/{COMPLETE_NAME}",
                        },
                    )
                    self.assertTrue(
                        all(info.date_time == ZIP_TIMESTAMP for info in archive.infolist())
                    )

    def test_sealer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            snapshot, expected = _snapshot(root / "snapshot")
            output = root / "bundle.zip"
            with patch.object(handoff, "EXPECTED_MODEL_FILES", expected):
                seal_snapshot(snapshot, output)
                with self.assertRaises(FileExistsError):
                    seal_snapshot(snapshot, output)

    def test_verifier_rejects_member_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            snapshot, expected = _snapshot(root / "snapshot")
            valid = root / "valid.zip"
            bad = root / "bad.zip"
            with patch.object(handoff, "EXPECTED_MODEL_FILES", expected):
                seal_snapshot(snapshot, valid)
                with zipfile.ZipFile(valid) as source, zipfile.ZipFile(bad, "w") as target:
                    for item in source.infolist():
                        target.writestr(item, source.read(item.filename))
                    target.writestr("EXTRA", b"bad")
                with self.assertRaisesRegex(ValueError, "member set drifted"):
                    verify_bundle(bad)


if __name__ == "__main__":
    unittest.main()
