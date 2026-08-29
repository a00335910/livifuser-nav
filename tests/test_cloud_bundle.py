import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from livifuser_nav.cloud_bundle import sha256_file, verify_cloud_bundle  # noqa: E402


class TestCloudBundle(unittest.TestCase):
    def create_bundle(self, root: Path):
        payload = root / "data" / "sample.bin"
        payload.parent.mkdir()
        payload.write_bytes(b"verified payload")
        manifest = {
            "schema_version": 1,
            "git_revision": "abc123",
            "file_count": 1,
            "total_bytes": payload.stat().st_size,
            "files": [
                {
                    "path": "data/sample.bin",
                    "size_bytes": payload.stat().st_size,
                    "sha256": sha256_file(payload),
                }
            ],
        }
        (root / "cloud_bundle_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return payload

    def test_every_manifested_file_is_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_bundle(root)
            result = verify_cloud_bundle(root)
            self.assertEqual(result["git_revision"], "abc123")
            self.assertEqual(result["file_count"], 1)

    def test_payload_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self.create_bundle(root)
            payload.write_bytes(b"tampered payload")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_cloud_bundle(root)

    def test_parent_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_bundle(root)
            path = root / "cloud_bundle_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["files"][0]["path"] = "../sample.bin"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe bundle path"):
                verify_cloud_bundle(root)
