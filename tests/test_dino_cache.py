"""Contract tests for legacy and official frozen DINO feature caches."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from livifuser_nav.dino_cache import (
    CACHE_SCHEMA_VERSION,
    OFFICIAL_BACKBONE_CONTRACT_SHA256,
    OFFICIAL_CACHE_NAME,
    OFFICIAL_CACHE_SCHEMA_VERSION,
    OFFICIAL_HELDOUT_CACHE_NAME,
    OFFICIAL_PREPROCESSING_ID,
    TEMPORARY_BACKBONE_LABEL,
    DINOFeatureCache,
)
from livifuser_nav.learning_data import ExportRun, sha256_file
from tests.test_learning_data import write_export


def self_hash(value: dict, field: str) -> str:
    payload = copy.deepcopy(value)
    payload.pop(field, None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest().upper()


def write_arrays(root: Path, count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "patch_tokens_7x7_float16.npy", np.zeros((count, 49, 384), np.float16))
    np.save(root / "pooled_features_float32.npy", np.zeros((count, 384), np.float32))


def write_official_cache(
    root: Path,
    run: ExportRun,
    *,
    cache_name: str = OFFICIAL_CACHE_NAME,
) -> dict:
    write_arrays(root, run.count)
    names = ("patch_tokens_7x7_float16.npy", "pooled_features_float32.npy")
    manifest = {
        "schema_version": OFFICIAL_CACHE_SCHEMA_VERSION,
        "cache_name": cache_name,
        "episode_id": run.run_id,
        "row_count": run.count,
        "source": {
            "export_manifest_sha256": sha256_file(run.root / "manifest.json"),
        },
        "backbone_contract_sha256": OFFICIAL_BACKBONE_CONTRACT_SHA256,
        "preprocessing_id": OFFICIAL_PREPROCESSING_ID,
        "features": {
            names[0]: {"shape": [run.count, 49, 384], "dtype": "float16"},
            names[1]: {"shape": [run.count, 384], "dtype": "float32"},
        },
        "outputs": {
            name: {
                "sha256": sha256_file(root / name),
                "size_bytes": (root / name).stat().st_size,
            }
            for name in names
        },
    }
    field = "manifest_sha256_excludes_self"
    manifest[field] = self_hash(manifest, field)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (root / "SUCCESS.json").write_text(
        json.dumps({"manifest_sha256": sha256_file(manifest_path)}, indent=2) + "\n"
    )
    return manifest


class DINOFeatureCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        export = write_export(self.root / "export", segment_lengths=[2])
        self.run = ExportRun(export)
        self.addCleanup(self._close_run)

    def _close_run(self) -> None:
        self.run.rgb._mmap.close()
        self.run.scan_ranges._mmap.close()
        self.run.vectors.close()

    def test_loads_verified_official_splus_cache(self) -> None:
        cache_root = self.root / "official"
        write_official_cache(cache_root, self.run)
        cache = DINOFeatureCache(cache_root, self.run)
        self.assertEqual(cache.cache_identity["status"], "locked_final_backbone")
        self.assertEqual(cache.patch_tokens.shape, (2, 49, 384))
        self.assertEqual(cache.pooled_features.shape, (2, 384))

    def test_loads_verified_official_heldout_splus_cache(self) -> None:
        cache_root = self.root / "official-heldout"
        write_official_cache(
            cache_root,
            self.run,
            cache_name=OFFICIAL_HELDOUT_CACHE_NAME,
        )
        cache = DINOFeatureCache(cache_root, self.run)
        self.assertEqual(cache.cache_identity["status"], "locked_final_backbone")
        self.assertEqual(cache.cache_identity["cache_name"], OFFICIAL_CACHE_NAME)

    def test_official_cache_rejects_unknown_container_name(self) -> None:
        cache_root = self.root / "unknown-name"
        write_official_cache(
            cache_root,
            self.run,
            cache_name="unapproved_official_cache",
        )
        with self.assertRaisesRegex(ValueError, "cache name"):
            DINOFeatureCache(cache_root, self.run)

    def test_official_cache_rejects_backbone_contract_drift(self) -> None:
        cache_root = self.root / "wrong-contract"
        manifest = write_official_cache(cache_root, self.run)
        manifest["backbone_contract_sha256"] = "0" * 64
        field = "manifest_sha256_excludes_self"
        manifest[field] = self_hash(manifest, field)
        manifest_path = cache_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (cache_root / "SUCCESS.json").write_text(
            json.dumps({"manifest_sha256": sha256_file(manifest_path)})
        )
        with self.assertRaisesRegex(ValueError, "backbone contract"):
            DINOFeatureCache(cache_root, self.run)

    def test_official_cache_rejects_output_hash_drift(self) -> None:
        cache_root = self.root / "wrong-output"
        write_official_cache(cache_root, self.run)
        pooled = np.load(cache_root / "pooled_features_float32.npy")
        pooled[0, 0] = 1.0
        np.save(cache_root / "pooled_features_float32.npy", pooled)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            DINOFeatureCache(cache_root, self.run)

    def test_legacy_cache_remains_supported_and_explicitly_temporary(self) -> None:
        cache_root = self.root / "legacy"
        write_arrays(cache_root, self.run.count)
        (cache_root / "manifest.json").write_text(
            json.dumps(
                {
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
                    "run_id": self.run.run_id,
                    "row_count": self.run.count,
                    "backbone": {"label": TEMPORARY_BACKBONE_LABEL},
                }
            )
        )
        cache = DINOFeatureCache(cache_root, self.run)
        self.assertEqual(cache.cache_identity["status"], "temporary_deviation")


if __name__ == "__main__":
    unittest.main()
