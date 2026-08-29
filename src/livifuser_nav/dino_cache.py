"""Frozen DINOv3 ONNX feature caching with explicit baseline provenance."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .learning_data import ExportRun, preprocess_rgb, sha256_file

CACHE_SCHEMA_VERSION = "1.0.0"
PREPROCESSING_ID = "full_fov_letterbox_320x240_to_224x224_imagenet_v1"
TEMPORARY_BACKBONE_LABEL = "frozen DINOv3 ViT-S/16 temporary baseline (not S+/16)"
OFFICIAL_CACHE_SCHEMA_VERSION = "2.1.0"
OFFICIAL_CACHE_NAME = "livifuser_dinov3_vits16plus_train_val_cache_v2"
OFFICIAL_HELDOUT_CACHE_NAME = "livifuser_dinov3_vits16plus_heldout_cache_v1"
OFFICIAL_CACHE_NAMES = frozenset((OFFICIAL_CACHE_NAME, OFFICIAL_HELDOUT_CACHE_NAME))
OFFICIAL_BACKBONE_CONTRACT_SHA256 = (
    "2957C78346DE608067DD5AC14D5C3E2F23438CD2BB3B1ECA847F898EBA68894A"
)
OFFICIAL_PREPROCESSING_ID = "full_fov_letterbox_320x240_to_224x224_imagenet_pillow_bicubic_v1"


def _manifest_self_hash(value: dict[str, Any], field: str) -> str:
    payload = deepcopy(value)
    payload.pop(field, None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest().upper()


def _latency_summary(values_ms: list[float]) -> dict[str, float]:
    ordered = sorted(values_ms)
    return {
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_ms": ordered[-1],
    }


def cache_dino_features(
    export_root: str | Path,
    output_root: str | Path,
    model_path: str | Path,
    *,
    expected_view: str = "policy",
) -> dict[str, object]:
    """Cache all accepted export rows, refusing to overwrite prior evidence."""

    # Imported here rather than at module level so that loading an existing
    # cache (`DINOFeatureCache`) works on hosts without onnxruntime; only
    # creating a cache runs the backbone.
    import onnxruntime as ort

    run = ExportRun(export_root, expected_view=expected_view)
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty cache {output}")
    output.mkdir(parents=True, exist_ok=True)
    model = Path(model_path).resolve()
    expected_hash = "C28A71B9AD81603CA4DCFB570E82C38E5CFEBE69EDAD1387686F9F7DD4A7E35A"
    model_hash = sha256_file(model)
    if model_hash != expected_hash:
        raise ValueError(f"temporary backbone hash mismatch: {model_hash}")

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = max(1, min(6, os.cpu_count() or 1))
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if [(item.name, item.shape) for item in session.get_inputs()] != [
        ("pixel_values", [1, 3, 224, 224])
    ]:
        raise ValueError("unexpected DINO input contract")

    patch_path = output / "patch_tokens_7x7_float16.npy"
    pooled_path = output / "pooled_features_float32.npy"
    patches = np.lib.format.open_memmap(
        patch_path, mode="w+", dtype=np.float16, shape=(run.count, 49, 384)
    )
    pooled = np.lib.format.open_memmap(
        pooled_path, mode="w+", dtype=np.float32, shape=(run.count, 384)
    )
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    for row in range(run.count):
        started = time.perf_counter()
        model_input = preprocess_rgb(np.asarray(run.rgb[row]))[None]
        preprocess_ms.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        hidden, pooled_output = session.run(None, {"pixel_values": model_input})
        inference_ms.append((time.perf_counter() - started) * 1000.0)
        if hidden.shape != (1, 201, 384) or pooled_output.shape != (1, 384):
            raise ValueError("unexpected DINO output contract")
        spatial = hidden[:, 5:, :].reshape(1, 14, 14, 384)
        spatial = spatial.reshape(1, 7, 2, 7, 2, 384).mean(axis=(2, 4))
        patches[row] = spatial.reshape(49, 384).astype(np.float16)
        pooled[row] = pooled_output[0]
    patches.flush()
    pooled.flush()
    del patches, pooled

    manifest: dict[str, object] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "run_id": run.run_id,
        "source_export": str(run.root),
        "source_export_view": expected_view,
        "source_export_manifest_sha256": sha256_file(run.root / "manifest.json"),
        "source_rgb_sha256": run.manifest["outputs"]["rgb_320x240_rgb8.npy"]["sha256"],
        "row_count": run.count,
        "backbone": {
            "label": TEMPORARY_BACKBONE_LABEL,
            "architecture": "DINOv3 ViT-S/16",
            "frozen": True,
            "model_path": str(model),
            "model_sha256": model_hash,
            "locked_final_deviation": (
                "Temporary pipeline baseline; final Architecture v1.1 requires frozen "
                "DINOv3 ViT-S+/16. This artifact must never be relabeled as S+/16."
            ),
        },
        "preprocessing": {
            "id": PREPROCESSING_ID,
            "source": "RGB uint8 320x240",
            "resize": "224x168 Pillow bicubic",
            "letterbox": {"output": [224, 224], "top": 28, "bottom": 28},
            "normalization": {
                "scale": "uint8 / 255",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "padding_value_after_normalization": 0.0,
            },
        },
        "tokens": {
            "source_shape": [201, 384],
            "dropped_prefix_tokens": 5,
            "source_patch_grid": [14, 14],
            "fixed_pool": "non-overlapping 2x2 mean",
            "cached_patch_shape": [run.count, 49, 384],
            "cached_patch_dtype": "float16",
            "cached_pooled_shape": [run.count, 384],
            "cached_pooled_dtype": "float32",
        },
        "timing": {
            "preprocess": _latency_summary(preprocess_ms),
            "onnx_backbone": _latency_summary(inference_ms),
            "sample_count": run.count,
            "host": platform.node(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "providers": session.get_providers(),
            "note": "Offline CPU cache timing; not capture-to-command deployment latency.",
        },
        "outputs": {
            patch_path.name: {
                "sha256": sha256_file(patch_path),
                "size_bytes": patch_path.stat().st_size,
            },
            pooled_path.name: {
                "sha256": sha256_file(pooled_path),
                "size_bytes": pooled_path.stat().st_size,
            },
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class DINOFeatureCache:
    def __init__(self, root: str | Path, expected_run: ExportRun) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text("utf-8"))
        schema = self.manifest.get("schema_version")
        if schema == OFFICIAL_CACHE_SCHEMA_VERSION:
            self._validate_official(manifest_path, expected_run)
            self.cache_identity = {
                "status": "locked_final_backbone",
                "architecture": "DINOv3 ViT-S+/16",
                "cache_schema_version": OFFICIAL_CACHE_SCHEMA_VERSION,
                "cache_name": OFFICIAL_CACHE_NAME,
                "backbone_contract_sha256": OFFICIAL_BACKBONE_CONTRACT_SHA256,
                "preprocessing_id": OFFICIAL_PREPROCESSING_ID,
            }
        else:
            self._validate_legacy(expected_run)
            self.cache_identity = {
                "status": "temporary_deviation",
                "architecture": "DINOv3 ViT-S/16",
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "label": TEMPORARY_BACKBONE_LABEL,
                "preprocessing_id": PREPROCESSING_ID,
            }

        patch_path = self.root / "patch_tokens_7x7_float16.npy"
        pooled_path = self.root / "pooled_features_float32.npy"
        self.patch_tokens = np.load(patch_path, mmap_mode="r")
        self.pooled_features = np.load(pooled_path, mmap_mode="r")
        if self.patch_tokens.shape != (expected_run.count, 49, 384):
            raise ValueError("unexpected cached patch-token shape")
        if self.patch_tokens.dtype != np.dtype("float16"):
            raise ValueError("unexpected cached patch-token dtype")
        if self.pooled_features.shape != (expected_run.count, 384):
            raise ValueError("unexpected cached pooled-feature shape")
        if self.pooled_features.dtype != np.dtype("float32"):
            raise ValueError("unexpected cached pooled-feature dtype")

    def _validate_legacy(self, expected_run: ExportRun) -> None:
        if self.manifest.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported feature cache schema")
        if self.manifest.get("run_id") != expected_run.run_id:
            raise ValueError("feature cache run_id does not match export")
        if int(self.manifest["row_count"]) != expected_run.count:
            raise ValueError("feature cache row count does not match export")
        if self.manifest["backbone"]["label"] != TEMPORARY_BACKBONE_LABEL:
            raise ValueError("feature cache backbone label is not the approved baseline")

    def _validate_official(self, manifest_path: Path, expected_run: ExportRun) -> None:
        manifest = self.manifest
        if manifest.get("cache_name") not in OFFICIAL_CACHE_NAMES:
            raise ValueError("official feature cache name mismatch")
        if manifest.get("episode_id") != expected_run.run_id:
            raise ValueError("feature cache episode_id does not match export")
        if int(manifest.get("row_count", -1)) != expected_run.count:
            raise ValueError("feature cache row count does not match export")
        if manifest.get("backbone_contract_sha256") != OFFICIAL_BACKBONE_CONTRACT_SHA256:
            raise ValueError("official feature cache backbone contract mismatch")
        if manifest.get("preprocessing_id") != OFFICIAL_PREPROCESSING_ID:
            raise ValueError("official feature cache preprocessing mismatch")

        field = "manifest_sha256_excludes_self"
        if manifest.get(field) != _manifest_self_hash(manifest, field):
            raise ValueError("official feature cache manifest self-hash mismatch")
        source = manifest.get("source", {})
        if source.get("export_manifest_sha256") != sha256_file(expected_run.root / "manifest.json"):
            raise ValueError("feature cache source export manifest mismatch")
        export_outputs = expected_run.manifest.get("outputs", {})
        rgb_record = export_outputs.get("rgb_320x240_rgb8.npy")
        if rgb_record is not None and source.get("rgb_sha256") != rgb_record.get("sha256"):
            raise ValueError("feature cache source RGB mismatch")

        success_path = self.root / "SUCCESS.json"
        success = json.loads(success_path.read_text("utf-8"))
        if success.get("manifest_sha256") != sha256_file(manifest_path):
            raise ValueError("official feature cache completion marker mismatch")

        expected_features = {
            "patch_tokens_7x7_float16.npy": {
                "shape": [expected_run.count, 49, 384],
                "dtype": "float16",
            },
            "pooled_features_float32.npy": {
                "shape": [expected_run.count, 384],
                "dtype": "float32",
            },
        }
        for name, expected in expected_features.items():
            feature = manifest.get("features", {}).get(name, {})
            if (
                feature.get("shape") != expected["shape"]
                or feature.get("dtype") != expected["dtype"]
            ):
                raise ValueError(f"official feature contract mismatch for {name}")
            path = self.root / name
            record = manifest.get("outputs", {}).get(name, {})
            if path.stat().st_size != int(record.get("size_bytes", -1)):
                raise ValueError(f"official feature cache size mismatch for {name}")
            if sha256_file(path) != record.get("sha256"):
                raise ValueError(f"official feature cache hash mismatch for {name}")
