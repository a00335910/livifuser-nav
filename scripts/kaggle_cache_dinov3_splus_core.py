"""Self-contained Kaggle helpers for the frozen DINOv3 S+/16 cache notebook."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import platform
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import PIL
import torch
from huggingface_hub import HfApi, snapshot_download
from PIL import Image
from transformers import AutoModel

EXPECTED_HANDOFF_SHA256 = "AB24252411EEF448BC0D853B0C9147AF184F0A1CC14D72BA39876BF179A92C6F"
EXPECTED_HANDOFF_NAME = "livifuser_confirmatory_v3_train_val_v1"
MODEL_ID = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
MODEL_REVISION = "c93d816fc9e567563bc068f01475bec89cc634a6"
MODEL_SAFETENSORS_SHA256 = "208146E499DACE99E4C9376DDB8A26F77D64C31C46C4DC4B86FF8BC63B0235E2"
MODEL_SAFETENSORS_SIZE = 114_794_096
CACHE_SCHEMA_VERSION = "2.1.0"
CACHE_NAME = "livifuser_dinov3_vits16plus_train_val_cache_v2"
PREPROCESSING_ID = "full_fov_letterbox_320x240_to_224x224_imagenet_pillow_bicubic_v1"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
OUTPUT_NAMES = (
    "patch_tokens_7x7_float16.npy",
    "pooled_features_float32.npy",
)
RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_stream(handle: Any, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def self_hash(value: dict[str, Any], field: str) -> str:
    payload = copy.deepcopy(value)
    payload.pop(field, None)
    return sha256_bytes(canonical_bytes(payload))


def load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def write_once_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        require(path.is_file() and path.read_bytes() == payload, f"refusing drift at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def safe_remove_tree(path: Path, required_parent: Path) -> None:
    resolved = path.resolve()
    parent = required_parent.resolve()
    require(resolved != parent and resolved.parent == parent, f"unsafe scratch path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def safe_remove_file(path: Path, required_parent: Path) -> None:
    resolved = path.resolve()
    parent = required_parent.resolve()
    require(resolved.parent == parent, f"unsafe temporary file path: {resolved}")
    if resolved.exists():
        require(resolved.is_file(), f"temporary path is not a file: {resolved}")
        resolved.unlink()


def sidecar_hash(path: Path) -> str:
    parts = path.read_text(encoding="utf-8").strip().split()
    require(len(parts) >= 1 and len(parts[0]) == 64, f"invalid checksum sidecar: {path}")
    return parts[0].upper()


def discover_handoff(input_root: Path) -> dict[str, Any]:
    candidates = []
    for path in input_root.rglob("handoff_manifest.json"):
        try:
            value = load_json_bytes(path.read_bytes(), str(path))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError):
            continue
        declared = value.get("manifest_sha256_excludes_self")
        if value.get("name") == EXPECTED_HANDOFF_NAME and declared == EXPECTED_HANDOFF_SHA256:
            require(
                self_hash(value, "manifest_sha256_excludes_self") == declared,
                "handoff self-hash mismatch",
            )
            candidates.append((path, value))
    require(len(candidates) == 1, f"expected one frozen handoff manifest, found {len(candidates)}")
    manifest_path, manifest = candidates[0]
    require(manifest["audit"]["episodes"] == 150, "handoff episode count drifted")
    require(manifest["audit"]["by_split"] == {"train": 120, "val_id": 30}, "split counts drifted")
    require(manifest["audit"]["accepted_samples"] == 69_253, "row count drifted")

    archives: dict[str, Path] = {}
    episode_sources: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        archive_matches = [path for path in input_root.rglob(shard["filename"]) if path.is_file()]
        extracted_name = Path(shard["filename"]).stem
        extracted_matches = [
            path
            for path in input_root.rglob(extracted_name)
            if path.is_dir() and (path / "SHARD_MANIFEST.json").is_file()
        ]
        require(
            len(archive_matches) + len(extracted_matches) == 1,
            f"expected one archive or extracted shard for {shard['filename']}, "
            f"found {len(archive_matches)} archives and {len(extracted_matches)} directories",
        )
        source_path = (archive_matches or extracted_matches)[0]
        source_layout = "zip" if archive_matches else "kaggle_extracted_directory"
        sidecars = list(input_root.rglob(shard["filename"] + ".sha256"))
        require(len(sidecars) == 1, f"missing checksum sidecar for {shard['filename']}")
        require(
            sidecar_hash(sidecars[0]) == shard["sha256"],
            f"sidecar mismatch for {shard['filename']}",
        )
        if source_layout == "zip":
            require(
                sha256_file(source_path) == shard["sha256"],
                f"archive hash mismatch: {source_path}",
            )
            with zipfile.ZipFile(source_path) as archive:
                raw = archive.read("SHARD_MANIFEST.json")
        else:
            raw = (source_path / "SHARD_MANIFEST.json").read_bytes()
        shard_manifest = load_json_bytes(raw, f"{source_path}:SHARD_MANIFEST.json")
        declared = shard_manifest.get("manifest_sha256_excludes_self")
        require(
            self_hash(shard_manifest, "manifest_sha256_excludes_self") == declared,
            "shard self-hash mismatch",
        )
        require(
            shard_manifest["episode_count"] == shard["episode_count"],
            "shard count mismatch",
        )
        require(shard_manifest["world_name"] == shard["world_name"], "shard world mismatch")
        if source_layout == "kaggle_extracted_directory":
            expected_members = {
                "SHARD_MANIFEST.json",
                *(member["path"] for member in shard_manifest["members"]),
            }
            observed_members = {
                path.relative_to(source_path).as_posix()
                for path in source_path.rglob("*")
                if path.is_file()
            }
            require(
                observed_members == expected_members, f"extracted member set drifted: {source_path}"
            )
            for member in shard_manifest["members"]:
                member_path = source_path / member["path"]
                require(
                    member_path.stat().st_size == member["size_bytes"],
                    f"extracted member size drifted: {member_path}",
                )
        archives[shard["filename"]] = source_path
        for episode in shard_manifest["episodes"]:
            episode_sources.append(
                {
                    "archive": source_path,
                    "archive_sha256": shard["sha256"],
                    "source_layout": source_layout,
                    "shard": shard,
                    "episode": episode,
                }
            )

    episode_sources.sort(key=lambda item: int(item["episode"]["ordinal"]))
    require(
        [item["episode"]["ordinal"] for item in episode_sources] == list(range(150)),
        "episode ordinals drifted",
    )
    require(
        Counter(item["episode"]["split"] for item in episode_sources)
        == Counter(train=120, val_id=30),
        "episode split drifted",
    )
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "archives": archives,
        "episodes": episode_sources,
    }


def member_name(episode_id: str, filename: str) -> str:
    value = f"episodes/{episode_id}/export/{filename}"
    parts = PurePosixPath(value).parts
    require(".." not in parts and parts[0] == "episodes", "unsafe archive member")
    return value


@contextmanager
def open_source_member(source: dict[str, Any], name: str) -> Any:
    if source["source_layout"] == "kaggle_extracted_directory":
        with (source["archive"] / name).open("rb") as handle:
            yield handle
        return
    with zipfile.ZipFile(source["archive"]) as archive, archive.open(name) as handle:
        yield handle


def materialize_rgb(source: dict[str, Any], scratch_root: Path) -> tuple[Path, dict[str, Any]]:
    episode = source["episode"]
    episode_id = episode["episode_id"]
    scratch = scratch_root / episode_id
    safe_remove_tree(scratch, scratch_root)
    scratch.mkdir(parents=True)
    with open_source_member(source, member_name(episode_id, "manifest.json")) as handle:
        manifest_raw = handle.read()
    require(
        sha256_bytes(manifest_raw) == episode["export_manifest_sha256"],
        "source manifest hash mismatch",
    )
    export_manifest = load_json_bytes(manifest_raw, f"{episode_id}:manifest")
    require(export_manifest["run_id"] == episode_id, "source manifest run mismatch")
    require(
        export_manifest["outputs"]["rgb_320x240_rgb8.npy"]
        == episode["outputs"]["rgb_320x240_rgb8.npy"],
        "source RGB record mismatch",
    )
    rgb_record = episode["outputs"]["rgb_320x240_rgb8.npy"]
    rgb_path = scratch / "rgb_320x240_rgb8.npy"
    digest = hashlib.sha256()
    size = 0
    with (
        open_source_member(
            source, member_name(episode_id, "rgb_320x240_rgb8.npy")
        ) as source_handle,
        rgb_path.open("wb") as target,
    ):
        while chunk := source_handle.read(8 * 1024 * 1024):
            target.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        require(
            size == rgb_record["size_bytes"],
            f"{episode_id}: RGB size mismatch",
        )
        require(
            digest.hexdigest().upper() == rgb_record["sha256"],
            f"{episode_id}: RGB hash mismatch",
        )
    return rgb_path, export_manifest


def preprocess_rgb_batch(images: np.ndarray) -> torch.Tensor:
    require(images.ndim == 4 and images.shape[1:] == (240, 320, 3), "unexpected RGB batch shape")
    require(images.dtype == np.uint8, "RGB source must be uint8")
    canvas = np.zeros((len(images), 224, 224, 3), dtype=np.float32)
    for index, image in enumerate(images):
        resized = np.asarray(Image.fromarray(image).resize((224, 168), Image.Resampling.BICUBIC))
        canvas[index, 28:196] = (resized.astype(np.float32) / 255.0 - RGB_MEAN) / RGB_STD
    return torch.from_numpy(np.ascontiguousarray(canvas.transpose(0, 3, 1, 2)))


def load_frozen_backbone(
    token: str,
    model_cache: Path,
    output_root: Path,
) -> tuple[torch.nn.Module, str, dict[str, Any]]:
    require(torch.cuda.is_available(), "enable a Kaggle GPU accelerator before running")
    require(bool(token.strip()), "HF_TOKEN is empty")
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, token=token)
    require(info.sha == MODEL_REVISION, f"resolved model revision drifted: {info.sha}")
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            token=token,
            cache_dir=model_cache,
            allow_patterns=("*.json", "*.safetensors", "LICENSE.md", "README.md"),
        )
    )
    weights = snapshot / "model.safetensors"
    require(weights.is_file(), "official snapshot omitted model.safetensors")
    require(weights.stat().st_size == MODEL_SAFETENSORS_SIZE, "checkpoint size drifted")
    require(sha256_file(weights) == MODEL_SAFETENSORS_SHA256, "checkpoint SHA-256 drifted")

    model = AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        attn_implementation="eager",
    )
    config = model.config
    require(int(config.patch_size) == 16, "DINO patch size drifted")
    require(int(config.hidden_size) == 384, "DINO hidden size drifted")
    require(int(config.num_register_tokens) == 4, "DINO register-token count drifted")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    require(28_000_000 <= parameter_count <= 30_000_000, "checkpoint is not ViT-S+/16 sized")
    if hasattr(config, "use_gated_mlp"):
        require(bool(config.use_gated_mlp), "S+ checkpoint must use its gated MLP")
    model.requires_grad_(False)
    model.float()
    model.eval()
    model.cuda(0)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        inference_model: torch.nn.Module = torch.nn.DataParallel(
            model, device_ids=list(range(torch.cuda.device_count()))
        )
    else:
        inference_model = model

    model_files = {}
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        model_files[path.relative_to(snapshot).as_posix()] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    contract: dict[str, Any] = {
        "schema_version": "1.0.0",
        "architecture_requirement": "frozen DINOv3 ViT-S+/16 only",
        "backbone": {
            "repository": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_file": "model.safetensors",
            "weights_sha256": MODEL_SAFETENSORS_SHA256,
            "weights_size_bytes": MODEL_SAFETENSORS_SIZE,
            "parameter_count": parameter_count,
            "frozen": True,
            "patch_size": 16,
            "hidden_size": 384,
            "register_tokens": 4,
            "prefix_tokens_dropped": 5,
            "source_patch_grid": [14, 14],
        },
        "preprocessing": {
            "id": PREPROCESSING_ID,
            "source": "RGB uint8 320x240",
            "resize": "224x168 Pillow bicubic",
            "letterbox": {"output": [224, 224], "top": 28, "bottom": 28},
            "normalization": {
                "scale": "uint8 / 255",
                "mean": RGB_MEAN.tolist(),
                "std": RGB_STD.tolist(),
                "padding_value_after_normalization": 0.0,
            },
            "geometry_note": "Full camera FOV retained for calibrated LiDAR projection.",
        },
        "features": {
            "patch_source": "last_hidden_state tokens 5:201",
            "fixed_pool": "non-overlapping 2x2 spatial mean",
            "patch_shape_per_frame": [49, 384],
            "patch_dtype": "float16",
            "pooled_source": "pooler_output",
            "pooled_shape_per_frame": [384],
            "pooled_dtype": "float32",
        },
        "inference": {
            "compute_dtype": "float32",
            "autocast": "disabled after T4 smoke rejected non-finite FP16 features",
            "attention_implementation": "eager",
            "tf32": False,
            "eval_mode": True,
            "gradients": False,
            "multi_gpu": "torch.nn.DataParallel" if len(device_names) > 1 else "disabled",
            "cuda_devices": device_names,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "huggingface_hub": __import__("huggingface_hub").__version__,
        },
        "model_files": model_files,
    }
    contract["contract_sha256_excludes_self"] = self_hash(contract, "contract_sha256_excludes_self")
    contract_payload = json_bytes(contract)
    write_once_or_verify(output_root / "BACKBONE_CONTRACT.json", contract_payload)
    contract_file_sha256 = sha256_bytes(contract_payload)
    return inference_model, contract_file_sha256, contract


def forward_features(
    model: torch.nn.Module, pixel_values: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    pixel_values = pixel_values.to(device="cuda:0", dtype=torch.float32, non_blocking=True)
    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values, return_dict=False)
    hidden = outputs[0]
    pooled = outputs[1]
    require(
        hidden.ndim == 3 and hidden.shape[1:] == (201, 384),
        f"hidden shape drifted: {tuple(hidden.shape)}",
    )
    require(
        pooled.ndim == 2 and pooled.shape[1:] == (384,),
        f"pooled shape drifted: {tuple(pooled.shape)}",
    )
    batch = hidden.shape[0]
    spatial = hidden[:, 5:, :].reshape(batch, 14, 14, 384)
    spatial = spatial.reshape(batch, 7, 2, 7, 2, 384).mean(dim=(2, 4))
    patches = spatial.reshape(batch, 49, 384).to(dtype=torch.float16).cpu().numpy()
    pooled_array = pooled.to(dtype=torch.float32).cpu().numpy()
    require(
        np.isfinite(patches).all() and np.isfinite(pooled_array).all(), "non-finite DINO feature"
    )
    return patches, pooled_array


def smoke_backbone(
    handoff: dict[str, Any], model: torch.nn.Module, scratch_root: Path
) -> dict[str, Any]:
    source = handoff["episodes"][0]
    rgb_path, _manifest = materialize_rgb(source, scratch_root)
    rgb = None
    try:
        rgb = np.load(rgb_path, mmap_mode="r")
        require(rgb.shape[0] >= 2, "smoke episode has too few frames")
        pixels = preprocess_rgb_batch(np.asarray(rgb[:2]))
        patches, pooled = forward_features(model, pixels)
        return {
            "episode_id": source["episode"]["episode_id"],
            "input_shape": list(pixels.shape),
            "patch_shape": list(patches.shape),
            "pooled_shape": list(pooled.shape),
            "patch_mean": float(patches.astype(np.float32).mean()),
            "pooled_norm_mean": float(np.linalg.norm(pooled, axis=1).mean()),
        }
    finally:
        if rgb is not None:
            del rgb
        safe_remove_tree(scratch_root / source["episode"]["episode_id"], scratch_root)


def verify_cache_episode(
    root: Path,
    source: dict[str, Any],
    backbone_contract_sha256: str,
) -> dict[str, Any]:
    episode = source["episode"]
    success_path = root / "SUCCESS.json"
    manifest_path = root / "manifest.json"
    require(success_path.is_file() and manifest_path.is_file(), f"incomplete cache: {root}")
    success = load_json_bytes(success_path.read_bytes(), str(success_path))
    manifest_raw = manifest_path.read_bytes()
    manifest = load_json_bytes(manifest_raw, str(manifest_path))
    require(success["manifest_sha256"] == sha256_bytes(manifest_raw), "cache SUCCESS mismatch")
    require(
        manifest["manifest_sha256_excludes_self"]
        == self_hash(manifest, "manifest_sha256_excludes_self"),
        "cache manifest self-hash mismatch",
    )
    require(manifest["episode_id"] == episode["episode_id"], "cache episode mismatch")
    require(manifest["row_count"] == episode["accepted_samples"], "cache row count mismatch")
    require(
        manifest["source"]["export_manifest_sha256"] == episode["export_manifest_sha256"],
        "cache source manifest mismatch",
    )
    require(
        manifest["source"]["rgb_sha256"] == episode["outputs"]["rgb_320x240_rgb8.npy"]["sha256"],
        "cache source RGB mismatch",
    )
    require(
        manifest["backbone_contract_sha256"] == backbone_contract_sha256, "cache backbone mismatch"
    )
    for name in OUTPUT_NAMES:
        path = root / name
        expected = manifest["outputs"][name]
        require(path.stat().st_size == expected["size_bytes"], f"cache size mismatch: {path}")
        require(sha256_file(path) == expected["sha256"], f"cache hash mismatch: {path}")
    patches = np.load(root / OUTPUT_NAMES[0], mmap_mode="r")
    pooled = np.load(root / OUTPUT_NAMES[1], mmap_mode="r")
    require(patches.shape == (episode["accepted_samples"], 49, 384), "cached patch shape mismatch")
    require(patches.dtype == np.float16, "cached patch dtype mismatch")
    require(pooled.shape == (episode["accepted_samples"], 384), "cached pooled shape mismatch")
    require(pooled.dtype == np.float32, "cached pooled dtype mismatch")
    return {
        "episode_id": episode["episode_id"],
        "ordinal": episode["ordinal"],
        "split": episode["split"],
        "world_name": episode["world_name"],
        "row_count": episode["accepted_samples"],
        "manifest_sha256": sha256_bytes(manifest_raw),
        "outputs": manifest["outputs"],
    }


def cache_episode(
    source: dict[str, Any],
    model: torch.nn.Module,
    backbone_contract_sha256: str,
    work_world_root: Path,
    scratch_root: Path,
    batch_size: int,
) -> dict[str, Any]:
    require(batch_size > 0, "batch_size must be positive")
    episode = source["episode"]
    episode_id = episode["episode_id"]
    episodes_root = work_world_root / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    final_root = episodes_root / episode_id
    if final_root.exists():
        return verify_cache_episode(final_root, source, backbone_contract_sha256)

    partial_root = episodes_root / f"{episode_id}.partial"
    safe_remove_tree(partial_root, episodes_root)
    partial_root.mkdir(parents=True)
    rgb_path: Path | None = None
    rgb: np.ndarray | None = None
    patches: np.memmap | None = None
    pooled: np.memmap | None = None
    started = time.monotonic()
    try:
        rgb_path, export_manifest = materialize_rgb(source, scratch_root)
        rgb = np.load(rgb_path, mmap_mode="r")
        row_count = int(episode["accepted_samples"])
        require(rgb.shape == (row_count, 240, 320, 3), f"{episode_id}: RGB shape drifted")
        require(rgb.dtype == np.uint8, f"{episode_id}: RGB dtype drifted")

        patch_path = partial_root / OUTPUT_NAMES[0]
        pooled_path = partial_root / OUTPUT_NAMES[1]
        patches = np.lib.format.open_memmap(
            patch_path,
            mode="w+",
            dtype=np.float16,
            shape=(row_count, 49, 384),
        )
        pooled = np.lib.format.open_memmap(
            pooled_path,
            mode="w+",
            dtype=np.float32,
            shape=(row_count, 384),
        )
        for start in range(0, row_count, batch_size):
            stop = min(start + batch_size, row_count)
            pixels = preprocess_rgb_batch(np.asarray(rgb[start:stop]))
            batch_patches, batch_pooled = forward_features(model, pixels)
            patches[start:stop] = batch_patches
            pooled[start:stop] = batch_pooled
        patches.flush()
        pooled.flush()
        del patches, pooled
        patches = None
        pooled = None

        output_records = {
            name: {
                "sha256": sha256_file(partial_root / name),
                "size_bytes": (partial_root / name).stat().st_size,
            }
            for name in OUTPUT_NAMES
        }
        manifest: dict[str, Any] = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_name": CACHE_NAME,
            "episode_id": episode_id,
            "ordinal": int(episode["ordinal"]),
            "split": episode["split"],
            "world_name": episode["world_name"],
            "condition": episode["condition"],
            "row_count": row_count,
            "source": {
                "handoff_manifest_sha256": EXPECTED_HANDOFF_SHA256,
                "source_shard_filename": source["shard"]["filename"],
                "source_shard_sha256": source["archive_sha256"],
                "source_layout": source["source_layout"],
                "export_manifest_sha256": episode["export_manifest_sha256"],
                "rgb_sha256": episode["outputs"]["rgb_320x240_rgb8.npy"]["sha256"],
                "export_schema_version": export_manifest["export_schema_version"],
            },
            "backbone_contract_sha256": backbone_contract_sha256,
            "preprocessing_id": PREPROCESSING_ID,
            "features": {
                OUTPUT_NAMES[0]: {
                    "shape": [row_count, 49, 384],
                    "dtype": "float16",
                    "meaning": "2x2 mean-pooled final-layer spatial patch tokens",
                },
                OUTPUT_NAMES[1]: {
                    "shape": [row_count, 384],
                    "dtype": "float32",
                    "meaning": "DINOv3 pooler_output for OOD statistics",
                },
            },
            "outputs": output_records,
        }
        manifest["manifest_sha256_excludes_self"] = self_hash(
            manifest, "manifest_sha256_excludes_self"
        )
        manifest_payload = json_bytes(manifest)
        (partial_root / "manifest.json").write_bytes(manifest_payload)
        (partial_root / "SUCCESS.json").write_bytes(
            json_bytes({"manifest_sha256": sha256_bytes(manifest_payload)})
        )
        record = verify_cache_episode(
            partial_root,
            source,
            backbone_contract_sha256,
        )
        os.replace(partial_root, final_root)
        elapsed = time.monotonic() - started
        print(
            f"cached ordinal={episode['ordinal']:03d} {episode_id} "
            f"rows={row_count} elapsed={elapsed:.1f}s"
        )
        return record
    except BaseException:
        safe_remove_tree(partial_root, episodes_root)
        raise
    finally:
        if patches is not None:
            patches.flush()
        if pooled is not None:
            pooled.flush()
        del patches, pooled, rgb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        safe_remove_tree(scratch_root / episode_id, scratch_root)


def deterministic_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def cache_shard_filename(source_shard: dict[str, Any]) -> str:
    source_name = Path(source_shard["filename"])
    require(source_name.suffix == ".zip", "source shard is not a ZIP archive")
    return f"{source_name.stem}.dinov3_vits16plus_cache.zip"


def write_file_to_zip(archive: zipfile.ZipFile, member: str, path: Path) -> None:
    with path.open("rb") as source, archive.open(deterministic_zip_info(member), "w") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def verify_cache_shard(
    archive_path: Path,
    source_shard: dict[str, Any],
    sources: list[dict[str, Any]],
    backbone_contract_sha256: str,
) -> dict[str, Any]:
    sidecar = archive_path.with_name(archive_path.name + ".sha256")
    require(archive_path.is_file() and sidecar.is_file(), f"incomplete shard: {archive_path}")
    archive_sha256 = sha256_file(archive_path)
    require(sidecar_hash(sidecar) == archive_sha256, f"cache sidecar mismatch: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, f"corrupt cache shard: {archive_path}")
        raw = archive.read("CACHE_SHARD_MANIFEST.json")
        manifest = load_json_bytes(raw, f"{archive_path}:CACHE_SHARD_MANIFEST.json")
    require(
        manifest["manifest_sha256_excludes_self"]
        == self_hash(manifest, "manifest_sha256_excludes_self"),
        f"cache shard self-hash mismatch: {archive_path}",
    )
    require(manifest["source_shard"] == source_shard, "source shard identity drifted")
    require(
        manifest["backbone_contract_sha256"] == backbone_contract_sha256,
        "cache shard backbone drifted",
    )
    expected_ids = [source["episode"]["episode_id"] for source in sources]
    require(
        [episode["episode_id"] for episode in manifest["episodes"]] == expected_ids,
        "cache shard episode order drifted",
    )
    require(
        manifest["row_count"]
        == sum(int(source["episode"]["accepted_samples"]) for source in sources),
        "cache shard row count drifted",
    )
    return {
        "filename": archive_path.name,
        "sha256": archive_sha256,
        "size_bytes": archive_path.stat().st_size,
        "split": manifest["split"],
        "world_name": manifest["world_name"],
        "episode_count": manifest["episode_count"],
        "row_count": manifest["row_count"],
        "manifest_sha256": sha256_bytes(raw),
    }


def build_cache_shard(
    output_root: Path,
    work_world_root: Path,
    source_shard: dict[str, Any],
    sources: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
    backbone_contract_sha256: str,
) -> dict[str, Any]:
    shards_root = output_root / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    archive_path = shards_root / cache_shard_filename(source_shard)
    if archive_path.exists():
        sidecar = archive_path.with_name(archive_path.name + ".sha256")
        if not sidecar.exists():
            digest = sha256_file(archive_path)
            write_once_or_verify(sidecar, f"{digest}  {archive_path.name}\n".encode())
        return verify_cache_shard(
            archive_path,
            source_shard,
            sources,
            backbone_contract_sha256,
        )

    partial_path = archive_path.with_name(archive_path.name + ".partial")
    safe_remove_file(partial_path, shards_root)
    members: list[dict[str, Any]] = []
    for record in episode_records:
        episode_root = work_world_root / "episodes" / record["episode_id"]
        for name in ("manifest.json", "SUCCESS.json", *OUTPUT_NAMES):
            path = episode_root / name
            members.append(
                {
                    "name": f"episodes/{record['episode_id']}/{name}",
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )

    shard_manifest: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_name": CACHE_NAME,
        "handoff_manifest_sha256": EXPECTED_HANDOFF_SHA256,
        "backbone_contract_sha256": backbone_contract_sha256,
        "source_shard": source_shard,
        "split": source_shard["split"],
        "world_name": source_shard["world_name"],
        "episode_count": len(episode_records),
        "row_count": sum(record["row_count"] for record in episode_records),
        "episodes": episode_records,
        "members": members,
    }
    shard_manifest["manifest_sha256_excludes_self"] = self_hash(
        shard_manifest, "manifest_sha256_excludes_self"
    )
    with zipfile.ZipFile(partial_path, "w", allowZip64=True) as archive:
        archive.writestr(
            deterministic_zip_info("CACHE_SHARD_MANIFEST.json"),
            json_bytes(shard_manifest),
        )
        for member in members:
            source_path = work_world_root / member["name"]
            write_file_to_zip(archive, member["name"], source_path)
    os.replace(partial_path, archive_path)
    digest = sha256_file(archive_path)
    sidecar = archive_path.with_name(archive_path.name + ".sha256")
    write_once_or_verify(sidecar, f"{digest}  {archive_path.name}\n".encode())
    return verify_cache_shard(
        archive_path,
        source_shard,
        sources,
        backbone_contract_sha256,
    )


def run_cache(
    handoff: dict[str, Any],
    model: torch.nn.Module,
    backbone_contract_sha256: str,
    output_root: Path,
    work_root: Path,
    scratch_root: Path,
    batch_size: int = 64,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in handoff["episodes"]:
        grouped[source["shard"]["filename"]].append(source)

    shard_records = []
    for source_shard in handoff["manifest"]["shards"]:
        sources = grouped[source_shard["filename"]]
        archive_path = output_root / "shards" / cache_shard_filename(source_shard)
        if archive_path.exists():
            record = verify_cache_shard(
                archive_path,
                source_shard,
                sources,
                backbone_contract_sha256,
            )
            print(f"verified existing cache shard {archive_path.name}")
            shard_records.append(record)
            continue

        world_root = work_root / source_shard["world_name"]
        world_root.mkdir(parents=True, exist_ok=True)
        episode_records = []
        for source in sources:
            episode_records.append(
                cache_episode(
                    source,
                    model,
                    backbone_contract_sha256,
                    world_root,
                    scratch_root,
                    batch_size,
                )
            )
        record = build_cache_shard(
            output_root,
            world_root,
            source_shard,
            sources,
            episode_records,
            backbone_contract_sha256,
        )
        shard_records.append(record)
        safe_remove_tree(world_root, work_root)
        print(f"sealed and verified cache shard {record['filename']}")

    master: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "name": CACHE_NAME,
        "handoff_manifest_sha256": EXPECTED_HANDOFF_SHA256,
        "backbone_contract_sha256": backbone_contract_sha256,
        "model": {
            "repository": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_sha256": MODEL_SAFETENSORS_SHA256,
        },
        "preprocessing_id": PREPROCESSING_ID,
        "feature_contract": {
            "patch_tokens": {"shape_per_frame": [49, 384], "dtype": "float16"},
            "pooled_features": {"shape_per_frame": [384], "dtype": "float32"},
        },
        "audit": {
            "episodes": sum(record["episode_count"] for record in shard_records),
            "accepted_samples": sum(record["row_count"] for record in shard_records),
            "by_split": dict(
                sorted(
                    Counter(source["episode"]["split"] for source in handoff["episodes"]).items()
                )
            ),
        },
        "shards": shard_records,
    }
    require(master["audit"]["episodes"] == 150, "final cache episode count drifted")
    require(master["audit"]["accepted_samples"] == 69_253, "final cache row count drifted")
    require(master["audit"]["by_split"] == {"train": 120, "val_id": 30}, "final split drifted")
    master["manifest_sha256_excludes_self"] = self_hash(master, "manifest_sha256_excludes_self")
    master_payload = json_bytes(master)
    write_once_or_verify(output_root / "cache_manifest.json", master_payload)
    write_once_or_verify(
        output_root / "CACHE_COMPLETE.json",
        json_bytes({"cache_manifest_sha256": sha256_bytes(master_payload)}),
    )
    return master


def verify_cache_output(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "cache_manifest.json"
    complete_path = output_root / "CACHE_COMPLETE.json"
    contract_path = output_root / "BACKBONE_CONTRACT.json"
    require(manifest_path.is_file(), "cache_manifest.json is missing")
    require(complete_path.is_file(), "CACHE_COMPLETE.json is missing")
    require(contract_path.is_file(), "BACKBONE_CONTRACT.json is missing")
    manifest_raw = manifest_path.read_bytes()
    manifest = load_json_bytes(manifest_raw, str(manifest_path))
    complete = load_json_bytes(complete_path.read_bytes(), str(complete_path))
    require(
        complete["cache_manifest_sha256"] == sha256_bytes(manifest_raw),
        "final completion marker mismatch",
    )
    require(
        manifest["manifest_sha256_excludes_self"]
        == self_hash(manifest, "manifest_sha256_excludes_self"),
        "final cache self-hash mismatch",
    )
    require(manifest["handoff_manifest_sha256"] == EXPECTED_HANDOFF_SHA256, "wrong handoff")
    require(manifest["audit"]["episodes"] == 150, "wrong final episode count")
    require(manifest["audit"]["accepted_samples"] == 69_253, "wrong final row count")
    require(
        manifest["backbone_contract_sha256"] == sha256_file(contract_path),
        "final backbone contract mismatch",
    )
    total_bytes = 0
    for shard in manifest["shards"]:
        path = output_root / "shards" / shard["filename"]
        require(path.stat().st_size == shard["size_bytes"], f"size mismatch: {path}")
        require(sha256_file(path) == shard["sha256"], f"hash mismatch: {path}")
        require(
            sidecar_hash(path.with_name(path.name + ".sha256")) == shard["sha256"],
            f"sidecar mismatch: {path}",
        )
        with zipfile.ZipFile(path) as archive:
            require(archive.testzip() is None, f"corrupt archive: {path}")
        total_bytes += shard["size_bytes"]
    return {
        "valid": True,
        "cache_manifest_sha256": sha256_bytes(manifest_raw),
        "backbone_contract_sha256": sha256_file(contract_path),
        "episodes": manifest["audit"]["episodes"],
        "accepted_samples": manifest["audit"]["accepted_samples"],
        "shards": len(manifest["shards"]),
        "shard_bytes": total_bytes,
        "shard_gib": round(total_bytes / (1024**3), 3),
        "output_root": str(output_root),
    }
