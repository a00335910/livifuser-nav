"""Deterministic acquisition and sealing of the locked DINOv3 S+/16 snapshot."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

MODEL_ID = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
MODEL_REVISION = "c93d816fc9e567563bc068f01475bec89cc634a6"
BUNDLE_ROOT = "livifuser_dinov3_vits16plus_backbone_c93d816"
BUNDLE_FILENAME = f"{BUNDLE_ROOT}_bundle.zip"
MANIFEST_NAME = "BACKBONE_BUNDLE_MANIFEST.json"
COMPLETE_NAME = "BACKBONE_BUNDLE_COMPLETE.json"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

# This exact set was independently recorded by the accepted train/validation
# DINO cache contract. Do not discover or widen it from a future Hub snapshot.
EXPECTED_MODEL_FILES: dict[str, dict[str, Any]] = {
    "LICENSE.md": {
        "size_bytes": 7_503,
        "sha256": "25D122EB8F5B880FD23C736FB6EA8018EE45C12237E00B8A86D14C653904999E",
    },
    "README.md": {
        "size_bytes": 14_528,
        "sha256": "75CD3E334E64FECFA2507B4ECE416964D4BCAFCDED94D8B351D00679B95A5B5D",
    },
    "config.json": {
        "size_bytes": 742,
        "sha256": "6F4AC67FEA1761FE684D2A7DB3139BAB2D0DFDF94C05063D5992717C4C1DA0AC",
    },
    "model.safetensors": {
        "size_bytes": 114_794_096,
        "sha256": "208146E499DACE99E4C9376DDB8A26F77D64C31C46C4DC4B86FF8BC63B0235E2",
    },
    "preprocessor_config.json": {
        "size_bytes": 585,
        "sha256": "960C41D1F3A7778B936365769A2D90550B318A6C0A53A0296957ADACFE5E0DD7",
    },
}

ACCEPTED_CACHE_BACKBONE_CONTRACT_FILE_SHA256 = (
    "2957C78346DE608067DD5AC14D5C3E2F23438CD2BB3B1ECA847F898EBA68894A"
)
ACCEPTED_CACHE_BACKBONE_CONTRACT_SELF_SHA256 = (
    "DA76FCBC0A0309DB6F98C742924CF579827123217B6B9410D74F83FC6AC0D772"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )


def self_hash(value: dict[str, Any], field: str) -> str:
    copy = dict(value)
    copy.pop(field, None)
    return sha256_bytes(json_bytes(copy))


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits = 0x800
    return info


def _validate_config(payload: bytes) -> None:
    config = json.loads(payload)
    required = {
        "patch_size": 16,
        "hidden_size": 384,
        "num_register_tokens": 4,
    }
    for key, expected in required.items():
        if int(config.get(key, -1)) != expected:
            raise ValueError(f"official DINO config {key} drifted")
    if "use_gated_mlp" in config and not bool(config["use_gated_mlp"]):
        raise ValueError("official S+ config no longer enables its gated MLP")


def verify_snapshot(snapshot: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(snapshot).resolve()
    if not root.is_dir():
        raise ValueError(f"snapshot directory does not exist: {root}")
    observed: dict[str, dict[str, Any]] = {}
    for name, expected in EXPECTED_MODEL_FILES.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"official snapshot omitted {name}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != expected["size_bytes"]:
            raise ValueError(f"{name} size drifted: {size}")
        if digest != expected["sha256"]:
            raise ValueError(f"{name} SHA-256 drifted: {digest}")
        observed[name] = {"size_bytes": size, "sha256": digest}
    _validate_config((root / "config.json").read_bytes())
    return observed


def download_snapshot(*, token: str, cache_dir: str | Path) -> Path:
    """Download only the frozen file set from the exact gated Hub revision."""

    if not token.strip():
        raise ValueError("HF_TOKEN is empty")
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised in Kaggle, not unit tests
        raise RuntimeError("install huggingface_hub==0.34.4 for the download handoff") from exc

    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, token=token)
    if info.sha != MODEL_REVISION:
        raise ValueError(f"resolved model revision drifted: {info.sha}")
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            token=token,
            cache_dir=Path(cache_dir),
            allow_patterns=sorted(EXPECTED_MODEL_FILES),
        )
    )
    verify_snapshot(snapshot)
    return snapshot


def seal_snapshot(snapshot: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Create a fixed-topology, fixed-metadata, non-overwriting transport ZIP."""

    root = Path(snapshot).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite backbone bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = verify_snapshot(root)
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "sealed_official_backbone",
        "model": {
            "repository": MODEL_ID,
            "revision": MODEL_REVISION,
            "frozen": True,
            "architecture": "DINOv3 ViT-S+/16",
        },
        "accepted_cache_contract": {
            "file_sha256": ACCEPTED_CACHE_BACKBONE_CONTRACT_FILE_SHA256,
            "self_sha256": ACCEPTED_CACHE_BACKBONE_CONTRACT_SELF_SHA256,
        },
        "files": files,
        "member_count_including_manifest_and_completion": len(files) + 2,
        "zip_contract": {
            "root": BUNDLE_ROOT,
            "compression": "stored",
            "timestamp": "1980-01-01T00:00:00",
            "unix_mode": "100644",
        },
    }
    manifest["manifest_sha256_excludes_self"] = self_hash(
        manifest, "manifest_sha256_excludes_self"
    )
    manifest_payload = json_bytes(manifest)
    completion = {
        "schema_version": "1.0.0",
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "manifest_file_sha256": sha256_bytes(manifest_payload),
        "manifest_sha256_excludes_self": manifest["manifest_sha256_excludes_self"],
        "member_count": len(files) + 2,
    }
    completion_payload = json_bytes(completion)

    with zipfile.ZipFile(output, "x", allowZip64=True) as archive:
        for name in sorted(files):
            archive.writestr(_zip_info(f"{BUNDLE_ROOT}/{name}"), (root / name).read_bytes())
        archive.writestr(_zip_info(f"{BUNDLE_ROOT}/{MANIFEST_NAME}"), manifest_payload)
        archive.writestr(_zip_info(f"{BUNDLE_ROOT}/{COMPLETE_NAME}"), completion_payload)
    return verify_bundle(output)


def verify_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Independently verify a returned backbone transport bundle in place."""

    bundle = Path(bundle_path).resolve()
    if not bundle.is_file():
        raise ValueError(f"backbone bundle does not exist: {bundle}")
    expected_names = {
        *(f"{BUNDLE_ROOT}/{name}" for name in EXPECTED_MODEL_FILES),
        f"{BUNDLE_ROOT}/{MANIFEST_NAME}",
        f"{BUNDLE_ROOT}/{COMPLETE_NAME}",
    }
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("backbone bundle contains duplicate members")
        if set(names) != expected_names:
            raise ValueError("backbone bundle member set drifted")
        for info in archive.infolist():
            if info.is_dir() or info.filename.startswith(("/", "\\")) or ".." in Path(
                info.filename
            ).parts:
                raise ValueError(f"unsafe backbone bundle member: {info.filename}")
            if info.date_time != ZIP_TIMESTAMP or info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"non-deterministic ZIP metadata: {info.filename}")
            if (info.external_attr >> 16) != 0o100644:
                raise ValueError(f"unexpected ZIP member mode: {info.filename}")
        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise ValueError(f"backbone bundle CRC failure: {bad_crc}")

        manifest_payload = archive.read(f"{BUNDLE_ROOT}/{MANIFEST_NAME}")
        manifest = json.loads(manifest_payload)
        if manifest.get("manifest_sha256_excludes_self") != self_hash(
            manifest, "manifest_sha256_excludes_self"
        ):
            raise ValueError("backbone manifest self-hash mismatch")
        if manifest.get("files") != EXPECTED_MODEL_FILES:
            raise ValueError("backbone manifest file contract drifted")
        if manifest.get("model", {}).get("revision") != MODEL_REVISION:
            raise ValueError("backbone manifest revision drifted")
        completion = json.loads(archive.read(f"{BUNDLE_ROOT}/{COMPLETE_NAME}"))
        if completion.get("status") != "complete":
            raise ValueError("backbone completion marker is not complete")
        if completion.get("manifest_file_sha256") != sha256_bytes(manifest_payload):
            raise ValueError("backbone completion marker manifest hash mismatch")
        if completion.get("manifest_sha256_excludes_self") != manifest.get(
            "manifest_sha256_excludes_self"
        ):
            raise ValueError("backbone completion marker self-hash mismatch")
        if int(completion.get("member_count", -1)) != len(expected_names):
            raise ValueError("backbone completion marker member count mismatch")

        for name, expected in EXPECTED_MODEL_FILES.items():
            payload = archive.read(f"{BUNDLE_ROOT}/{name}")
            if len(payload) != expected["size_bytes"] or sha256_bytes(payload) != expected[
                "sha256"
            ]:
                raise ValueError(f"sealed {name} identity mismatch")
        _validate_config(archive.read(f"{BUNDLE_ROOT}/config.json"))

    return {
        "schema_version": "1.0.0",
        "status": "verified",
        "bundle_path": str(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
        "bundle_sha256": sha256_file(bundle),
        "model_repository": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weights_size_bytes": EXPECTED_MODEL_FILES["model.safetensors"]["size_bytes"],
        "weights_sha256": EXPECTED_MODEL_FILES["model.safetensors"]["sha256"],
        "member_count": len(expected_names),
        "manifest_file_sha256": sha256_bytes(manifest_payload),
        "manifest_sha256_excludes_self": manifest["manifest_sha256_excludes_self"],
    }

