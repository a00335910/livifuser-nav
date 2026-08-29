"""Hash-manifest verification for portable cloud-training bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_relative_path(value: str) -> Path:
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise ValueError(f"unsafe bundle path: {value!r}")
    return Path(*posix.parts)


def verify_cloud_bundle(repository_root: Path) -> dict[str, Any]:
    manifest_path = repository_root / "cloud_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported cloud bundle schema")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("cloud bundle manifest has no files")
    seen: set[str] = set()
    total_bytes = 0
    for entry in entries:
        relative_text = str(entry["path"])
        if relative_text in seen:
            raise ValueError(f"duplicate bundle path: {relative_text}")
        seen.add(relative_text)
        relative = validate_relative_path(relative_text)
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_size = int(entry["size_bytes"])
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"bundle size mismatch: {relative_text}")
        if sha256_file(path).lower() != str(entry["sha256"]).lower():
            raise ValueError(f"bundle hash mismatch: {relative_text}")
        total_bytes += actual_size
    if int(manifest["file_count"]) != len(entries):
        raise ValueError("cloud bundle file count mismatch")
    if int(manifest["total_bytes"]) != total_bytes:
        raise ValueError("cloud bundle total byte count mismatch")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "git_revision": str(manifest["git_revision"]),
        "file_count": len(entries),
        "total_bytes": total_bytes,
    }
