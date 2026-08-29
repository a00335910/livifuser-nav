#!/usr/bin/env python3
"""Materialize the frozen simulation train/validation inputs for Kaggle.

The source handoff contains about 15 GiB of RGB arrays. Training consumes the
already frozen DINO cache, so this script deliberately extracts only export
manifests, LiDAR arrays, and vectors while retaining the manifested RGB identity
that the cache loader cross-checks. Held-out manifests or bundles fail closed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

HANDOFF_NAME = "livifuser_confirmatory_v3_train_val_v1"
HANDOFF_SELF_SHA256 = "AB24252411EEF448BC0D853B0C9147AF184F0A1CC14D72BA39876BF179A92C6F"
CACHE_NAME = "livifuser_dinov3_vits16plus_train_val_cache_v2"
CACHE_MANIFEST_SHA256 = "80F2B2AA5265EE8EB179687E8C9AEFEDA618FC4BDC49C4E9FEE25448BCBFC154"
CACHE_SELF_SHA256 = "94319460362DA9D9D47C89ADBEC916DAA1A04384C5A288EC272582A635A50C54"
CACHE_BUNDLE_NAME = "livifuser_dinov3_splus_cache_v2_bundle.zip"
CACHE_BUNDLE_SHA256 = "D395CC63E17F97AFB889D7C39ED481652A8C7197F51096B418912C43975C2B7C"
BACKBONE_CONTRACT_SHA256 = "2957C78346DE608067DD5AC14D5C3E2F23438CD2BB3B1ECA847F898EBA68894A"
SOURCE_FILES = ("manifest.json", "scan_ranges.npy", "vectors.npz")
CACHE_FILES = (
    "manifest.json",
    "SUCCESS.json",
    "patch_tokens_7x7_float16.npy",
    "pooled_features_float32.npy",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def self_hash(value: dict[str, Any], field: str) -> str:
    payload = copy.deepcopy(value)
    payload.pop(field, None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(raw)


def load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def sidecar_hash(path: Path) -> str:
    parts = path.read_text(encoding="utf-8").strip().split()
    require(parts and len(parts[0]) == 64, f"invalid checksum sidecar: {path}")
    return parts[0].upper()


def safe_member(name: str) -> str:
    posix = PurePosixPath(name)
    require(not posix.is_absolute() and ".." not in posix.parts, f"unsafe member: {name}")
    return posix.as_posix()


def write_verified(
    source: BinaryIO,
    target: Path,
    expected_size: int,
    expected_sha256: str,
) -> None:
    if target.is_file():
        require(target.stat().st_size == expected_size, f"existing size drift: {target}")
        require(sha256_file(target) == expected_sha256, f"existing hash drift: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        require(partial.is_file(), f"partial target is not a file: {partial}")
        partial.unlink()
    digest = hashlib.sha256()
    size = 0
    with partial.open("wb") as output:
        while chunk := source.read(8 * 1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    require(size == expected_size, f"materialized size mismatch: {target}")
    require(digest.hexdigest().upper() == expected_sha256, f"materialized hash mismatch: {target}")
    os.replace(partial, target)


def find_one_file(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    require(len(matches) == 1, f"expected one {name}, found {len(matches)}")
    return matches[0]


def refuse_heldout(root: Path) -> None:
    for path in root.rglob("*"):
        relative_parts = path.relative_to(root).parts
        if any("heldout" in part.lower() or "held-out" in part.lower() for part in relative_parts):
            raise RuntimeError(
                f"held-out input name is attached; detach it before execution: {path}"
            )
    if any(
        path.name == "livifuser_dinov3_splus_heldout_cache_v1_bundle.zip"
        for path in root.rglob("*.zip")
    ):
        raise RuntimeError("held-out cache bundle is attached; detach it before training")
    for path in root.rglob("cache_manifest.json"):
        try:
            manifest = load_json_bytes(path.read_bytes(), str(path))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError):
            continue
        splits = set(manifest.get("audit", {}).get("by_split", {}))
        if splits.intersection({"test_id", "test_ood"}):
            raise RuntimeError(f"held-out cache manifest is attached: {path}")
    for path in root.rglob("handoff_manifest.json"):
        try:
            manifest = load_json_bytes(path.read_bytes(), str(path))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError):
            continue
        splits = set(manifest.get("audit", {}).get("by_split", {}))
        if splits.intersection({"test_id", "test_ood"}):
            raise RuntimeError(f"held-out source handoff is attached: {path}")


def discover_handoff(root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for path in root.rglob("handoff_manifest.json"):
        try:
            manifest = load_json_bytes(path.read_bytes(), str(path))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError):
            continue
        if manifest.get("name") == HANDOFF_NAME:
            candidates.append((path, manifest))
    require(len(candidates) == 1, f"expected one frozen source handoff, found {len(candidates)}")
    path, manifest = candidates[0]
    require(
        manifest.get("manifest_sha256_excludes_self") == HANDOFF_SELF_SHA256,
        "source handoff identity mismatch",
    )
    require(
        self_hash(manifest, "manifest_sha256_excludes_self") == HANDOFF_SELF_SHA256,
        "source handoff self-hash mismatch",
    )
    require(manifest["audit"]["episodes"] == 150, "source episode count mismatch")
    require(manifest["audit"]["accepted_samples"] == 69_253, "source row count mismatch")
    require(manifest["audit"]["by_split"] == {"train": 120, "val_id": 30}, "source splits mismatch")
    return path, manifest


def extract_outer_cache_bundle(
    input_root: Path,
    work_root: Path,
    included_splits: set[str] | None = None,
) -> Path:
    bundles = [path for path in input_root.rglob(CACHE_BUNDLE_NAME) if path.is_file()]
    require(len(bundles) == 1, f"expected one {CACHE_BUNDLE_NAME}, found {len(bundles)}")
    bundle = bundles[0]
    require(sha256_file(bundle) == CACHE_BUNDLE_SHA256, "cache bundle hash mismatch")
    target_root = work_root / "_cache_bundle"
    with zipfile.ZipFile(bundle) as archive:
        infos = archive.infolist()
        names = [safe_member(info.filename) for info in infos if not info.is_dir()]
        require(len(names) == len(set(names)) == 19, "cache bundle member set mismatch")
        manifest_name = next(name for name in names if name.endswith("/cache_manifest.json"))
        manifest_raw = archive.read(manifest_name)
        manifest = load_json_bytes(manifest_raw, manifest_name)
        expected_hashes = {
            manifest_name: CACHE_MANIFEST_SHA256,
            next(
                name for name in names if name.endswith("/BACKBONE_CONTRACT.json")
            ): BACKBONE_CONTRACT_SHA256,
        }
        complete_name = next(name for name in names if name.endswith("/CACHE_COMPLETE.json"))
        expected_hashes[complete_name] = sha256_bytes(archive.read(complete_name))
        for shard in manifest["shards"]:
            shard_name = next(
                name for name in names if name.endswith("/shards/" + shard["filename"])
            )
            sidecar_name = shard_name + ".sha256"
            expected_hashes[shard_name] = shard["sha256"]
            expected_hashes[sidecar_name] = sha256_bytes(archive.read(sidecar_name))
        require(set(expected_hashes) == set(names), "cache bundle expected members mismatch")
        selected_shards = {
            shard["filename"]
            for shard in manifest["shards"]
            if included_splits is None or shard["split"] in included_splits
        }
        for info in infos:
            name = safe_member(info.filename)
            if info.is_dir():
                continue
            if "/shards/" in name:
                filename = name.rsplit("/", 1)[-1]
                shard_filename = filename.removesuffix(".sha256")
                if shard_filename not in selected_shards:
                    continue
            target = target_root.joinpath(*PurePosixPath(name).parts)
            with archive.open(info) as source:
                write_verified(source, target, info.file_size, expected_hashes[name])
    return find_one_file(target_root, "cache_manifest.json")


def discover_cache(
    input_root: Path,
    work_root: Path,
    included_splits: set[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for path in input_root.rglob("cache_manifest.json"):
        try:
            raw = path.read_bytes()
            manifest = load_json_bytes(raw, str(path))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError):
            continue
        if manifest.get("name") == CACHE_NAME:
            candidates.append((path, raw, manifest))
    if not candidates:
        path = extract_outer_cache_bundle(input_root, work_root, included_splits)
        raw = path.read_bytes()
        candidates.append((path, raw, load_json_bytes(raw, str(path))))
    require(len(candidates) == 1, f"expected one train/validation cache, found {len(candidates)}")
    path, raw, manifest = candidates[0]
    require(sha256_bytes(raw) == CACHE_MANIFEST_SHA256, "cache manifest file hash mismatch")
    require(
        manifest.get("manifest_sha256_excludes_self") == CACHE_SELF_SHA256,
        "cache manifest declared self-hash mismatch",
    )
    require(
        self_hash(manifest, "manifest_sha256_excludes_self") == CACHE_SELF_SHA256,
        "cache manifest self-hash mismatch",
    )
    require(
        manifest.get("handoff_manifest_sha256") == HANDOFF_SELF_SHA256, "cache handoff mismatch"
    )
    require(
        manifest.get("backbone_contract_sha256") == BACKBONE_CONTRACT_SHA256,
        "cache backbone mismatch",
    )
    require(
        manifest["audit"]
        == {"episodes": 150, "accepted_samples": 69_253, "by_split": {"train": 120, "val_id": 30}},
        "cache audit mismatch",
    )
    complete = load_json_bytes((path.parent / "CACHE_COMPLETE.json").read_bytes(), "CACHE_COMPLETE")
    require(
        complete.get("cache_manifest_sha256") == CACHE_MANIFEST_SHA256, "cache completion mismatch"
    )
    require(
        sha256_file(path.parent / "BACKBONE_CONTRACT.json") == BACKBONE_CONTRACT_SHA256,
        "backbone file mismatch",
    )
    return path, manifest


def source_layout(root: Path, shard: dict[str, Any]) -> tuple[str, Path]:
    archives = [path for path in root.rglob(shard["filename"]) if path.is_file()]
    stem = Path(shard["filename"]).stem
    directories = [
        path
        for path in root.rglob(stem)
        if path.is_dir() and (path / "SHARD_MANIFEST.json").is_file()
    ]
    require(
        len(archives) + len(directories) == 1,
        f"source shard discovery mismatch: {shard['filename']}",
    )
    if archives:
        archive = archives[0]
        require(
            archive.stat().st_size == shard["size_bytes"], f"source shard size mismatch: {archive}"
        )
        require(sha256_file(archive) == shard["sha256"], f"source shard hash mismatch: {archive}")
        sidecar = find_one_file(root, shard["filename"] + ".sha256")
        require(sidecar_hash(sidecar) == shard["sha256"], f"source sidecar mismatch: {archive}")
        return "zip", archive
    return "directory", directories[0]


@contextmanager
def open_member(layout: str, source: Path, name: str) -> Iterator[BinaryIO]:
    name = safe_member(name)
    if layout == "directory":
        with source.joinpath(*PurePosixPath(name).parts).open("rb") as handle:
            yield handle
        return
    with zipfile.ZipFile(source) as archive, archive.open(name) as handle:
        yield handle


def read_member(layout: str, source: Path, name: str) -> bytes:
    with open_member(layout, source, name) as handle:
        return handle.read()


def materialize_sources(
    input_root: Path,
    work_root: Path,
    handoff: dict[str, Any],
    included_splits: set[str] | None = None,
) -> dict[str, Path]:
    episodes_by_id = {episode["episode_id"]: episode for episode in handoff["episodes"]}
    outputs: dict[str, Path] = {}
    for shard in handoff["shards"]:
        layout, source = source_layout(input_root, shard)
        raw = read_member(layout, source, "SHARD_MANIFEST.json")
        shard_manifest = load_json_bytes(raw, f"{source}:SHARD_MANIFEST.json")
        field = "manifest_sha256_excludes_self"
        require(
            shard_manifest.get(field) == self_hash(shard_manifest, field),
            "source shard self-hash mismatch",
        )
        for shard_episode in shard_manifest["episodes"]:
            episode_id = shard_episode["episode_id"]
            expected = episodes_by_id[episode_id]
            require(shard_episode == expected, f"source episode record mismatch: {episode_id}")
            if included_splits is not None and expected["split"] not in included_splits:
                continue
            target_root = work_root / "exports" / episode_id
            for name in SOURCE_FILES:
                member = f"episodes/{episode_id}/export/{name}"
                if name == "manifest.json":
                    record = {"sha256": expected["export_manifest_sha256"]}
                    record["size_bytes"] = next(
                        item["size_bytes"]
                        for item in shard_manifest["members"]
                        if item["path"] == member
                    )
                else:
                    record = expected["outputs"][name]
                with open_member(layout, source, member) as handle:
                    write_verified(
                        handle, target_root / name, int(record["size_bytes"]), record["sha256"]
                    )
            outputs[episode_id] = target_root
    expected_count = sum(
        included_splits is None or episode["split"] in included_splits
        for episode in handoff["episodes"]
    )
    require(len(outputs) == expected_count, "materialized source episode count mismatch")
    return outputs


def cache_layout(search_root: Path, shard: dict[str, Any]) -> tuple[str, Path]:
    archive = search_root / "shards" / shard["filename"]
    if archive.is_file():
        require(
            archive.stat().st_size == shard["size_bytes"], f"cache shard size mismatch: {archive}"
        )
        require(sha256_file(archive) == shard["sha256"], f"cache shard hash mismatch: {archive}")
        sidecar = archive.with_name(archive.name + ".sha256")
        require(sidecar_hash(sidecar) == shard["sha256"], f"cache sidecar mismatch: {archive}")
        return "zip", archive
    stem = Path(shard["filename"]).stem
    directories = [
        path
        for path in search_root.rglob(stem)
        if path.is_dir() and (path / "CACHE_SHARD_MANIFEST.json").is_file()
    ]
    require(len(directories) == 1, f"cache shard discovery mismatch: {shard['filename']}")
    return "directory", directories[0]


def materialize_caches(
    cache_manifest_path: Path,
    work_root: Path,
    cache_manifest: dict[str, Any],
    included_splits: set[str] | None = None,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for shard in cache_manifest["shards"]:
        if included_splits is not None and shard["split"] not in included_splits:
            continue
        layout, source = cache_layout(cache_manifest_path.parent, shard)
        raw = read_member(layout, source, "CACHE_SHARD_MANIFEST.json")
        require(sha256_bytes(raw) == shard["manifest_sha256"], "cache shard manifest hash mismatch")
        manifest = load_json_bytes(raw, f"{source}:CACHE_SHARD_MANIFEST.json")
        field = "manifest_sha256_excludes_self"
        require(manifest.get(field) == self_hash(manifest, field), "cache shard self-hash mismatch")
        declared_members = {item["name"]: item for item in manifest["members"]}
        if layout == "zip":
            with zipfile.ZipFile(source) as archive:
                observed = {info.filename for info in archive.infolist() if not info.is_dir()}
            require(
                observed == {"CACHE_SHARD_MANIFEST.json", *declared_members},
                "cache shard member set mismatch",
            )
        for episode in manifest["episodes"]:
            episode_id = episode["episode_id"]
            require(
                episode["split"] in {"train", "val_id"}, f"held-out cache episode: {episode_id}"
            )
            target_root = work_root / "caches" / episode_id
            for name in CACHE_FILES:
                member = f"episodes/{episode_id}/{name}"
                record = declared_members[member]
                with open_member(layout, source, member) as handle:
                    write_verified(
                        handle, target_root / name, int(record["size_bytes"]), record["sha256"]
                    )
            outputs[episode_id] = target_root
    expected_count = sum(
        int(shard["episode_count"])
        for shard in cache_manifest["shards"]
        if included_splits is None or shard["split"] in included_splits
    )
    require(len(outputs) == expected_count, "materialized cache episode count mismatch")
    return outputs


def build_validation_plan(
    work_root: Path,
    handoff: dict[str, Any],
    exports: dict[str, Path],
    caches: dict[str, Path],
) -> dict[str, Any]:
    episodes = [episode for episode in handoff["episodes"] if episode["split"] == "val_id"]
    episodes.sort(key=lambda episode: int(episode["ordinal"]))
    plan = {
        "schema_version": 1,
        "purpose": "validation_score_freeze_only",
        "source_handoff_self_sha256": HANDOFF_SELF_SHA256,
        "cache_manifest_sha256": CACHE_MANIFEST_SHA256,
        "cache_manifest_self_sha256": CACHE_SELF_SHA256,
        "backbone_contract_sha256": BACKBONE_CONTRACT_SHA256,
        "heldout_attached": False,
        "excluded_splits": ["test_id", "test_ood"],
        "work_root": str(work_root.resolve()),
        "validation": {
            "episode_ids": [episode["episode_id"] for episode in episodes],
            "exports": [str(exports[episode["episode_id"]]) for episode in episodes],
            "caches": [str(caches[episode["episode_id"]]) for episode in episodes],
            "episode_count": len(episodes),
            "accepted_samples": sum(int(episode["accepted_samples"]) for episode in episodes),
            "windows_k8_h8": sum(int(episode["windowable_k8_h8"]) for episode in episodes),
        },
    }
    require(
        set(exports) == set(caches) == set(plan["validation"]["episode_ids"]),
        "validation identities differ",
    )
    require(plan["validation"]["episode_count"] == 30, "validation plan count mismatch")
    require(
        plan["validation"]["accepted_samples"] == 13_125,
        "validation row count mismatch",
    )
    require(
        plan["validation"]["windows_k8_h8"] == 9_459,
        "validation window count mismatch",
    )
    return plan


def build_plan(
    work_root: Path,
    handoff: dict[str, Any],
    exports: dict[str, Path],
    caches: dict[str, Path],
) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "schema_version": 1,
        "source_handoff_self_sha256": HANDOFF_SELF_SHA256,
        "cache_manifest_sha256": CACHE_MANIFEST_SHA256,
        "cache_manifest_self_sha256": CACHE_SELF_SHA256,
        "backbone_contract_sha256": BACKBONE_CONTRACT_SHA256,
        "heldout_attached": False,
        "excluded_splits": ["test_id", "test_ood"],
        "work_root": str(work_root.resolve()),
    }
    for source_split, plan_split in (("train", "train"), ("val_id", "validation")):
        episodes = [episode for episode in handoff["episodes"] if episode["split"] == source_split]
        episodes.sort(key=lambda episode: int(episode["ordinal"]))
        plan[plan_split] = {
            "episode_ids": [episode["episode_id"] for episode in episodes],
            "exports": [str(exports[episode["episode_id"]]) for episode in episodes],
            "caches": [str(caches[episode["episode_id"]]) for episode in episodes],
            "episode_count": len(episodes),
            "accepted_samples": sum(int(episode["accepted_samples"]) for episode in episodes),
            "windows_k8_h8": sum(int(episode["windowable_k8_h8"]) for episode in episodes),
        }
    require(plan["train"]["episode_count"] == 120, "training plan count mismatch")
    require(plan["validation"]["episode_count"] == 30, "validation plan count mismatch")
    require(plan["train"]["accepted_samples"] == 56_128, "training row count mismatch")
    require(plan["validation"]["accepted_samples"] == 13_125, "validation row count mismatch")
    require(plan["train"]["windows_k8_h8"] == 41_367, "training window count mismatch")
    require(plan["validation"]["windows_k8_h8"] == 9_459, "validation window count mismatch")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    work_root = args.work_root.resolve()
    require(input_root.is_dir(), f"input root is missing: {input_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    refuse_heldout(input_root)
    _handoff_path, handoff = discover_handoff(input_root)
    included_splits = {"val_id"} if args.validation_only else None
    cache_path, cache_manifest = discover_cache(input_root, work_root, included_splits)
    exports = materialize_sources(input_root, work_root, handoff, included_splits)
    caches = materialize_caches(cache_path, work_root, cache_manifest, included_splits)
    require(set(exports) == set(caches), "source/cache episode identities differ")
    plan = (
        build_validation_plan(work_root, handoff, exports, caches)
        if args.validation_only
        else build_plan(work_root, handoff, exports, caches)
    )
    output = args.plan_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plan, indent=2) + "\n"
    if output.exists():
        require(output.read_text(encoding="utf-8") == payload, "existing plan differs")
    else:
        output.write_text(payload, encoding="utf-8", newline="\n")
    report = {"plan": str(output), "validation": plan["validation"]}
    if "train" in plan:
        report["train"] = plan["train"]
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
