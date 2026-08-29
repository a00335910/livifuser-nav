#!/usr/bin/env python3
"""Verify and materialize only the frozen simulation held-out inputs."""

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

HANDOFF_NAME = "livifuser_confirmatory_v3_heldout_v1"
HANDOFF_SELF_SHA256 = "3F48A7E54A1596947A469B59B8D63EE96FD29294C7BD57EFAC924736A984492C"
CACHE_NAME = "livifuser_dinov3_vits16plus_heldout_cache_v1"
CACHE_BUNDLE_NAME = "livifuser_dinov3_splus_heldout_cache_v1_bundle.zip"
CACHE_BUNDLE_SHA256 = "7FB323948427AB6FC1F5F82F2CEF5E66DDB51F056C031132B2EE8C9B9F0484E5"
CACHE_MANIFEST_SHA256 = "6E7E51176FE494634303D756BDCF8D9BB5D28C81754CC26F6C11180BFCB1FD42"
CACHE_SELF_SHA256 = "9FC559291790B5FDB1E77F62FD1A160C4498BFD4591B86B420BEC8C89BC4A5F7"
BACKBONE_CONTRACT_SHA256 = "2957C78346DE608067DD5AC14D5C3E2F23438CD2BB3B1ECA847F898EBA68894A"
SOURCE_FILES = ("manifest.json", "scan_ranges.npy", "vectors.npz")
CACHE_FILES = (
    "manifest.json",
    "SUCCESS.json",
    "patch_tokens_7x7_float16.npy",
    "pooled_features_float32.npy",
)
EXPECTED_SPLITS = {"test_id": 30, "test_ood": 80}
EXPECTED_CONDITIONS = {"C0": 50, "C1": 20, "C3": 20, "C4": 20}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_stream(handle: BinaryIO, chunk_size: int = 8 * 1024 * 1024) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while chunk := handle.read(chunk_size):
        digest.update(chunk)
        size += len(chunk)
    return size, digest.hexdigest().upper()


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


def safe_member(name: str) -> str:
    member = PurePosixPath(name)
    require(not member.is_absolute() and ".." not in member.parts, f"unsafe member: {name}")
    return member.as_posix()


def sidecar_hash(path: Path) -> str:
    fields = path.read_text(encoding="utf-8").strip().split()
    require(fields and len(fields[0]) == 64, f"invalid checksum sidecar: {path}")
    return fields[0].upper()


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
    require(not partial.exists(), f"refusing stale partial output: {partial}")
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


def discover_handoff(root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for path in root.rglob("handoff_manifest.json"):
        try:
            raw = path.read_bytes()
            manifest = load_json_bytes(raw, str(path))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError):
            continue
        if manifest.get("name") == HANDOFF_NAME:
            candidates.append((path, manifest))
    require(len(candidates) == 1, f"expected one held-out handoff, found {len(candidates)}")
    path, manifest = candidates[0]
    field = "manifest_sha256_excludes_self"
    require(manifest.get(field) == HANDOFF_SELF_SHA256, "held-out handoff identity drift")
    require(self_hash(manifest, field) == HANDOFF_SELF_SHA256, "held-out handoff self-hash drift")
    audit = manifest["audit"]
    require(
        audit["episodes"] == 110
        and audit["accepted_samples"] == 47_326
        and audit["windowable_k8_h8"] == 34_503
        and audit["by_split"] == EXPECTED_SPLITS
        and audit["by_condition"] == EXPECTED_CONDITIONS,
        "held-out handoff audit drift",
    )
    ordinals = sorted(int(row["ordinal"]) for row in manifest["episodes"])
    require(ordinals == list(range(150, 260)), "held-out ordinal set drift")
    return path, manifest


def extract_cache_bundle(input_root: Path, work_root: Path) -> Path:
    candidates = [path for path in input_root.rglob(CACHE_BUNDLE_NAME) if path.is_file()]
    require(len(candidates) == 1, f"expected one {CACHE_BUNDLE_NAME}, found {len(candidates)}")
    bundle = candidates[0]
    require(sha256_file(bundle) == CACHE_BUNDLE_SHA256, "held-out cache transport hash drift")
    target_root = work_root / "_heldout_cache_bundle"
    with zipfile.ZipFile(bundle) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = [safe_member(info.filename) for info in infos]
        require(len(names) == len(set(names)) == 23, "held-out cache bundle member drift")
        require(archive.testzip() is None, "held-out cache bundle CRC failure")
        manifest_info = next(info for info in infos if info.filename == "cache_manifest.json")
        manifest_raw = archive.read(manifest_info)
        require(sha256_bytes(manifest_raw) == CACHE_MANIFEST_SHA256, "cache manifest hash drift")
        manifest = load_json_bytes(manifest_raw, "cache_manifest.json")
        expected = {
            "cache_manifest.json": CACHE_MANIFEST_SHA256,
            "BACKBONE_CONTRACT.json": BACKBONE_CONTRACT_SHA256,
            "CACHE_COMPLETE.json": sha256_bytes(archive.read("CACHE_COMPLETE.json")),
        }
        for shard in manifest["shards"]:
            name = "shards/" + shard["filename"]
            expected[name] = shard["sha256"]
            expected[name + ".sha256"] = sha256_bytes(archive.read(name + ".sha256"))
        require(set(names) == set(expected), "held-out cache transport exact set drift")
        for info in infos:
            name = safe_member(info.filename)
            target = target_root.joinpath(*PurePosixPath(name).parts)
            with archive.open(info) as source:
                write_verified(source, target, info.file_size, expected[name])
    return target_root / "cache_manifest.json"


def discover_cache(input_root: Path, work_root: Path) -> tuple[Path, dict[str, Any]]:
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
        path = extract_cache_bundle(input_root, work_root)
        raw = path.read_bytes()
        candidates.append((path, raw, load_json_bytes(raw, str(path))))
    require(len(candidates) == 1, f"expected one held-out cache, found {len(candidates)}")
    path, raw, manifest = candidates[0]
    field = "manifest_sha256_excludes_self"
    require(sha256_bytes(raw) == CACHE_MANIFEST_SHA256, "cache manifest file hash drift")
    require(manifest.get(field) == CACHE_SELF_SHA256, "cache declared self-hash drift")
    require(self_hash(manifest, field) == CACHE_SELF_SHA256, "cache self-hash drift")
    require(manifest.get("handoff_manifest_sha256") == HANDOFF_SELF_SHA256, "cache/handoff drift")
    require(
        manifest.get("backbone_contract_sha256") == BACKBONE_CONTRACT_SHA256,
        "cache/backbone drift",
    )
    require(
        manifest["audit"]
        == {"episodes": 110, "accepted_samples": 47_326, "by_split": EXPECTED_SPLITS},
        "cache audit drift",
    )
    complete = load_json_bytes((path.parent / "CACHE_COMPLETE.json").read_bytes(), "CACHE_COMPLETE")
    require(complete == {"cache_manifest_sha256": CACHE_MANIFEST_SHA256}, "completion marker drift")
    require(
        sha256_file(path.parent / "BACKBONE_CONTRACT.json") == BACKBONE_CONTRACT_SHA256,
        "backbone contract hash drift",
    )
    return path, manifest


def shard_layout(root: Path, filename: str, manifest_name: str) -> tuple[str, Path]:
    archives = [path for path in root.rglob(filename) if path.is_file()]
    stem = Path(filename).stem
    directories = [
        path for path in root.rglob(stem) if path.is_dir() and (path / manifest_name).is_file()
    ]
    require(
        len(archives) + len(directories) == 1,
        f"shard discovery mismatch: {filename}",
    )
    return ("zip", archives[0]) if archives else ("directory", directories[0])


@contextmanager
def open_member(layout: str, source: Path, name: str) -> Iterator[BinaryIO]:
    member = safe_member(name)
    if layout == "directory":
        with source.joinpath(*PurePosixPath(member).parts).open("rb") as handle:
            yield handle
        return
    with zipfile.ZipFile(source) as archive, archive.open(member) as handle:
        yield handle


def read_member(layout: str, source: Path, name: str) -> bytes:
    with open_member(layout, source, name) as handle:
        return handle.read()


def verify_nested_members(
    layout: str,
    source: Path,
    manifest_name: str,
    records: dict[str, dict[str, Any]],
) -> None:
    expected = {manifest_name, *records}
    if layout == "zip":
        with zipfile.ZipFile(source) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            observed = {safe_member(info.filename) for info in infos}
            require(
                len(infos) == len(observed) and observed == expected, f"nested set drift: {source}"
            )
            require(archive.testzip() is None, f"nested CRC failure: {source}")
            for name, record in records.items():
                with archive.open(name) as handle:
                    size, digest = sha256_stream(handle)
                require(
                    size == int(record["size_bytes"]) and digest == record["sha256"],
                    f"nested member drift: {source}:{name}",
                )
        return
    observed = {path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()}
    require(observed == expected, f"expanded nested set drift: {source}")
    for name, record in records.items():
        target = source.joinpath(*PurePosixPath(name).parts)
        require(
            target.stat().st_size == int(record["size_bytes"])
            and sha256_file(target) == record["sha256"],
            f"expanded nested member drift: {target}",
        )


def materialize_sources(
    input_root: Path,
    work_root: Path,
    handoff: dict[str, Any],
) -> dict[str, Path]:
    expected_episodes = {row["episode_id"]: row for row in handoff["episodes"]}
    outputs: dict[str, Path] = {}
    for shard in handoff["shards"]:
        layout, source = shard_layout(input_root, shard["filename"], "SHARD_MANIFEST.json")
        sidecars = [
            path for path in input_root.rglob(shard["filename"] + ".sha256") if path.is_file()
        ]
        require(
            len(sidecars) == 1 and sidecar_hash(sidecars[0]) == shard["sha256"],
            "source sidecar drift",
        )
        if layout == "zip":
            require(
                source.stat().st_size == int(shard["size_bytes"]), f"source size drift: {source}"
            )
            require(sha256_file(source) == shard["sha256"], f"source shard hash drift: {source}")
        raw = read_member(layout, source, "SHARD_MANIFEST.json")
        manifest = load_json_bytes(raw, f"{source}:SHARD_MANIFEST.json")
        field = "manifest_sha256_excludes_self"
        require(manifest.get(field) == self_hash(manifest, field), "source shard self-hash drift")
        records = {str(row["path"]): row for row in manifest["members"]}
        verify_nested_members(layout, source, "SHARD_MANIFEST.json", records)
        for episode in manifest["episodes"]:
            identity = str(episode["episode_id"])
            require(
                episode == expected_episodes[identity], f"source episode record drift: {identity}"
            )
            target_root = work_root / "exports" / identity
            for name in SOURCE_FILES:
                member = f"episodes/{identity}/export/{name}"
                record = records[member]
                with open_member(layout, source, member) as handle:
                    write_verified(
                        handle, target_root / name, int(record["size_bytes"]), record["sha256"]
                    )
            outputs[identity] = target_root
    require(set(outputs) == set(expected_episodes), "source episode set drift")
    return outputs


def materialize_caches(
    cache_manifest_path: Path,
    work_root: Path,
    cache_manifest: dict[str, Any],
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for shard in cache_manifest["shards"]:
        layout, source = shard_layout(
            cache_manifest_path.parent, shard["filename"], "CACHE_SHARD_MANIFEST.json"
        )
        sidecars = [
            path
            for path in cache_manifest_path.parent.rglob(shard["filename"] + ".sha256")
            if path.is_file()
        ]
        require(
            len(sidecars) == 1 and sidecar_hash(sidecars[0]) == shard["sha256"],
            "cache sidecar drift",
        )
        if layout == "zip":
            require(
                source.stat().st_size == int(shard["size_bytes"]), f"cache size drift: {source}"
            )
            require(sha256_file(source) == shard["sha256"], f"cache shard hash drift: {source}")
        raw = read_member(layout, source, "CACHE_SHARD_MANIFEST.json")
        require(sha256_bytes(raw) == shard["manifest_sha256"], "cache shard manifest hash drift")
        manifest = load_json_bytes(raw, f"{source}:CACHE_SHARD_MANIFEST.json")
        field = "manifest_sha256_excludes_self"
        require(manifest.get(field) == self_hash(manifest, field), "cache shard self-hash drift")
        records = {str(row["name"]): row for row in manifest["members"]}
        verify_nested_members(layout, source, "CACHE_SHARD_MANIFEST.json", records)
        for episode in manifest["episodes"]:
            identity = str(episode["episode_id"])
            require(episode["split"] in EXPECTED_SPLITS, f"non-held-out cache episode: {identity}")
            target_root = work_root / "caches" / identity
            for name in CACHE_FILES:
                member = f"episodes/{identity}/{name}"
                record = records[member]
                with open_member(layout, source, member) as handle:
                    write_verified(
                        handle, target_root / name, int(record["size_bytes"]), record["sha256"]
                    )
            outputs[identity] = target_root
    require(len(outputs) == 110, "cache episode count drift")
    return outputs


def build_plan(
    work_root: Path,
    handoff: dict[str, Any],
    exports: dict[str, Path],
    caches: dict[str, Path],
) -> dict[str, Any]:
    episodes = sorted(handoff["episodes"], key=lambda row: int(row["ordinal"]))
    require(
        set(exports) == set(caches) == {row["episode_id"] for row in episodes},
        "episode identity drift",
    )
    records = []
    for episode in episodes:
        condition = "C3b" if episode["condition"] == "C3" else episode["condition"]
        records.append(
            {
                "episode_id": episode["episode_id"],
                "split": episode["split"],
                "world_name": episode["world_name"],
                "condition": condition,
                "episode_index": int(episode["episode_index"]),
                "observation_seed": int(episode["observation_seed"]),
                "ordinal": int(episode["ordinal"]),
                "accepted_samples": int(episode["accepted_samples"]),
                "windows_k8_h8": int(episode["windowable_k8_h8"]),
                "export": str(exports[episode["episode_id"]]),
                "cache": str(caches[episode["episode_id"]]),
            }
        )
    plan = {
        "schema_version": 1,
        "purpose": "approved_one_time_simulation_heldout_evaluation",
        "source_handoff_self_sha256": HANDOFF_SELF_SHA256,
        "cache_transport_sha256": CACHE_BUNDLE_SHA256,
        "cache_manifest_sha256": CACHE_MANIFEST_SHA256,
        "cache_manifest_self_sha256": CACHE_SELF_SHA256,
        "backbone_contract_sha256": BACKBONE_CONTRACT_SHA256,
        "heldout_attached": True,
        "allowed_splits": ["test_id", "test_ood"],
        "work_root": str(work_root.resolve()),
        "episode_count": len(records),
        "accepted_samples": sum(row["accepted_samples"] for row in records),
        "windows_k8_h8": sum(row["windows_k8_h8"] for row in records),
        "episodes": records,
    }
    require(
        (plan["episode_count"], plan["accepted_samples"], plan["windows_k8_h8"])
        == (110, 47_326, 34_503),
        "held-out plan totals drift",
    )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    work_root = args.work_root.resolve()
    require(input_root.is_dir(), f"input root is missing: {input_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    _handoff_path, handoff = discover_handoff(input_root)
    cache_path, cache_manifest = discover_cache(input_root, work_root)
    exports = materialize_sources(input_root, work_root, handoff)
    caches = materialize_caches(cache_path, work_root, cache_manifest)
    plan = build_plan(work_root, handoff, exports, caches)
    output = args.plan_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plan, indent=2) + chr(10)
    if output.exists():
        require(output.read_text(encoding="utf-8") == payload, "existing held-out plan drift")
    else:
        output.write_text(payload, encoding="utf-8", newline=chr(10))
    print(
        json.dumps(
            {
                "plan": str(output),
                "plan_sha256": sha256_file(output),
                "episode_count": plan["episode_count"],
                "accepted_samples": plan["accepted_samples"],
                "windows_k8_h8": plan["windows_k8_h8"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
