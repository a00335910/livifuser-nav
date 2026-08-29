#!/usr/bin/env python3
"""Audit and package the frozen 110 held-out confirmatory-v3 exports.

The package contains accepted derived exports only. Raw MCAPs stay local as
provenance evidence. Test-ID and test-OOD are separated into homogeneous
split/world/condition ZIP shards for frozen DINO feature extraction.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from package_sim_train_val_handoff import (  # noqa: E402
    AuditedEpisode,
    audit_episode,
    canonical_bytes,
    json_bytes,
    load_json,
    sha256_bytes,
    sha256_file,
)

DEFAULT_SIMULATION_ROOT = REPOSITORY_ROOT / "artifacts/simulation/confirmatory_v3"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "artifacts/simulation/gpu_handoff_heldout_v1"
EXPECTED_SCHEDULE_SHA256 = "85F48F81D52F82A5CD64D14D704E032617582D7F5462A7B6FD46C5E64899047D"
EXPECTED_AUDIT_SHA256 = "643E70F9A36EF8B4146E29673328C8A5F257BD77275BCB5446EFDABFE90F7220"
EXPECTED_SPLITS = {"test_id": 30, "test_ood": 80}
EXPECTED_CONDITIONS = {"C0": 50, "C1": 20, "C3": 20, "C4": 20}
EXPECTED_ORDINALS = list(range(150, 260))
EXPECTED_ACCEPTED_SAMPLES = 47_326
EXPECTED_WINDOWS = 34_503
HANDOFF_NAME = "livifuser_confirmatory_v3_heldout_v1"
HANDOFF_SCHEMA = "1.0.0"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class Audit:
    schedule: dict[str, Any]
    schedule_file_sha256: str
    post_collection_audit_file_sha256: str
    episodes: tuple[AuditedEpisode, ...]
    summary: dict[str, Any]


def manifest_self_hash(value: dict[str, Any], field: str) -> str:
    payload = copy.deepcopy(value)
    payload.pop(field, None)
    return sha256_bytes(canonical_bytes(payload))


def select_entries(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    declared = schedule.get("schedule_sha256_excludes_self")
    if manifest_self_hash(schedule, "schedule_sha256_excludes_self") != declared:
        raise ValueError("schedule self-hash mismatch")
    if declared != EXPECTED_SCHEDULE_SHA256:
        raise ValueError(f"unexpected schedule identity: {declared}")
    selected = [entry for entry in schedule["episodes"] if entry["split"] in EXPECTED_SPLITS]
    selected.sort(key=lambda entry: int(entry["ordinal"]))
    if [int(entry["ordinal"]) for entry in selected] != EXPECTED_ORDINALS:
        raise ValueError("held-out entries are not frozen ordinals 150-259")
    if Counter(entry["split"] for entry in selected) != Counter(EXPECTED_SPLITS):
        raise ValueError("held-out split counts drifted")
    if Counter(entry["condition"] for entry in selected) != Counter(EXPECTED_CONDITIONS):
        raise ValueError("held-out condition counts drifted")
    return selected


def verify_post_collection_audit(simulation_root: Path) -> tuple[dict[str, Any], str]:
    path = simulation_root / "post_collection_audit.json"
    value = load_json(path)
    declared = value.get("audit_sha256_excludes_self")
    if manifest_self_hash(value, "audit_sha256_excludes_self") != declared:
        raise ValueError("post-collection audit self-hash mismatch")
    if declared != EXPECTED_AUDIT_SHA256 or value.get("status") != "PASS":
        raise ValueError("post-collection audit is not the accepted PASS artifact")
    if int(value["counts"]["episodes"]) != 260:
        raise ValueError("post-collection audit does not cover all 260 episodes")
    return value, sha256_file(path)


def audit_handoff(simulation_root: Path) -> Audit:
    simulation_root = simulation_root.resolve()
    schedule_path = simulation_root / "schedule.json"
    schedule = load_json(schedule_path)
    selected = select_entries(schedule)
    _post_audit, post_audit_file_sha = verify_post_collection_audit(simulation_root)
    episodes = tuple(audit_episode(simulation_root, entry) for entry in selected)
    by_split = Counter(item.record["split"] for item in episodes)
    by_condition = Counter(item.record["condition"] for item in episodes)
    by_world = Counter(item.record["world_name"] for item in episodes)
    source_trees = sorted({item.record["source_tree_sha256"] for item in episodes})
    configurations = sorted({item.record["effective_configuration_sha256"] for item in episodes})
    if len(source_trees) != 1 or len(configurations) != 1:
        raise ValueError("held-out exports do not share one source/configuration")
    accepted_samples = sum(item.record["accepted_samples"] for item in episodes)
    windows = sum(item.record["windowable_k8_h8"] for item in episodes)
    if accepted_samples != EXPECTED_ACCEPTED_SAMPLES:
        raise ValueError(f"held-out sample count drifted: {accepted_samples}")
    if windows != EXPECTED_WINDOWS:
        raise ValueError(f"held-out window count drifted: {windows}")
    total_bytes = sum(
        output["size_bytes"] for item in episodes for output in item.record["outputs"].values()
    )
    summary = {
        "episodes": len(episodes),
        "by_split": dict(sorted(by_split.items())),
        "by_condition": dict(sorted(by_condition.items())),
        "by_world": dict(sorted(by_world.items())),
        "accepted_samples": accepted_samples,
        "windowable_k8_h8": windows,
        "export_payload_bytes": total_bytes,
        "export_payload_gib": round(total_bytes / 1024**3, 3),
        "source_tree_sha256": source_trees[0],
        "effective_configuration_sha256": configurations[0],
    }
    return Audit(
        schedule=schedule,
        schedule_file_sha256=sha256_file(schedule_path),
        post_collection_audit_file_sha256=post_audit_file_sha,
        episodes=episodes,
        summary=summary,
    )


def shard_key(episode: AuditedEpisode) -> str:
    record = episode.record
    world_name = str(record["world_name"])
    condition = str(record["condition"]).lower()
    if record["split"] == "test_id":
        return f"{world_name}_{condition}"
    if not world_name.startswith("test_id_"):
        raise ValueError(f"unexpected held-out world name: {world_name}")
    return f"test_ood_{world_name.removeprefix('test_id_')}_{condition}"


def shard_groups(audit: Audit) -> dict[str, tuple[AuditedEpisode, ...]]:
    grouped: dict[str, list[AuditedEpisode]] = defaultdict(list)
    for episode in audit.episodes:
        grouped[shard_key(episode)].append(episode)
    result = {
        key: tuple(sorted(values, key=lambda item: item.record["ordinal"]))
        for key, values in sorted(grouped.items())
    }
    if len(result) != 10:
        raise ValueError(f"expected 10 held-out shards, found {len(result)}")
    for key, episodes in result.items():
        if len({item.record["split"] for item in episodes}) != 1:
            raise ValueError(f"mixed split in shard {key}")
        if len({item.record["condition"] for item in episodes}) != 1:
            raise ValueError(f"mixed condition in shard {key}")
        if len({item.record["world_name"] for item in episodes}) != 1:
            raise ValueError(f"mixed world in shard {key}")
    return result


def shard_manifest(audit: Audit, key: str, episodes: tuple[AuditedEpisode, ...]) -> dict[str, Any]:
    members = []
    for episode in episodes:
        base = f"episodes/{episode.record['episode_id']}/export"
        members.append(
            {
                "path": f"{base}/manifest.json",
                "size_bytes": (episode.export_root / "manifest.json").stat().st_size,
                "sha256": episode.record["export_manifest_sha256"],
            }
        )
        for name, record in episode.record["outputs"].items():
            members.append({"path": f"{base}/{name}", **record})
    value = {
        "schema_version": HANDOFF_SCHEMA,
        "handoff": HANDOFF_NAME,
        "schedule_sha256_excludes_self": audit.schedule["schedule_sha256_excludes_self"],
        "post_collection_audit_sha256_excludes_self": EXPECTED_AUDIT_SHA256,
        "shard_key": key,
        "world_name": episodes[0].record["world_name"],
        "split": episodes[0].record["split"],
        "condition": episodes[0].record["condition"],
        "episode_count": len(episodes),
        "episodes": [episode.record for episode in episodes],
        "members": sorted(members, key=lambda item: item["path"]),
    }
    value["manifest_sha256_excludes_self"] = sha256_bytes(canonical_bytes(value))
    return value


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def add_file(archive: zipfile.ZipFile, source: Path, member: str) -> None:
    with source.open("rb") as input_handle, archive.open(zip_info(member), "w") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)


def build_shard(
    output_root: Path,
    audit: Audit,
    key: str,
    episodes: tuple[AuditedEpisode, ...],
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    archive_path = output_root / f"{key}.zip"
    sidecar_path = output_root / f"{key}.zip.sha256"
    manifest = shard_manifest(audit, key, episodes)
    if archive_path.exists():
        if not sidecar_path.is_file():
            raise FileExistsError(f"unverified existing shard: {archive_path}")
        expected = sidecar_path.read_text(encoding="ascii").strip().split()[0].upper()
        actual = sha256_file(archive_path)
        if expected != actual:
            raise ValueError(f"existing shard hash mismatch: {archive_path.name}")
        return shard_record(archive_path, episodes)

    partial = archive_path.with_suffix(".zip.partial")
    if partial.exists():
        raise FileExistsError(f"remove interrupted partial before retrying: {partial}")
    with zipfile.ZipFile(
        partial,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        archive.writestr(zip_info("SHARD_MANIFEST.json"), json_bytes(manifest))
        for episode in episodes:
            base = f"episodes/{episode.record['episode_id']}/export"
            add_file(
                archive,
                episode.export_root / "manifest.json",
                f"{base}/manifest.json",
            )
            for name in sorted(episode.record["outputs"]):
                add_file(archive, episode.export_root / name, f"{base}/{name}")
    partial.replace(archive_path)
    digest = sha256_file(archive_path)
    sidecar_path.write_text(f"{digest}  {archive_path.name}\n", encoding="ascii")
    return shard_record(archive_path, episodes)


def shard_record(archive_path: Path, episodes: tuple[AuditedEpisode, ...]) -> dict[str, Any]:
    return {
        "filename": archive_path.name,
        "sha256": sha256_file(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "episode_count": len(episodes),
        "world_name": episodes[0].record["world_name"],
        "split": episodes[0].record["split"],
        "condition": episodes[0].record["condition"],
        "status": "complete",
    }


def write_handoff_index(
    output_root: Path, audit: Audit, shard_records: list[dict[str, Any]]
) -> dict[str, Any]:
    value = {
        "schema_version": HANDOFF_SCHEMA,
        "name": HANDOFF_NAME,
        "source": {
            "schedule_sha256_excludes_self": audit.schedule["schedule_sha256_excludes_self"],
            "schedule_file_sha256": audit.schedule_file_sha256,
            "post_collection_audit_sha256_excludes_self": EXPECTED_AUDIT_SHA256,
            "post_collection_audit_file_sha256": (audit.post_collection_audit_file_sha256),
            "recollection_freeze_manifest": audit.schedule["recollection_freeze_manifest"],
            "recollection_freeze_manifest_sha256": audit.schedule[
                "recollection_freeze_manifest_sha256"
            ],
        },
        "selection": {
            "splits": ["test_id", "test_ood"],
            "conditions": ["C0", "C1", "C3", "C4"],
            "ordinal_range_inclusive": [150, 259],
            "raw_mcap_included": False,
            "accepted_attempt_only": True,
            "held_out_from_training_and_statistic_fitting": True,
        },
        "audit": audit.summary,
        "episodes": [episode.record for episode in audit.episodes],
        "shards": sorted(shard_records, key=lambda item: item["filename"]),
    }
    value["manifest_sha256_excludes_self"] = sha256_bytes(canonical_bytes(value))
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "handoff_manifest.json"
    payload = json_bytes(value)
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError(f"refusing to overwrite different handoff: {path}")
    path.write_bytes(payload)
    readme = (
        "LiViFuser confirmatory-v3 held-out GPU handoff.\n"
        "Contains test-ID and test-OOD accepted exports only; raw MCAPs are excluded. "
        "These features must never be used for training, normalization, Gaussian or "
        "Mahalanobis fitting, checkpoint selection, or threshold tuning. Verify every "
        "ZIP and embedded SHARD_MANIFEST.json before DINO extraction.\n"
    )
    readme_path = output_root / "README.txt"
    if readme_path.exists() and readme_path.read_text(encoding="utf-8") != readme:
        raise FileExistsError(f"refusing to overwrite different README: {readme_path}")
    readme_path.write_text(readme, encoding="utf-8")
    return value


def verify_archives(output_root: Path, index: dict[str, Any], *, deep: bool) -> None:
    for shard in index["shards"]:
        path = output_root / shard["filename"]
        if path.stat().st_size != int(shard["size_bytes"]):
            raise ValueError(f"archive size mismatch: {path.name}")
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"archive hash mismatch: {path.name}")
        sidecar = path.with_name(path.name + ".sha256")
        if sidecar.read_text(encoding="ascii").strip().split()[0].upper() != shard["sha256"]:
            raise ValueError(f"archive sidecar mismatch: {path.name}")
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"corrupt archive: {path.name}")
            manifest = json.loads(archive.read("SHARD_MANIFEST.json"))
            expected_names = {
                "SHARD_MANIFEST.json",
                *(member["path"] for member in manifest["members"]),
            }
            if set(archive.namelist()) != expected_names:
                raise ValueError(f"archive member set mismatch: {path.name}")
            declared = manifest.pop("manifest_sha256_excludes_self")
            if sha256_bytes(canonical_bytes(manifest)) != declared:
                raise ValueError(f"embedded manifest self-hash mismatch: {path.name}")
            if deep:
                for member in manifest["members"]:
                    payload = archive.read(member["path"])
                    if len(payload) != int(member["size_bytes"]):
                        raise ValueError(f"member size mismatch: {path.name}:{member['path']}")
                    if sha256_bytes(payload) != member["sha256"]:
                        raise ValueError(f"member hash mismatch: {path.name}:{member['path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-root", type=Path, default=DEFAULT_SIMULATION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    audit = audit_handoff(args.simulation_root)
    print(json.dumps({"audit": audit.summary}, indent=2))
    groups = shard_groups(audit)
    if not args.build and not args.verify:
        return 0

    output_root = args.output.resolve()
    records = []
    selected = list(groups.items())
    if args.limit is not None:
        selected = selected[: args.limit]
    if args.build:
        for index, (key, episodes) in enumerate(selected, start=1):
            print(f"[{index}/{len(selected)}] {key}", flush=True)
            record = build_shard(output_root, audit, key, episodes)
            records.append(record)
            print(json.dumps(record, indent=2), flush=True)
        if len(records) == len(groups):
            index = write_handoff_index(output_root, audit, records)
            verify_archives(output_root, index, deep=False)
            print(
                json.dumps(
                    {
                        "complete": True,
                        "output": str(output_root),
                        "shards": len(records),
                        "compressed_gib": round(
                            sum(item["size_bytes"] for item in records) / 1024**3,
                            3,
                        ),
                        "manifest_sha256_excludes_self": index["manifest_sha256_excludes_self"],
                    },
                    indent=2,
                )
            )
        else:
            print(json.dumps({"complete": False, "built_this_run": len(records)}, indent=2))
    if args.verify:
        index = load_json(output_root / "handoff_manifest.json")
        declared = index.get("manifest_sha256_excludes_self")
        if manifest_self_hash(index, "manifest_sha256_excludes_self") != declared:
            raise ValueError("handoff manifest self-hash mismatch")
        verify_archives(output_root, index, deep=args.deep)
        print(json.dumps({"verified": True, "deep": args.deep}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
