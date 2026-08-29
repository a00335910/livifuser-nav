#!/usr/bin/env python3
"""Audit and package the frozen v3 train/validation exports for GPU transfer.

The raw MCAPs are provenance evidence and are intentionally excluded.  The
handoff contains only the accepted export selected by each SUCCESS marker.
Default mode is a write-free audit.  ``--build`` creates one deterministic,
lossless ZIP per world and can resume from already verified shards.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIMULATION_ROOT = REPOSITORY_ROOT / "artifacts/simulation/confirmatory_v3"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "artifacts/simulation/gpu_handoff_train_val_v1"
EXPECTED_SCHEDULE_SHA256 = "85F48F81D52F82A5CD64D14D704E032617582D7F5462A7B6FD46C5E64899047D"
EXPECTED_COUNTS = {"train": 120, "val_id": 30}
EXPECTED_EXPORT_SCHEMA = "1.3.0"
HANDOFF_SCHEMA = "1.0.0"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def manifest_self_hash(value: dict[str, Any], field: str) -> str:
    payload = copy.deepcopy(value)
    payload.pop(field, None)
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode())


@dataclass(frozen=True)
class AuditedEpisode:
    entry: dict[str, Any]
    export_root: Path
    record: dict[str, Any]


@dataclass(frozen=True)
class Audit:
    schedule: dict[str, Any]
    schedule_file_sha256: str
    episodes: tuple[AuditedEpisode, ...]
    summary: dict[str, Any]


def verify_schedule(
    simulation_root: Path,
    *,
    expected_counts: dict[str, int] = EXPECTED_COUNTS,
    expected_schedule_sha256: str | None = EXPECTED_SCHEDULE_SHA256,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    schedule_path = simulation_root / "schedule.json"
    schedule = load_json(schedule_path)
    declared = schedule.get("schedule_sha256_excludes_self")
    payload = copy.deepcopy(schedule)
    payload.pop("schedule_sha256_excludes_self", None)
    actual = sha256_bytes(canonical_bytes(payload))
    if declared != actual:
        raise ValueError(f"schedule self-hash mismatch: {declared} != {actual}")
    if expected_schedule_sha256 is not None and actual != expected_schedule_sha256:
        raise ValueError(f"unexpected schedule identity: {actual}")

    selected = [
        entry for entry in schedule["episodes"] if entry["split"] in expected_counts
    ]
    counts = Counter(entry["split"] for entry in selected)
    if counts != Counter(expected_counts):
        raise ValueError(f"train/validation counts drifted: {dict(counts)}")
    selected.sort(key=lambda entry: int(entry["ordinal"]))
    if [int(entry["ordinal"]) for entry in selected] != list(range(len(selected))):
        raise ValueError("train/validation entries are not the frozen ordinal prefix")
    if {entry["condition"] for entry in selected} != {"C0"}:
        raise ValueError("train/validation handoff must contain C0 only")
    return schedule, selected, sha256_file(schedule_path)


def audit_episode(simulation_root: Path, entry: dict[str, Any]) -> AuditedEpisode:
    episode_id = entry["episode_id"]
    episode_root = simulation_root / "episodes" / episode_id
    success_path = episode_root / "SUCCESS.json"
    success = load_json(success_path)
    if success.get("status") != "accepted":
        raise ValueError(f"{episode_id}: success marker is not accepted")
    if success.get("episode_id") != episode_id:
        raise ValueError(f"{episode_id}: success marker identity mismatch")
    if success.get("entry_sha256") != entry["entry_sha256"]:
        raise ValueError(f"{episode_id}: success marker belongs to another schedule")
    attempt_name = success.get("accepted_attempt", "")
    if not (
        isinstance(attempt_name, str)
        and attempt_name.startswith("attempt_")
        and attempt_name[8:].isdigit()
    ):
        raise ValueError(f"{episode_id}: unsafe accepted-attempt name")
    attempt_root = episode_root / attempt_name
    attempt_path = attempt_root / "ATTEMPT.json"
    if sha256_file(attempt_path) != success.get("attempt_manifest_sha256"):
        raise ValueError(f"{episode_id}: accepted attempt hash mismatch")
    attempt = load_json(attempt_path)
    if attempt.get("status") != "accepted" or int(attempt.get("return_code", -1)) != 0:
        raise ValueError(f"{episode_id}: accepted attempt did not finish cleanly")
    if attempt.get("entry_sha256") != entry["entry_sha256"]:
        raise ValueError(f"{episode_id}: attempt belongs to another schedule")

    export_root = attempt_root / "export"
    export_manifest_path = export_root / "manifest.json"
    export_manifest_hash = sha256_file(export_manifest_path)
    if attempt["sha256"].get("export/manifest.json") != export_manifest_hash:
        raise ValueError(f"{episode_id}: export manifest hash mismatch")
    manifest = load_json(export_manifest_path)
    if manifest.get("manifest_sha256_excludes_self") != manifest_self_hash(
        manifest, "manifest_sha256_excludes_self"
    ):
        raise ValueError(f"{episode_id}: export manifest self-hash mismatch")
    required = {
        "export_schema_version": EXPECTED_EXPORT_SCHEMA,
        "run_id": episode_id,
        "environment_id": entry["world_name"],
        "domain": "simulation",
        "view": "policy",
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"{episode_id}: {field} drifted")
    if manifest["effective_configuration"].get("lidar_causal") is not True:
        raise ValueError(f"{episode_id}: export is not causally associated")
    if int(manifest["contiguity"].get("windowable_k8_h8", 0)) <= 0:
        raise ValueError(f"{episode_id}: export has no K=8/H=8 windows")

    output_records: dict[str, dict[str, Any]] = manifest["outputs"]
    expected_files = {"manifest.json", *output_records}
    observed_files = {
        path.relative_to(export_root).as_posix()
        for path in export_root.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise ValueError(
            f"{episode_id}: unexpected export files: "
            f"missing={sorted(expected_files - observed_files)}, "
            f"extra={sorted(observed_files - expected_files)}"
        )
    normalized_outputs: dict[str, dict[str, Any]] = {}
    for name, expected in sorted(output_records.items()):
        if Path(name).name != name:
            raise ValueError(f"{episode_id}: unsafe output name {name}")
        path = export_root / name
        size = path.stat().st_size
        if size != int(expected["size_bytes"]):
            raise ValueError(f"{episode_id}: {name} size mismatch")
        digest = sha256_file(path)
        if digest != expected["sha256"]:
            raise ValueError(f"{episode_id}: {name} hash mismatch")
        normalized_outputs[name] = {"size_bytes": size, "sha256": digest}

    record = {
        "episode_id": episode_id,
        "ordinal": int(entry["ordinal"]),
        "split": entry["split"],
        "world_name": entry["world_name"],
        "world_index": int(entry["world_index"]),
        "world_seed": int(entry["world_seed"]),
        "episode_index": int(entry["episode_index"]),
        "observation_seed": int(entry["observation_seed"]),
        "condition": entry["condition"],
        "entry_sha256": entry["entry_sha256"],
        "success_sha256": sha256_file(success_path),
        "accepted_attempt": attempt_name,
        "attempt_manifest_sha256": success["attempt_manifest_sha256"],
        "export_manifest_sha256": export_manifest_hash,
        "source_tree_sha256": manifest["code"]["source_tree_sha256"],
        "effective_configuration_sha256": manifest["effective_configuration_sha256"],
        "accepted_samples": int(manifest["counts"]["accepted_samples"]),
        "windowable_k8_h8": int(manifest["contiguity"]["windowable_k8_h8"]),
        "outputs": normalized_outputs,
    }
    return AuditedEpisode(entry=entry, export_root=export_root, record=record)


def audit_handoff(
    simulation_root: Path,
    *,
    expected_counts: dict[str, int] = EXPECTED_COUNTS,
    expected_schedule_sha256: str | None = EXPECTED_SCHEDULE_SHA256,
) -> Audit:
    simulation_root = simulation_root.resolve()
    schedule, selected, schedule_file_sha = verify_schedule(
        simulation_root,
        expected_counts=expected_counts,
        expected_schedule_sha256=expected_schedule_sha256,
    )
    audited = tuple(audit_episode(simulation_root, entry) for entry in selected)
    by_split = Counter(item.record["split"] for item in audited)
    by_world = Counter(item.record["world_name"] for item in audited)
    source_trees = sorted({item.record["source_tree_sha256"] for item in audited})
    configurations = sorted(
        {item.record["effective_configuration_sha256"] for item in audited}
    )
    if len(source_trees) != 1:
        raise ValueError(f"multiple export source trees observed: {source_trees}")
    if len(configurations) != 1:
        raise ValueError(f"multiple effective export configurations: {configurations}")
    total_bytes = sum(
        output["size_bytes"]
        for item in audited
        for output in item.record["outputs"].values()
    )
    summary = {
        "episodes": len(audited),
        "by_split": dict(sorted(by_split.items())),
        "by_world": dict(sorted(by_world.items())),
        "accepted_samples": sum(item.record["accepted_samples"] for item in audited),
        "windowable_k8_h8": sum(item.record["windowable_k8_h8"] for item in audited),
        "export_payload_bytes": total_bytes,
        "export_payload_gib": round(total_bytes / 1024**3, 3),
        "source_tree_sha256": source_trees[0],
        "effective_configuration_sha256": configurations[0],
    }
    return Audit(
        schedule=schedule,
        schedule_file_sha256=schedule_file_sha,
        episodes=audited,
        summary=summary,
    )


def shard_groups(audit: Audit) -> dict[str, tuple[AuditedEpisode, ...]]:
    grouped: dict[str, list[AuditedEpisode]] = defaultdict(list)
    for episode in audit.episodes:
        grouped[episode.record["world_name"]].append(episode)
    return {
        world: tuple(sorted(episodes, key=lambda item: item.record["ordinal"]))
        for world, episodes in sorted(grouped.items())
    }


def shard_manifest(
    audit: Audit, world_name: str, episodes: tuple[AuditedEpisode, ...]
) -> dict[str, Any]:
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
        "handoff": "livifuser_confirmatory_v3_train_val_v1",
        "schedule_sha256_excludes_self": audit.schedule[
            "schedule_sha256_excludes_self"
        ],
        "world_name": world_name,
        "split": episodes[0].record["split"],
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
    with source.open("rb") as input_handle, archive.open(zip_info(member), "w") as output:
        shutil.copyfileobj(input_handle, output, length=8 * 1024 * 1024)


def build_shard(
    output_root: Path,
    audit: Audit,
    world_name: str,
    episodes: tuple[AuditedEpisode, ...],
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    archive_path = output_root / f"{world_name}.zip"
    sidecar_path = output_root / f"{world_name}.zip.sha256"
    manifest = shard_manifest(audit, world_name, episodes)
    if archive_path.exists():
        if not sidecar_path.is_file():
            raise FileExistsError(f"unverified existing shard: {archive_path}")
        expected = sidecar_path.read_text(encoding="ascii").strip().split()[0].upper()
        actual = sha256_file(archive_path)
        if expected != actual:
            raise ValueError(f"existing shard hash mismatch: {archive_path.name}")
        return {
            "filename": archive_path.name,
            "sha256": actual,
            "size_bytes": archive_path.stat().st_size,
            "episode_count": len(episodes),
            "world_name": world_name,
            "split": episodes[0].record["split"],
            "status": "complete",
        }

    partial = archive_path.with_suffix(".zip.partial")
    if partial.exists():
        raise FileExistsError(f"remove interrupted partial before retrying: {partial}")
    with zipfile.ZipFile(
        partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as archive:
        archive.writestr(zip_info("SHARD_MANIFEST.json"), json_bytes(manifest))
        for episode in episodes:
            base = f"episodes/{episode.record['episode_id']}/export"
            add_file(archive, episode.export_root / "manifest.json", f"{base}/manifest.json")
            for name in sorted(episode.record["outputs"]):
                add_file(archive, episode.export_root / name, f"{base}/{name}")
    partial.replace(archive_path)
    digest = sha256_file(archive_path)
    sidecar_path.write_text(f"{digest}  {archive_path.name}\n", encoding="ascii")
    return {
        "filename": archive_path.name,
        "sha256": digest,
        "size_bytes": archive_path.stat().st_size,
        "episode_count": len(episodes),
        "world_name": world_name,
        "split": episodes[0].record["split"],
        "status": "complete",
    }


def write_handoff_index(
    output_root: Path, audit: Audit, shard_records: list[dict[str, Any]]
) -> dict[str, Any]:
    value = {
        "schema_version": HANDOFF_SCHEMA,
        "name": "livifuser_confirmatory_v3_train_val_v1",
        "source": {
            "schedule_sha256_excludes_self": audit.schedule[
                "schedule_sha256_excludes_self"
            ],
            "schedule_file_sha256": audit.schedule_file_sha256,
            "recollection_freeze_manifest": audit.schedule[
                "recollection_freeze_manifest"
            ],
            "recollection_freeze_manifest_sha256": audit.schedule[
                "recollection_freeze_manifest_sha256"
            ],
        },
        "selection": {
            "splits": ["train", "val_id"],
            "conditions": ["C0"],
            "ordinal_range_inclusive": [0, 149],
            "raw_mcap_included": False,
            "accepted_attempt_only": True,
        },
        "audit": audit.summary,
        "episodes": [episode.record for episode in audit.episodes],
        "shards": sorted(shard_records, key=lambda item: item["filename"]),
    }
    value["manifest_sha256_excludes_self"] = sha256_bytes(canonical_bytes(value))
    path = output_root / "handoff_manifest.json"
    path.write_bytes(json_bytes(value))
    (output_root / "README.txt").write_text(
        "LiViFuser confirmatory-v3 train/validation GPU handoff.\n"
        "Raw MCAP bags are deliberately excluded. Verify every ZIP with its .sha256 "
        "sidecar, then verify extracted members against SHARD_MANIFEST.json before "
        "DINO feature extraction. Train and validation must remain separate.\n",
        encoding="utf-8",
    )
    return value


def verify_archives(output_root: Path, index: dict[str, Any], *, deep: bool) -> None:
    for shard in index["shards"]:
        path = output_root / shard["filename"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"archive hash mismatch: {path.name}")
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            manifest = json.loads(archive.read("SHARD_MANIFEST.json"))
            expected = {"SHARD_MANIFEST.json", *(item["path"] for item in manifest["members"])}
            if names != expected:
                raise ValueError(f"archive member set mismatch: {path.name}")
            declared = manifest.pop("manifest_sha256_excludes_self")
            if sha256_bytes(canonical_bytes(manifest)) != declared:
                raise ValueError(f"embedded manifest self-hash mismatch: {path.name}")
            if deep:
                for member in manifest["members"]:
                    if sha256_bytes(archive.read(member["path"])) != member["sha256"]:
                        raise ValueError(
                            f"archive member hash mismatch: {path.name}:{member['path']}"
                        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-root", type=Path, default=DEFAULT_SIMULATION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--deep", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    audit = audit_handoff(args.simulation_root)
    print(json.dumps({"audit": audit.summary}, indent=2))
    if not args.build and not args.verify:
        return 0

    output_root = args.output.resolve()
    groups = shard_groups(audit)
    records = []
    selected_groups = list(groups.items())
    if args.limit is not None:
        selected_groups = selected_groups[: args.limit]
    if args.build:
        for index, (world, episodes) in enumerate(selected_groups, start=1):
            print(f"[{index}/{len(selected_groups)}] {world}", flush=True)
            record = build_shard(output_root, audit, world, episodes)
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
                            sum(item["size_bytes"] for item in records) / 1024**3, 3
                        ),
                        "manifest_sha256_excludes_self": index[
                            "manifest_sha256_excludes_self"
                        ],
                    },
                    indent=2,
                )
            )
        else:
            print(json.dumps({"complete": False, "built_this_run": len(records)}, indent=2))
    if args.verify:
        index = load_json(output_root / "handoff_manifest.json")
        declared = index.pop("manifest_sha256_excludes_self")
        if sha256_bytes(canonical_bytes(index)) != declared:
            raise ValueError("handoff index self-hash mismatch")
        verify_archives(
            output_root,
            {**index, "manifest_sha256_excludes_self": declared},
            deep=args.deep,
        )
        print(json.dumps({"verified": True, "deep": args.deep}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
