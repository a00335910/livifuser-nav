#!/usr/bin/env python3
"""Repair accepted confirmatory exports whose post-close hashes drifted.

The source MCAP and original attempt are immutable.  A repair creates a new
attempt, hard-links the same bag, independently re-exports it, proves that all
derived files reproduce the existing bytes, and atomically advances SUCCESS.
Default mode is a write-free audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPOSITORY_ROOT / "artifacts/simulation/confirmatory_v3"
EXPECTED_SCHEDULE_SHA256 = (
    "85F48F81D52F82A5CD64D14D704E032617582D7F5462A7B6FD46C5E64899047D"
)
OUTPUT_NAMES = (
    "rgb_320x240_rgb8.npy",
    "scan_ranges.npy",
    "vectors.npz",
    "samples.jsonl",
    "rejections.jsonl",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def manifest_self_hash(manifest: dict[str, Any]) -> str:
    value = dict(manifest)
    value.pop("manifest_sha256_excludes_self", None)
    return sha256_bytes(json.dumps(value, sort_keys=True).encode("utf-8"))


def actual_output_records(export: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in OUTPUT_NAMES:
        path = export / name
        if not path.is_file():
            raise ValueError(f"missing export output: {path}")
        records[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return records


def output_mismatches(export: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    actual = actual_output_records(export)
    mismatches = []
    declared = manifest.get("outputs", {})
    for name in OUTPUT_NAMES:
        if declared.get(name) != actual[name]:
            mismatches.append(
                {"file": name, "declared": declared.get(name), "actual": actual[name]}
            )
    return mismatches


def load_schedule(root: Path) -> dict[str, Any]:
    schedule = load_json(root / "schedule.json")
    declared = schedule.pop("schedule_sha256_excludes_self", None)
    actual = sha256_bytes(canonical_bytes(schedule))
    schedule["schedule_sha256_excludes_self"] = declared
    if declared != EXPECTED_SCHEDULE_SHA256 or actual != declared:
        raise ValueError("confirmatory-v3 schedule hash mismatch")
    return schedule


def accepted_attempt(root: Path, entry: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    episode = root / "episodes" / entry["episode_id"]
    success_path = episode / "SUCCESS.json"
    success = load_json(success_path)
    if success.get("status") != "accepted":
        raise ValueError(f"{entry['episode_id']}: not accepted")
    if success.get("entry_sha256") != entry["entry_sha256"]:
        raise ValueError(f"{entry['episode_id']}: SUCCESS belongs to another schedule")
    attempt = episode / str(success["accepted_attempt"])
    attempt_path = attempt / "ATTEMPT.json"
    if sha256_file(attempt_path) != success.get("attempt_manifest_sha256"):
        raise ValueError(f"{entry['episode_id']}: SUCCESS-to-ATTEMPT hash mismatch")
    record = load_json(attempt_path)
    manifest_path = attempt / "export/manifest.json"
    if sha256_file(manifest_path) != record.get("sha256", {}).get("export/manifest.json"):
        raise ValueError(f"{entry['episode_id']}: ATTEMPT-to-manifest hash mismatch")
    manifest = load_json(manifest_path)
    if manifest_self_hash(manifest) != manifest.get("manifest_sha256_excludes_self"):
        raise ValueError(f"{entry['episode_id']}: manifest self-hash mismatch")
    return attempt, success


def audit(root: Path, ordinal_stop: int) -> list[dict[str, Any]]:
    schedule = load_schedule(root)
    findings = []
    for entry in schedule["episodes"]:
        if int(entry["ordinal"]) >= ordinal_stop:
            continue
        attempt, _success = accepted_attempt(root, entry)
        manifest = load_json(attempt / "export/manifest.json")
        mismatches = output_mismatches(attempt / "export", manifest)
        if mismatches:
            findings.append(
                {
                    "ordinal": entry["ordinal"],
                    "episode_id": entry["episode_id"],
                    "attempt": attempt.name,
                    "mismatches": mismatches,
                }
            )
    return findings


def next_attempt_dir(episode: Path) -> Path:
    numbers = []
    for path in episode.glob("attempt_*"):
        try:
            numbers.append(int(path.name.removeprefix("attempt_")))
        except ValueError:
            pass
    return episode / f"attempt_{max(numbers, default=0) + 1:03d}"


def hardlink_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)


def to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    relative = resolved.relative_to(resolved.anchor).as_posix()
    return f"/mnt/{drive}/{relative}"


def run_export(
    distribution: str, bag: Path, output: Path, entry: dict[str, Any]
) -> subprocess.CompletedProcess[str]:
    command = [
        "source /opt/ros/humble/setup.bash",
        f"source {shlex.quote('/mnt/d/LiViFuser/ros2_ws/install/setup.bash')}",
        "python3 "
        + " ".join(
            shlex.quote(value)
            for value in (
                "/mnt/d/LiViFuser/scripts/export_pilot_dataset.py",
                to_wsl_path(bag),
                "--output",
                to_wsl_path(output),
                "--environment-id",
                str(entry["world_name"]),
                "--run-id",
                str(entry["episode_id"]),
                "--domain",
                "simulation",
                "--view",
                "policy",
                "--lidar-causal",
            )
        ),
    ]
    return subprocess.run(
        ["wsl.exe", "-d", distribution, "--", "bash", "-lc", " && ".join(command)],
        cwd=REPOSITORY_ROOT,
        check=False,
        text=True,
    )


def repair_one(root: Path, entry: dict[str, Any], distribution: str) -> dict[str, Any]:
    source_attempt, source_success = accepted_attempt(root, entry)
    source_export = source_attempt / "export"
    source_manifest_path = source_export / "manifest.json"
    source_manifest = load_json(source_manifest_path)
    source_mismatches = output_mismatches(source_export, source_manifest)
    if not source_mismatches:
        return {"episode_id": entry["episode_id"], "status": "already_clean"}

    episode = source_attempt.parent
    target = next_attempt_dir(episode)
    target.mkdir(parents=False, exist_ok=False)
    started = time.time()
    try:
        hardlink_tree(source_attempt / "bag", target / "bag")
        for name in ("verify.json", "runtime.json"):
            shutil.copy2(source_attempt / name, target / name)
        (target / "SOURCE_SUCCESS.json").write_bytes(json_bytes(source_success))
        completed = run_export(distribution, target / "bag", target / "export", entry)
        if completed.returncode != 0:
            raise RuntimeError(f"offline exporter returned {completed.returncode}")

        fresh_manifest_path = target / "export/manifest.json"
        fresh_manifest = load_json(fresh_manifest_path)
        post_close_corrections = output_mismatches(target / "export", fresh_manifest)
        if post_close_corrections:
            fresh_manifest["outputs"] = actual_output_records(target / "export")
            fresh_manifest["manifest_sha256_excludes_self"] = manifest_self_hash(
                fresh_manifest
            )
            fresh_manifest_path.write_bytes(json_bytes(fresh_manifest))

        if output_mismatches(target / "export", load_json(fresh_manifest_path)):
            raise ValueError("repaired manifest still disagrees with its outputs")
        source_outputs = actual_output_records(source_export)
        repaired_outputs = actual_output_records(target / "export")
        if source_outputs != repaired_outputs:
            raise ValueError("independent re-export did not reproduce source output bytes")
        if fresh_manifest["effective_configuration_sha256"] != source_manifest[
            "effective_configuration_sha256"
        ]:
            raise ValueError("repair changed the effective export configuration")
        if fresh_manifest["code"]["source_tree_sha256"] != source_manifest["code"][
            "source_tree_sha256"
        ]:
            raise ValueError("repair changed the export source tree")

        repair = {
            "schema_version": "1.0.0",
            "reason": "post_close_export_output_hash_mismatch",
            "episode_id": entry["episode_id"],
            "source_attempt": source_attempt.name,
            "source_success_sha256": sha256_bytes(json_bytes(source_success)),
            "source_attempt_sha256": sha256_file(source_attempt / "ATTEMPT.json"),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_mismatches": source_mismatches,
            "independent_reexport_byte_identical": True,
            "post_close_manifest_corrections": post_close_corrections,
        }
        (target / "REPAIR.json").write_bytes(json_bytes(repair))
        mcap_files = sorted((target / "bag").glob("*.mcap"))
        if len(mcap_files) != 1:
            raise ValueError("repair attempt must contain exactly one MCAP")
        hashes = {
            "verify.json": sha256_file(target / "verify.json"),
            "runtime.json": sha256_file(target / "runtime.json"),
            "export/manifest.json": sha256_file(fresh_manifest_path),
            "bag/metadata.yaml": sha256_file(target / "bag/metadata.yaml"),
            f"bag/{mcap_files[0].name}": sha256_file(mcap_files[0]),
            "REPAIR.json": sha256_file(target / "REPAIR.json"),
            "SOURCE_SUCCESS.json": sha256_file(target / "SOURCE_SUCCESS.json"),
        }
        attempt_record = {
            "schema_version": "1.0.0",
            "status": "accepted",
            "repair": True,
            "episode_id": entry["episode_id"],
            "entry_sha256": entry["entry_sha256"],
            "started_unix_sec": started,
            "finished_unix_sec": time.time(),
            "return_code": 0,
            "sha256": hashes,
        }
        (target / "ATTEMPT.json").write_bytes(json_bytes(attempt_record))
        success = {
            "schema_version": "1.0.0",
            "status": "accepted",
            "episode_id": entry["episode_id"],
            "entry_sha256": entry["entry_sha256"],
            "accepted_attempt": target.name,
            "attempt_manifest_sha256": sha256_file(target / "ATTEMPT.json"),
        }
        temporary = episode / f"SUCCESS.repairing.{os.getpid()}.json"
        temporary.write_bytes(json_bytes(success))
        os.replace(temporary, episode / "SUCCESS.json")
        return {
            "episode_id": entry["episode_id"],
            "status": "repaired",
            "source_attempt": source_attempt.name,
            "accepted_attempt": target.name,
            "post_close_corrections": len(post_close_corrections),
        }
    except Exception as error:
        failure = {
            "schema_version": "1.0.0",
            "status": "repair_failed",
            "episode_id": entry["episode_id"],
            "entry_sha256": entry["entry_sha256"],
            "started_unix_sec": started,
            "finished_unix_sec": time.time(),
            "error": str(error),
        }
        (target / "ATTEMPT.json").write_bytes(json_bytes(failure))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--ordinal-stop", type=int, default=150)
    parser.add_argument("--distribution", default="Ubuntu-TB3")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    findings = audit(args.root, args.ordinal_stop)
    print(json.dumps({"episodes_audited": args.ordinal_stop, "findings": findings}, indent=2))
    if not args.repair:
        return 1 if findings else 0

    schedule = load_schedule(args.root)
    entries = {entry["episode_id"]: entry for entry in schedule["episodes"]}
    selected = findings[: args.limit] if args.limit is not None else findings
    results = []
    for index, finding in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] repairing {finding['episode_id']}", flush=True)
        results.append(repair_one(args.root, entries[finding["episode_id"]], args.distribution))
        print(json.dumps(results[-1], indent=2), flush=True)
    remaining = audit(args.root, args.ordinal_stop)
    print(json.dumps({"repaired": results, "remaining_findings": remaining}, indent=2))
    return 1 if remaining and args.limit is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
