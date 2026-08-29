#!/usr/bin/env python3
"""Plan, prepare, and resume the frozen 260-episode Gazebo recollection.

The default mode is a write-free dry run. ``--prepare`` writes the immutable
schedule and world assets but launches nothing. ``--run`` requires that exact
prepared schedule, enforces the disk-space gate, and runs only episodes without
an accepted SUCCESS marker. Failed attempts are preserved in numbered folders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_sim.visual_skin import verify_installed_visual_assets  # noqa: E402
from livifuser_sim.world_generator import (  # noqa: E402
    derive_condition,
    generate_world,
)
from livifuser_sim.world_sdf import render_world_sdf  # noqa: E402

DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "simulation_confirmatory_batch_v3.json"
TEMPLATE = PACKAGE_ROOT / "worlds" / "livifuser_lab.sdf"
EPISODE_SCRIPT = REPOSITORY_ROOT / "scripts" / "run_confirmatory_sim_episode.sh"
GIB = 1024**3


@dataclass(frozen=True)
class Plan:
    schedule: dict[str, Any]
    assets: dict[str, bytes]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def verify_frozen_inputs(config: dict[str, Any]) -> tuple[dict, dict]:
    manifest_path = REPOSITORY_ROOT / config["freeze_manifest"]
    manifest = load_json(manifest_path)
    if manifest.get("status") != "frozen_pre_recollection":
        raise ValueError("replacement-collection manifest is not frozen")
    if config["base_freeze_commit"] != manifest["base_freeze_commit"]:
        raise ValueError("batch config does not name the base freeze commit")
    for relative, expected in manifest["sha256"].items():
        actual = sha256_file(REPOSITORY_ROOT / relative)
        if actual != expected:
            raise ValueError(
                f"frozen replacement input drift: {relative}: {actual} != {expected}"
            )

    amendments: dict[int, dict] = {}
    for record in config["amendments"]:
        number = int(record["number"])
        path = REPOSITORY_ROOT / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Amendment {number} checksum does not match the config")
        amendment = load_json(path)
        if int(amendment.get("amendment", -1)) != number:
            raise ValueError(f"Amendment {number} identity does not match its record")
        amendments[number] = amendment
    if set(amendments) != {1, 2, 3, 4}:
        raise ValueError(
            "replacement collection requires Amendments 1, 2, 3, and 4"
        )
    amendment = amendments[1]
    if amendment.get("status") != "adopted_pre_results":
        raise ValueError("Amendment 1 is not adopted pre-results")
    if amendments[2].get("status") != "adopted_post_results_before_recollection":
        raise ValueError("Amendment 2 does not invalidate the defective acquisition")
    if amendments[3].get("status") != "adopted_before_replacement_recollection":
        raise ValueError("Amendment 3 does not freeze the replacement visual skin")
    if not amendments[2]["invalidated_collection"]["whole_collection_excluded"]:
        raise ValueError("Amendment 2 must exclude the whole predecessor collection")
    if not amendments[3]["invalidated_collection"]["reuse_forbidden"]:
        raise ValueError("Amendment 3 must forbid predecessor episode reuse")
    if (
        amendments[4].get("status")
        != "adopted_pre_results_after_operational_failure"
    ):
        raise ValueError("Amendment 4 does not freeze the COLLADA repair")
    if not amendments[4]["failed_collection"]["whole_collection_excluded"]:
        raise ValueError("Amendment 4 must exclude the failed v2 collection")
    if not amendments[4]["failed_collection"]["reuse_forbidden"]:
        raise ValueError("Amendment 4 must forbid v2 episode reuse")
    if amendment["c3"]["closed_loop_condition"] != "C3b":
        raise ValueError("Amendment 1 must bind closed-loop C3 to C3b")
    if float(amendment["deadlines"]["scientific_simulated_seconds"]) != float(
        config["scientific_simulated_deadline_sec"]
    ):
        raise ValueError("scientific deadline differs from Amendment 1")
    required_topics = {
        "/camera/image_raw",
        "/camera/camera_info",
        "/scan",
        "/odom",
        "/livifuser/sim/ground_truth/odom",
        "/livifuser/goal_relative",
        "/livifuser/cmd_vel_stamped",
        "/tf",
        "/tf_static",
        "/clock",
    }
    if set(config["topics"]) != required_topics:
        raise ValueError("replacement topic contract is incomplete or unexpected")
    if config["output_root"] in config["forbidden_predecessor_roots"]:
        raise ValueError("replacement output root aliases an invalidated predecessor")
    visual_assets = verify_installed_visual_assets()
    if not visual_assets["valid"]:
        raise ValueError(f"frozen visual assets drifted: {visual_assets['issues']}")
    return manifest, amendment


def _variant(payload: dict, name: str) -> dict:
    return payload if name == "C0" else derive_condition(payload, name)


def _asset_bytes(payload: dict) -> tuple[bytes, bytes]:
    return (
        (json.dumps(payload, indent=2) + "\n").encode("utf-8"),
        render_world_sdf(payload, TEMPLATE).encode("utf-8"),
    )


def _asset_suffix(variant: str) -> str:
    return "" if variant == "C0" else f".{variant}"


def _entry_hash(entry: dict[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(entry))


def build_plan(config: dict[str, Any]) -> Plan:
    manifest, amendment = verify_frozen_inputs(config)
    groups = config["groups"]
    implementations = config["condition_implementation"]
    seed_blocks = config["observation_seed_blocks"]
    stride = int(config["observation_seed_world_stride"])

    worlds: dict[tuple[str, int], dict] = {}
    assets: dict[str, bytes] = {}
    asset_records: list[dict[str, str]] = []
    for group in ("train", "val_id", "test_id"):
        for world_index in range(int(groups[group]["worlds"])):
            payload = generate_world(group, world_index)
            worlds[(group, world_index)] = payload
            variants = ("C0", "C1", "C4") if group == "test_id" else ("C0",)
            for variant_name in variants:
                value = _variant(payload, variant_name)
                suffix = _asset_suffix(variant_name)
                stem = f"worlds/{payload['name']}{suffix}"
                json_payload, sdf_payload = _asset_bytes(value)
                for relative, content in (
                    (f"{stem}.json", json_payload),
                    (f"{stem}.sdf", sdf_payload),
                ):
                    assets[relative] = content
                    asset_records.append(
                        {"path": relative, "sha256": sha256_bytes(content)}
                    )

    entries: list[dict[str, Any]] = []
    ordinal = 0
    for group in ("train", "val_id", "test_id", "test_ood"):
        group_config = groups[group]
        source_group = str(group_config.get("source_group", group))
        for world_index in range(int(group_config["worlds"])):
            payload = worlds[(source_group, world_index)]
            for condition in group_config["conditions"]:
                implementation = implementations[condition]
                variant_name = str(implementation["world_variant"])
                suffix = _asset_suffix(variant_name)
                stem = f"worlds/{payload['name']}{suffix}"
                for episode_index in range(int(group_config["episodes_per_world"])):
                    observation_seed = (
                        int(seed_blocks[group]) + world_index * stride + episode_index
                    )
                    implementation_name = str(
                        implementation["lidar_condition"]
                        if condition == "C3"
                        else condition
                    ).lower()
                    episode_id = (
                        f"{group}_{payload['archetype']}_{world_index:03d}_"
                        f"{implementation_name}_e{episode_index:03d}_"
                        f"s{observation_seed}"
                    )
                    entry = {
                        "ordinal": ordinal,
                        "episode_id": episode_id,
                        "split": group,
                        "source_world_group": source_group,
                        "world_index": world_index,
                        "world_name": payload["name"],
                        "world_seed": payload["seed"],
                        "condition": condition,
                        "world_variant": variant_name,
                        "lidar_condition": implementation["lidar_condition"],
                        "episode_index": episode_index,
                        "observation_seed": observation_seed,
                        "world_json": f"{stem}.json",
                        "world_sdf": f"{stem}.sdf",
                        "scientific_simulated_deadline_sec": float(
                            config["scientific_simulated_deadline_sec"]
                        ),
                        "operational_wall_watchdog_sec": float(
                            config["operational_wall_watchdog_sec"]
                        ),
                    }
                    entry["entry_sha256"] = _entry_hash(entry)
                    entries.append(entry)
                    ordinal += 1

    _validate_entries(entries, manifest, amendment)
    schedule = {
        "schema_version": "3.0.0",
        "name": config["name"],
        "base_freeze_commit": config["base_freeze_commit"],
        "recollection_freeze_manifest": config["freeze_manifest"],
        "recollection_freeze_manifest_sha256": sha256_file(
            REPOSITORY_ROOT / config["freeze_manifest"]
        ),
        "amendments": config["amendments"],
        "invalidated_predecessor_roots": config["forbidden_predecessor_roots"],
        "batch_config_sha256": sha256_file(DEFAULT_CONFIG),
        "batch_runner_sha256": sha256_file(Path(__file__)),
        "episode_runner_sha256": sha256_file(EPISODE_SCRIPT),
        "counts": {
            "episodes": len(entries),
            "by_split": dict(Counter(entry["split"] for entry in entries)),
            "by_condition": dict(Counter(entry["condition"] for entry in entries)),
        },
        "assets": sorted(asset_records, key=lambda item: item["path"]),
        "episodes": entries,
    }
    schedule["schedule_sha256_excludes_self"] = sha256_bytes(
        canonical_bytes(schedule)
    )
    return Plan(schedule=schedule, assets=assets)


def _validate_entries(entries: list[dict], manifest: dict, amendment: dict) -> None:
    frozen = manifest["frozen_design"]["episodes"]
    if len(entries) != int(frozen["total_confirmatory"]):
        raise ValueError("schedule does not contain the frozen 260 episodes")
    expected_splits = {
        "train": int(frozen["train"]),
        "val_id": int(frozen["val_id"]),
        "test_id": int(frozen["test_id"]),
        "test_ood": int(frozen["test_ood"]),
    }
    if Counter(entry["split"] for entry in entries) != Counter(expected_splits):
        raise ValueError("schedule split counts differ from the freeze")
    if len({entry["episode_id"] for entry in entries}) != len(entries):
        raise ValueError("episode IDs are not unique")
    if [entry["ordinal"] for entry in entries] != list(range(len(entries))):
        raise ValueError("episode ordinals are not contiguous")

    ood = [entry for entry in entries if entry["split"] == "test_ood"]
    expected_ood = {"C0": 20, "C1": 20, "C3": 20, "C4": 20}
    if Counter(entry["condition"] for entry in ood) != Counter(expected_ood):
        raise ValueError("OOD condition counts differ from the frozen grid")
    if any(
        entry["lidar_condition"] != amendment["c3"]["closed_loop_condition"]
        for entry in ood
        if entry["condition"] == "C3"
    ):
        raise ValueError("closed-loop C3 schedule is not amendment-bound C3b")

    paired: dict[tuple[int, int], list[dict]] = {}
    for entry in ood:
        paired.setdefault((entry["world_index"], entry["episode_index"]), []).append(
            entry
        )
    if len(paired) != 20:
        raise ValueError("OOD schedule lacks 20 matched world/episode pairs")
    for values in paired.values():
        if {entry["condition"] for entry in values} != {"C0", "C1", "C3", "C4"}:
            raise ValueError("an OOD pair lacks a condition")
        for field in (
            "source_world_group",
            "world_index",
            "world_name",
            "world_seed",
            "episode_index",
            "observation_seed",
        ):
            if len({entry[field] for entry in values}) != 1:
                raise ValueError(f"paired OOD field differs across conditions: {field}")

    non_ood = [entry for entry in entries if entry["split"] != "test_ood"]
    non_ood_seeds = {entry["observation_seed"] for entry in non_ood}
    ood_seeds = {entry["observation_seed"] for entry in ood}
    if non_ood_seeds & ood_seeds:
        raise ValueError("observation seed blocks overlap")


def write_immutable(path: Path, payload: bytes) -> str:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace non-matching artifact: {path}")
        return "verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return "written"


def prepare(plan: Plan, output_root: Path) -> dict[str, int]:
    counts = Counter()
    counts[write_immutable(output_root / "schedule.json", json_bytes(plan.schedule))] += 1
    for relative, payload in plan.assets.items():
        counts[write_immutable(output_root / relative, payload)] += 1
    return dict(counts)


def validate_prepared(plan: Plan, output_root: Path) -> None:
    schedule_path = output_root / "schedule.json"
    if not schedule_path.is_file() or schedule_path.read_bytes() != json_bytes(
        plan.schedule
    ):
        raise ValueError("prepared schedule is absent or differs from this runner")
    for relative, payload in plan.assets.items():
        path = output_root / relative
        if not path.is_file() or sha256_file(path) != sha256_bytes(payload):
            raise ValueError(f"prepared world asset is absent or drifted: {relative}")


def nearest_existing(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"no existing parent for {path}")
        candidate = candidate.parent
    return candidate


def free_gib(path: Path) -> float:
    return shutil.disk_usage(nearest_existing(path)).free / GIB


def to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    drive = resolved.drive.rstrip(":").lower()
    relative = resolved.relative_to(resolved.anchor).as_posix()
    return f"/mnt/{drive}/{relative}"


def accepted_success(entry: dict, episode_dir: Path) -> bool:
    marker = episode_dir / "SUCCESS.json"
    if not marker.is_file():
        return False
    value = load_json(marker)
    if value.get("status") != "accepted":
        raise ValueError(f"invalid success marker status: {marker}")
    if value.get("entry_sha256") != entry["entry_sha256"]:
        raise ValueError(f"success marker belongs to a different schedule: {marker}")
    return True


def next_attempt_dir(episode_dir: Path) -> Path:
    existing = []
    if episode_dir.is_dir():
        for path in episode_dir.glob("attempt_*"):
            try:
                existing.append(int(path.name.removeprefix("attempt_")))
            except ValueError:
                continue
    return episode_dir / f"attempt_{max(existing, default=0) + 1:03d}"


def _validate_attempt(attempt_dir: Path, entry: dict) -> dict[str, str]:
    verify_path = attempt_dir / "verify.json"
    runtime_path = attempt_dir / "runtime.json"
    export_manifest_path = attempt_dir / "export" / "manifest.json"
    bag_metadata_path = attempt_dir / "bag" / "metadata.yaml"
    for path in (verify_path, runtime_path, export_manifest_path, bag_metadata_path):
        if not path.is_file():
            raise ValueError(f"successful runner omitted {path}")
    verifier = load_json(verify_path)
    if not verifier.get("valid") or not verifier["closed_loop"]["goal_reached"]:
        raise ValueError("verifier did not accept goal completion")
    if verifier["closed_loop"]["collision"]:
        raise ValueError("expert data-generation episode collided")
    if (
        float(verifier["simulated_span_sec"]["action"])
        > float(entry["scientific_simulated_deadline_sec"]) + 1.0
    ):
        raise ValueError("episode exceeded the scientific simulated-time deadline")
    export_manifest = load_json(export_manifest_path)
    if int(export_manifest["counts"]["accepted_samples"]) <= 0:
        raise ValueError("export contains no accepted samples")
    if int(export_manifest["contiguity"]["windowable_k8_h8"]) <= 0:
        raise ValueError("export contains no K=8/H=8 training window")
    mcap_paths = sorted((attempt_dir / "bag").glob("*.mcap"))
    if len(mcap_paths) != 1:
        raise ValueError("accepted bag must contain exactly one MCAP shard")
    return {
        "verify.json": sha256_file(verify_path),
        "runtime.json": sha256_file(runtime_path),
        "export/manifest.json": sha256_file(export_manifest_path),
        "bag/metadata.yaml": sha256_file(bag_metadata_path),
        f"bag/{mcap_paths[0].name}": sha256_file(mcap_paths[0]),
    }


def _write_json_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_bytes(json_bytes(payload))


def run_entry(
    entry: dict,
    output_root: Path,
    distribution: str,
) -> bool:
    episode_dir = output_root / "episodes" / entry["episode_id"]
    if accepted_success(entry, episode_dir):
        return True
    attempt_dir = next_attempt_dir(episode_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    command = [
        "wsl.exe",
        "-d",
        distribution,
        "--",
        "bash",
        to_wsl_path(EPISODE_SCRIPT),
        to_wsl_path(output_root / entry["world_json"]),
        to_wsl_path(output_root / entry["world_sdf"]),
        str(entry["lidar_condition"]),
        str(entry["observation_seed"]),
        str(entry["episode_id"]),
        to_wsl_path(attempt_dir),
        str(entry["operational_wall_watchdog_sec"]),
        str(entry["scientific_simulated_deadline_sec"]),
    ]
    started = time.time()
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    attempt_record = {
        "schema_version": "1.0.0",
        "episode_id": entry["episode_id"],
        "entry_sha256": entry["entry_sha256"],
        "started_unix_sec": started,
        "finished_unix_sec": time.time(),
        "return_code": completed.returncode,
    }
    accepted = False
    if completed.returncode == 0:
        try:
            hashes = _validate_attempt(attempt_dir, entry)
        except (KeyError, TypeError, ValueError) as error:
            attempt_record["status"] = "postvalidation_failed"
            attempt_record["error"] = str(error)
        else:
            attempt_record["status"] = "accepted"
            attempt_record["sha256"] = hashes
            _write_json_once(attempt_dir / "ATTEMPT.json", attempt_record)
            _write_json_once(
                episode_dir / "SUCCESS.json",
                {
                    "schema_version": "1.0.0",
                    "status": "accepted",
                    "episode_id": entry["episode_id"],
                    "entry_sha256": entry["entry_sha256"],
                    "accepted_attempt": attempt_dir.name,
                    "attempt_manifest_sha256": sha256_file(
                        attempt_dir / "ATTEMPT.json"
                    ),
                },
            )
            accepted = True
    else:
        attempt_record["status"] = "runner_failed"
    if not (attempt_dir / "ATTEMPT.json").exists():
        _write_json_once(attempt_dir / "ATTEMPT.json", attempt_record)
    return accepted


def summarize(plan: Plan, output_root: Path, show_next: int) -> dict[str, Any]:
    completed = 0
    pending = []
    attempts = 0
    for entry in plan.schedule["episodes"]:
        episode_dir = output_root / "episodes" / entry["episode_id"]
        if accepted_success(entry, episode_dir):
            completed += 1
        else:
            pending.append(entry)
        if episode_dir.is_dir():
            attempts += len(list(episode_dir.glob("attempt_*")))
    return {
        "schedule_sha256": plan.schedule["schedule_sha256_excludes_self"],
        "total": len(plan.schedule["episodes"]),
        "accepted": completed,
        "pending": len(pending),
        "attempt_directories": attempts,
        "free_gib": round(free_gib(output_root), 2),
        "minimum_free_gib_before_run": float(
            load_json(DEFAULT_CONFIG)["minimum_free_gib_before_run"]
        ),
        "next": [entry["episode_id"] for entry in pending[:show_next]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--from-ordinal", type=int, default=0)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--show-next", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    if config_path != DEFAULT_CONFIG.resolve():
        raise ValueError("alternate batch configs are not permitted by v3")
    output_root = (REPOSITORY_ROOT / config["output_root"]).resolve()
    if args.output_root is not None and args.output_root.resolve() != output_root:
        raise ValueError("alternate output roots are forbidden by the v3 freeze")
    forbidden_roots = {
        (REPOSITORY_ROOT / relative).resolve()
        for relative in config["forbidden_predecessor_roots"]
    }
    if output_root in forbidden_roots:
        raise ValueError("refusing to use an invalidated predecessor output root")
    plan = build_plan(config)

    if args.prepare:
        result = prepare(plan, output_root)
        payload = {
            "prepared": result,
            **summarize(plan, output_root, args.show_next),
        }
        print(json.dumps(payload, indent=2))
        return 0

    if not args.run:
        summary = summarize(plan, output_root, args.show_next)
        summary["mode"] = "dry_run_no_writes"
        summary["prepared"] = (output_root / "schedule.json").is_file()
        print(json.dumps(summary, indent=2))
        return 0

    validate_prepared(plan, output_root)
    available = free_gib(output_root)
    required = float(config["minimum_free_gib_before_run"])
    if available < required:
        raise RuntimeError(
            f"disk gate failed: {available:.2f} GiB free, {required:.2f} GiB required"
        )
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    pending = [
        entry
        for entry in plan.schedule["episodes"]
        if entry["ordinal"] >= args.from_ordinal
        and not accepted_success(
            entry, output_root / "episodes" / entry["episode_id"]
        )
    ]
    selected = pending if args.limit is None else pending[: args.limit]
    failures = 0
    for position, entry in enumerate(selected, start=1):
        print(
            f"[{position}/{len(selected)}] ordinal={entry['ordinal']} "
            f"{entry['episode_id']}"
        )
        if not run_entry(entry, output_root, str(config["wsl_distribution"])):
            failures += 1
            if not args.continue_on_failure:
                break
    print(json.dumps(summarize(plan, output_root, args.show_next), indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
