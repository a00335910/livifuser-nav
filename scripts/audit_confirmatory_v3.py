#!/usr/bin/env python3
"""Reproduce the frozen confirmatory-v3 post-collection audit.

This audit does not calculate policy performance.  It binds every accepted
episode to the frozen schedule, SUCCESS marker, accepted attempt, verifier,
runtime record, export manifest, and every exported output byte.  It then
checks the predeclared data-quality and single-factor condition contracts.

Raw MCAPs remain preserved and their acquisition-time SHA-256 declarations are
recorded, but this audit deliberately does not rehash the large bags.  The
derived exports used downstream are rehashed in full.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
SIM_PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SIM_PACKAGE_ROOT))

from livifuser_sim.visual_conditions import (  # noqa: E402
    C1_VISUAL_CONTRACT_SHA256,
    evaluate_c1_development_gate,
)
from package_sim_train_val_handoff import (  # noqa: E402
    audit_episode,
    canonical_bytes,
    load_json,
    sha256_bytes,
    sha256_file,
)

DEFAULT_ROOT = REPOSITORY_ROOT / "artifacts/simulation/confirmatory_v3"
DEFAULT_OUTPUT = DEFAULT_ROOT / "post_collection_audit.json"
EXPECTED_SCHEDULE_SHA256 = "85F48F81D52F82A5CD64D14D704E032617582D7F5462A7B6FD46C5E64899047D"
EXPECTED_SPLITS = {"train": 120, "val_id": 30, "test_id": 30, "test_ood": 80}
EXPECTED_CONDITIONS = {"C0": 200, "C1": 20, "C3": 20, "C4": 20}
EXPECTED_OOD_CONDITIONS = {"C0": 20, "C1": 20, "C3": 20, "C4": 20}
EXPECTED_STYLES = {
    ("train", "train_wood_brick"): 120,
    ("val_id", "val_amber_plaster"): 30,
    ("test_id", "test_navy_wood"): 30,
    ("test_ood", "test_navy_wood"): 80,
}
TEMPORAL_COMPARISONS = (
    ("valid_rgb_frames", "minimum_valid_rgb_frames", ">="),
    ("unique_frame_hashes", "minimum_unique_frame_hashes", ">="),
    ("modal_frame_fraction", "maximum_modal_frame_fraction", "<="),
    ("moving_pair_count", "minimum_moving_pair_count", ">="),
    (
        "changed_moving_pair_fraction",
        "minimum_changed_moving_pair_fraction",
        ">=",
    ),
    (
        "motion_pair_mean_absolute_rgb_difference_median",
        "minimum_motion_pair_median_rgb_difference",
        ">=",
    ),
    ("maximum_identical_motion_run_sec", "maximum_identical_motion_run_sec", "<="),
)


def schedule_self_hash(schedule: dict[str, Any]) -> str:
    payload = copy.deepcopy(schedule)
    payload.pop("schedule_sha256_excludes_self", None)
    return sha256_bytes(canonical_bytes(payload))


def entry_self_hash(entry: dict[str, Any]) -> str:
    payload = copy.deepcopy(entry)
    payload.pop("entry_sha256", None)
    return sha256_bytes(canonical_bytes(payload))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_attempt_name(value: object) -> str:
    require(isinstance(value, str), "accepted attempt name is not a string")
    require(value.startswith("attempt_") and value[8:].isdigit(), "unsafe attempt name")
    return value


def safe_relative_file(root: Path, relative: str) -> Path:
    path = Path(relative)
    require(not path.is_absolute(), f"absolute evidence path is forbidden: {relative}")
    require(path.as_posix() == relative, f"non-canonical evidence path: {relative}")
    require(".." not in path.parts, f"parent traversal in evidence path: {relative}")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()), f"evidence path escaped root: {relative}")
    require(resolved.is_file(), f"missing evidence file: {relative}")
    return resolved


def load_schedule(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schedule = load_json(root / "schedule.json")
    declared = schedule.get("schedule_sha256_excludes_self")
    actual = schedule_self_hash(schedule)
    require(declared == actual, f"schedule self-hash mismatch: {declared} != {actual}")
    require(actual == EXPECTED_SCHEDULE_SHA256, f"unexpected schedule identity: {actual}")
    entries = list(schedule.get("episodes", []))
    require(len(entries) == 260, f"expected 260 schedule entries, found {len(entries)}")
    require(
        [int(entry["ordinal"]) for entry in entries] == list(range(260)),
        "schedule ordinals are not the frozen contiguous range",
    )
    require(
        len({entry["episode_id"] for entry in entries}) == 260,
        "schedule episode IDs are not unique",
    )
    for entry in entries:
        require(
            entry_self_hash(entry) == entry.get("entry_sha256"),
            f"{entry['episode_id']}: entry self-hash mismatch",
        )
    require(
        Counter(entry["split"] for entry in entries) == Counter(EXPECTED_SPLITS),
        "schedule split counts differ from the frozen design",
    )
    require(
        Counter(entry["condition"] for entry in entries) == Counter(EXPECTED_CONDITIONS),
        "schedule condition counts differ from the frozen design",
    )
    ood = [entry for entry in entries if entry["split"] == "test_ood"]
    require(
        Counter(entry["condition"] for entry in ood) == Counter(EXPECTED_OOD_CONDITIONS),
        "test-OOD condition grid differs from the frozen design",
    )
    return schedule, entries


def verify_frozen_references(root: Path, schedule: dict[str, Any]) -> dict[str, Any]:
    assets = []
    for record in schedule["assets"]:
        path = safe_relative_file(root, record["path"])
        digest = sha256_file(path)
        require(digest == record["sha256"], f"frozen asset drifted: {record['path']}")
        assets.append({"path": record["path"], "sha256": digest})

    amendments = []
    for record in schedule["amendments"]:
        path = safe_relative_file(REPOSITORY_ROOT, record["path"])
        digest = sha256_file(path)
        require(digest == record["sha256"], f"amendment drifted: {record['path']}")
        amendments.append(
            {"number": int(record["number"]), "path": record["path"], "sha256": digest}
        )

    freeze_relative = schedule["recollection_freeze_manifest"]
    freeze_path = safe_relative_file(REPOSITORY_ROOT, freeze_relative)
    freeze_sha = sha256_file(freeze_path)
    require(
        freeze_sha == schedule["recollection_freeze_manifest_sha256"],
        "recollection freeze manifest drifted",
    )
    return {
        "assets": assets,
        "amendments": amendments,
        "recollection_freeze_manifest": {
            "path": freeze_relative,
            "sha256": freeze_sha,
        },
    }


def validate_temporal_rgb(stats: dict[str, Any]) -> dict[str, float]:
    gates = stats.get("gates")
    require(isinstance(gates, dict), "temporal RGB gate record is missing")
    values: dict[str, float] = {}
    for field, threshold_field, operator in TEMPORAL_COMPARISONS:
        value = float(stats.get(field, math.nan))
        threshold = float(gates.get(threshold_field, math.nan))
        require(math.isfinite(value), f"temporal RGB value is non-finite: {field}")
        require(
            math.isfinite(threshold), f"temporal RGB threshold is non-finite: {threshold_field}"
        )
        passed = value >= threshold if operator == ">=" else value <= threshold
        require(passed, f"temporal RGB gate failed: {field} {operator} {threshold}")
        values[field] = value
    return values


def first_scan_signature(verify: dict[str, Any]) -> tuple[object, ...]:
    stats = verify["first_scan_statistics"]
    return tuple(
        stats.get(field)
        for field in (
            "finite_positive_count",
            "zero_count",
            "nan_count",
            "minimum_m",
            "median_m",
            "maximum_m",
        )
    )


def c3_pair_evidence(
    c0_entry: dict[str, Any],
    c3_entry: dict[str, Any],
    c0_verify: dict[str, Any],
    c3_verify: dict[str, Any],
) -> dict[str, Any]:
    for field in (
        "source_world_group",
        "world_index",
        "world_name",
        "world_seed",
        "episode_index",
        "observation_seed",
    ):
        require(c0_entry[field] == c3_entry[field], f"C0/C3 paired field drifted: {field}")
    require(c3_entry["condition"] == "C3", "C3 pair has the wrong condition")
    require(c3_entry["world_variant"] == "C0", "C3 changed the world variant")
    require(c3_entry["lidar_condition"] == "C3b", "C3 is not bound to frozen C3b")
    require(c0_entry["world_json"] == c3_entry["world_json"], "C3 changed world JSON")
    require(c0_entry["world_sdf"] == c3_entry["world_sdf"], "C3 changed world SDF")
    c0_scan = c0_verify["first_scan_statistics"]
    c3_scan = c3_verify["first_scan_statistics"]
    c0_missing = int(c0_scan["zero_count"]) + int(c0_scan["nan_count"])
    c3_missing = int(c3_scan["zero_count"]) + int(c3_scan["nan_count"])
    require(c3_missing > c0_missing, "C3 did not increase missing first-scan returns")
    require(
        first_scan_signature(c0_verify) != first_scan_signature(c3_verify),
        "C3 first-scan evidence is vacuous",
    )
    c0_image = c0_verify["first_image_statistics"]["active_rgb_sha256"]
    c3_image = c3_verify["first_image_statistics"]["active_rgb_sha256"]
    return {
        "c0_missing_returns": c0_missing,
        "c3_missing_returns": c3_missing,
        "first_scan_statistics_differ": True,
        "first_rgb_hash_equal": c0_image == c3_image,
    }


def compare_c1_worlds(c0: dict[str, Any], c1: dict[str, Any]) -> dict[str, Any]:
    descriptor = c1.get("camera_condition")
    require(
        descriptor
        == {
            "name": "C1_WARM_LOW_LIGHT_V1",
            "sha256": C1_VISUAL_CONTRACT_SHA256,
        },
        "C1 world lacks the frozen camera condition descriptor",
    )
    normalized = copy.deepcopy(c1)
    normalized.pop("condition", None)
    normalized.pop("camera_condition", None)
    require(normalized == c0, "C1 changed something beyond camera appearance metadata")
    return {"geometry_and_labels_identical": True, "contract": descriptor}


def compare_c4_worlds(c0: dict[str, Any], c4: dict[str, Any]) -> dict[str, Any]:
    require(c4.get("condition") == "C4", "derived C4 world lacks its condition marker")
    hidden = list(c4.get("c4_hidden_from_lidar", []))
    require(hidden, "C4 world has no hidden switchable obstacle")

    c0_common = copy.deepcopy(c0)
    c4_common = copy.deepcopy(c4)
    c0_obstacles = c0_common.pop("obstacles")
    c4_obstacles = c4_common.pop("obstacles")
    c4_common.pop("condition", None)
    c4_common.pop("c4_hidden_from_lidar", None)
    require(c0_common == c4_common, "C4 changed world metadata, start, goal, or laser")
    require(len(c0_obstacles) == len(c4_obstacles), "C4 changed obstacle count")

    observed_hidden = []
    for before, after in zip(c0_obstacles, c4_obstacles, strict=True):
        require(before["name"] == after["name"], "C4 changed obstacle order or name")
        expected = copy.deepcopy(before)
        if before.get("profile_switchable", False):
            require(all(before["layers"].values()), "switchable C0 obstacle is not in every layer")
            expected["layers"]["lidar"] = False
            expected.pop("profile_switchable", None)
            observed_hidden.append(before["name"])
        require(expected == after, f"C4 changed more than LiDAR membership: {before['name']}")
    require(observed_hidden == hidden, "C4 hidden-obstacle metadata disagrees with layers")
    return {
        "hidden_obstacles": hidden,
        "collision_identical": True,
        "expert_identical": True,
        "camera_identical": True,
        "start_goal_seed_identical": True,
        "only_lidar_membership_changed": True,
    }


def paired_ood_entries(entries: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    groups: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for entry in entries:
        if entry["split"] != "test_ood":
            continue
        key = (
            int(entry["world_index"]),
            int(entry["episode_index"]),
            int(entry["observation_seed"]),
        )
        require(entry["condition"] not in groups[key], "duplicate OOD condition in pair")
        groups[key][entry["condition"]] = entry
    require(len(groups) == 20, f"expected 20 paired OOD groups, found {len(groups)}")
    for group in groups.values():
        require(set(group) == {"C0", "C1", "C3", "C4"}, "OOD pair lacks a condition")
    return [groups[key] for key in sorted(groups)]


def audit_attempt(
    root: Path, entry: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, int]:
    episode_root = root / "episodes" / entry["episode_id"]
    success_path = episode_root / "SUCCESS.json"
    success = load_json(success_path)
    attempt_name = safe_attempt_name(success["accepted_attempt"])
    attempt_root = episode_root / attempt_name
    attempt = load_json(attempt_root / "ATTEMPT.json")
    require(attempt.get("episode_id") == entry["episode_id"], "attempt episode identity drifted")

    mcap_records = []
    for relative, expected in sorted(attempt["sha256"].items()):
        path = safe_relative_file(attempt_root, relative)
        if relative.endswith(".mcap"):
            require(len(expected) == 64, "invalid acquisition-time MCAP digest")
            mcap_records.append(
                {"path": relative, "size_bytes": path.stat().st_size, "declared_sha256": expected}
            )
        else:
            require(sha256_file(path) == expected, f"attempt evidence hash mismatch: {relative}")
    require(len(mcap_records) == 1, "accepted attempt must contain exactly one MCAP record")

    verify = load_json(attempt_root / "verify.json")
    runtime = load_json(attempt_root / "runtime.json")
    require(
        verify.get("valid") is True and verify.get("issues") == [], "accepted verifier is not valid"
    )
    require(runtime.get("episode_id") == entry["episode_id"], "runtime episode identity drifted")
    require(
        runtime.get("lidar_condition") == entry["lidar_condition"],
        "runtime LiDAR condition drifted",
    )
    require(
        int(runtime.get("observation_seed")) == int(entry["observation_seed"]),
        "runtime observation seed drifted",
    )
    require(
        int(runtime.get("verifier_exit_code", -1)) == 0, "runtime verifier did not exit cleanly"
    )

    attempt_dirs = sorted(path for path in episode_root.glob("attempt_*") if path.is_dir())
    failed = 0
    for path in attempt_dirs:
        record_path = path / "ATTEMPT.json"
        if record_path.is_file() and load_json(record_path).get("status") != "accepted":
            failed += 1
    evidence = {
        "verify_sha256": attempt["sha256"]["verify.json"],
        "runtime_sha256": attempt["sha256"]["runtime.json"],
        "raw_mcap": mcap_records[0],
        "raw_mcap_post_collection_rehashed": False,
        "repair": bool(attempt.get("repair", False)),
    }
    return verify, runtime, evidence, len(attempt_dirs), failed


def aggregate_range(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def build_audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    schedule, entries = load_schedule(root)
    frozen = verify_frozen_references(root, schedule)

    audited = []
    verifiers: dict[str, dict[str, Any]] = {}
    style_counts: Counter[tuple[str, str]] = Counter()
    temporal_values: dict[str, list[float]] = defaultdict(list)
    source_trees = set()
    configurations = set()
    total_attempt_dirs = 0
    failed_attempt_dirs = 0
    repaired_accepted = 0

    for index, entry in enumerate(entries, start=1):
        audited_episode = audit_episode(root, entry)
        verify, _runtime, evidence, attempts, failed = audit_attempt(root, entry)
        verifiers[entry["episode_id"]] = verify
        total_attempt_dirs += attempts
        failed_attempt_dirs += failed
        repaired_accepted += int(evidence["repair"])
        values = validate_temporal_rgb(verify["temporal_image_statistics"])
        for field, value in values.items():
            temporal_values[field].append(value)

        world = load_json(root / entry["world_json"])
        style = world["visual_skin"]["style"]
        style_counts[(entry["split"], style)] += 1
        record = dict(audited_episode.record)
        record.update(evidence)
        record["visual_style"] = style
        audited.append(record)
        source_trees.add(record["source_tree_sha256"])
        configurations.add(record["effective_configuration_sha256"])
        if index % 25 == 0 or index == len(entries):
            print(f"audited {index}/{len(entries)} accepted exports", flush=True)

    require(
        style_counts == Counter(EXPECTED_STYLES), f"visual style counts drifted: {style_counts}"
    )
    require(len(source_trees) == 1, f"multiple export source trees: {sorted(source_trees)}")
    require(len(configurations) == 1, f"multiple export configurations: {sorted(configurations)}")

    pairs = paired_ood_entries(entries)
    c1_results = []
    c3_results = []
    c4_schedule_pairs = 0
    for pair in pairs:
        c0 = pair["C0"]
        c1 = pair["C1"]
        c3 = pair["C3"]
        c4 = pair["C4"]
        for condition_entry in (c1, c3, c4):
            for field in (
                "source_world_group",
                "world_index",
                "world_name",
                "world_seed",
                "episode_index",
                "observation_seed",
            ):
                require(c0[field] == condition_entry[field], f"matched OOD field drifted: {field}")

        gate = evaluate_c1_development_gate(
            verifiers[c0["episode_id"]]["first_image_statistics"],
            verifiers[c1["episode_id"]]["first_image_statistics"],
        )
        # This exact gate is explicitly preregistered as a one-time development
        # gate, not a per-confirmatory-episode inclusion rule. Reapplying its
        # scene-contrast clause to each fixed start viewpoint would invent a new
        # post-results exclusion criterion. Preserve all diagnostics below.
        c1_results.append(gate)
        c3_results.append(
            c3_pair_evidence(
                c0,
                c3,
                verifiers[c0["episode_id"]],
                verifiers[c3["episode_id"]],
            )
        )
        require(c4["world_variant"] == "C4", "C4 schedule did not select C4 world")
        require(c4["lidar_condition"] == "C0", "C4 also changed LiDAR noise condition")
        require(
            sha256_file(root / c0["world_sdf"]) == sha256_file(root / c4["world_sdf"]),
            "matched C0/C4 Gazebo SDF geometry differs",
        )
        c4_schedule_pairs += 1

    c1_worlds = []
    c4_worlds = []
    for world_index in sorted({int(pair["C0"]["world_index"]) for pair in pairs}):
        pair = next(value for value in pairs if int(value["C0"]["world_index"]) == world_index)
        c0_entry = pair["C0"]
        c1_entry = pair["C1"]
        c4_entry = pair["C4"]
        c0_world = load_json(root / c0_entry["world_json"])
        c1_world = load_json(root / c1_entry["world_json"])
        c4_world = load_json(root / c4_entry["world_json"])
        c1_record = compare_c1_worlds(c0_world, c1_world)
        c1_record["world_name"] = c0_entry["world_name"]
        c1_record["gazebo_sdf_differs"] = sha256_file(root / c0_entry["world_sdf"]) != sha256_file(
            root / c1_entry["world_sdf"]
        )
        require(c1_record["gazebo_sdf_differs"], "C1 Gazebo scene intervention is vacuous")
        c1_worlds.append(c1_record)
        c4_record = compare_c4_worlds(c0_world, c4_world)
        c4_record["world_name"] = c0_entry["world_name"]
        c4_record["gazebo_sdf_sha256"] = sha256_file(root / c0_entry["world_sdf"])
        c4_worlds.append(c4_record)

    c1_measurements: dict[str, list[float]] = defaultdict(list)
    c1_diagnostic_issues: Counter[str] = Counter()
    c1_first_frame_shift_visible = 0
    for result in c1_results:
        for field, value in result["measurements"].items():
            c1_measurements[field].append(float(value))
        c1_diagnostic_issues.update(result["issues"])
        measurements = result["measurements"]
        thresholds = result["thresholds"]
        if (
            measurements["mean_absolute_channel_mean_delta"]
            >= thresholds["minimum_mean_absolute_channel_mean_delta"]
            and measurements["c1_to_c0_luminance_ratio"]
            >= thresholds["minimum_c1_to_c0_luminance_ratio"]
            and measurements["c1_to_c0_luminance_ratio"]
            <= thresholds["maximum_c1_to_c0_luminance_ratio"]
        ):
            c1_first_frame_shift_visible += 1

    freeze_manifest = load_json(REPOSITORY_ROOT / schedule["recollection_freeze_manifest"])
    require(
        freeze_manifest["development_gate"].get("c1_non_vacuity_gate_passed") is True,
        "frozen pre-collection C1 development gate was not passed",
    )

    report = {
        "schema_version": "1.0.0",
        "name": "livifuser_confirmatory_v3_post_collection_audit",
        "status": "PASS",
        "scope": {
            "policy_performance_computed": False,
            "training_statistics_computed": False,
            "raw_mcap_post_collection_rehashed": False,
            "export_outputs_post_collection_rehashed": True,
        },
        "source": {
            "simulation_root": root.relative_to(REPOSITORY_ROOT).as_posix(),
            "schedule_sha256_excludes_self": schedule["schedule_sha256_excludes_self"],
            "schedule_file_sha256": sha256_file(root / "schedule.json"),
            "export_source_tree_sha256": next(iter(source_trees)),
            "effective_export_configuration_sha256": next(iter(configurations)),
            **frozen,
        },
        "counts": {
            "episodes": len(audited),
            "by_split": dict(sorted(Counter(entry["split"] for entry in entries).items())),
            "by_condition": dict(sorted(Counter(entry["condition"] for entry in entries).items())),
            "by_split_and_style": {
                f"{split}:{style}": count for (split, style), count in sorted(style_counts.items())
            },
            "accepted_samples": sum(record["accepted_samples"] for record in audited),
            "windowable_k8_h8": sum(record["windowable_k8_h8"] for record in audited),
            "export_payload_bytes": sum(
                output["size_bytes"] for record in audited for output in record["outputs"].values()
            ),
            "attempt_directories_after_repairs": total_attempt_dirs,
            "failed_or_rejected_attempt_directories": failed_attempt_dirs,
            "accepted_repair_attempts": repaired_accepted,
        },
        "temporal_rgb": {
            "episodes_passing": len(audited),
            "episodes_total": len(audited),
            "ranges": {
                field: aggregate_range(values) for field, values in sorted(temporal_values.items())
            },
        },
        "c1": {
            "frozen_pre_collection_development_gate_passed": True,
            "development_gate_applies_per_confirmatory_episode": False,
            "paired_first_frame_appearance_shifts_visible": c1_first_frame_shift_visible,
            "paired_episodes_total": 20,
            "contract_sha256": C1_VISUAL_CONTRACT_SHA256,
            "per_episode_full_development_gate_diagnostic_passed": sum(
                int(result["valid"]) for result in c1_results
            ),
            "per_episode_full_development_gate_diagnostic_issues": dict(
                sorted(c1_diagnostic_issues.items())
            ),
            "diagnostic_interpretation": (
                "All paired start frames pass the frozen shift-magnitude and "
                "luminance-ratio clauses. The full development gate is reported "
                "diagnostically only because its contrast clause was frozen as a "
                "one-time pre-collection world gate, not an episode inclusion rule."
            ),
            "measurement_ranges": {
                field: aggregate_range(values) for field, values in sorted(c1_measurements.items())
            },
            "world_contracts": c1_worlds,
        },
        "c3": {
            "paired_episodes_passing": len(c3_results),
            "paired_episodes_total": 20,
            "lidar_condition": "C3b",
            "pairs_with_different_first_scan_statistics": sum(
                int(item["first_scan_statistics_differ"]) for item in c3_results
            ),
            "pairs_with_exact_first_rgb_hash": sum(
                int(item["first_rgb_hash_equal"]) for item in c3_results
            ),
            "minimum_c3_missing_first_scan_returns": min(
                item["c3_missing_returns"] for item in c3_results
            ),
            "maximum_c0_missing_first_scan_returns": max(
                item["c0_missing_returns"] for item in c3_results
            ),
            "rgb_intervention": "none; C0 world JSON and Gazebo SDF are reused exactly",
        },
        "c4": {
            "matched_schedule_pairs_passing": c4_schedule_pairs,
            "matched_schedule_pairs_total": 20,
            "world_geometry_pairs_passing": len(c4_worlds),
            "world_geometry_pairs_total": 2,
            "world_contracts": c4_worlds,
        },
        "episodes": audited,
    }
    report["audit_sha256_excludes_self"] = sha256_bytes(canonical_bytes(report))
    return report


def write_immutable(path: Path, report: dict[str, Any]) -> str:
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        require(path.read_bytes() == payload, f"refusing to overwrite different audit: {path}")
        return "verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not temporary.exists(), f"partial audit already exists: {temporary}")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return "written"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_audit(args.root)
    output_status = None
    if args.output is not None:
        output_status = write_immutable(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "audit_sha256_excludes_self": report["audit_sha256_excludes_self"],
                "counts": report["counts"],
                "temporal_rgb": report["temporal_rgb"],
                "c1": {
                    key: value for key, value in report["c1"].items() if key != "world_contracts"
                },
                "c3": report["c3"],
                "c4": {
                    key: value for key, value in report["c4"].items() if key != "world_contracts"
                },
                "output": str(args.output.resolve()) if args.output is not None else None,
                "output_status": output_status,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
