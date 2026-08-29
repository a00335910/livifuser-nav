#!/usr/bin/env python3
"""Aggregate explicit Nav2 development-probe artifacts without glob selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("C0", "C4"), required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("episodes", type=Path, nargs="+")
    args = parser.parse_args()
    if args.expected_count <= 0:
        parser.error("--expected-count must be positive")
    if len(args.episodes) != args.expected_count:
        parser.error(
            f"expected {args.expected_count} explicit episodes, got {len(args.episodes)}"
        )
    if len({path.resolve() for path in args.episodes}) != len(args.episodes):
        parser.error("episode paths must be unique")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")

    rows = []
    for path in sorted(args.episodes):
        verifier = json.loads(path.read_text(encoding="utf-8"))
        stem = path.with_suffix("")
        status_path = Path(str(stem) + ".nav2_status.json")
        map_path = Path(str(stem) + ".map.pgm")
        map_manifest_path = Path(str(stem) + ".map.manifest.json")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        map_manifest = json.loads(map_manifest_path.read_text(encoding="utf-8"))
        if status["condition"] != args.condition:
            raise ValueError(
                f"{status_path} condition {status['condition']} != {args.condition}"
            )
        closed_loop = verifier["closed_loop"]
        rows.append(
            {
                "episode": str(path.resolve()),
                "world": status["world"],
                "valid": bool(verifier["valid"]),
                "issues": list(verifier["issues"]),
                "goal_reached": bool(closed_loop["goal_reached"]),
                "collision": bool(closed_loop["collision"]),
                "final_goal_distance_m": closed_loop["final_goal_distance_m"],
                "minimum_collision_clearance_m": closed_loop[
                    "minimum_collision_clearance_m"
                ],
                "wall_elapsed_sec": verifier["wall_elapsed_sec"],
                "nav2_phase": status["phase"],
                "nav2_result_status": status["result_status"],
                "command_count": status["command_count"],
                "clamped_command_count": status["clamped_command_count"],
                "invalid_command_count": status["invalid_command_count"],
                "verifier_sha256": sha256(path),
                "nav2_status_sha256": sha256(status_path),
                "map_pgm_sha256": sha256(map_path),
                "map_included_obstacles": map_manifest["included_obstacles"],
                "map_excluded_switchable_obstacles": map_manifest[
                    "excluded_switchable_obstacles"
                ],
            }
        )

    successes = sum(row["goal_reached"] and row["valid"] for row in rows)
    collisions = sum(row["collision"] for row in rows)
    clearances = [float(row["minimum_collision_clearance_m"]) for row in rows]
    wall_times = [float(row["wall_elapsed_sec"]) for row in rows]
    output = {
        "schema_version": 1,
        "scope": "development-only Nav2 structural probe",
        "condition": args.condition,
        "params_path": str(args.params.resolve()),
        "params_sha256": sha256(args.params),
        "episode_count": len(rows),
        "success_count": successes,
        "success_rate": successes / len(rows),
        "collision_count": collisions,
        "collision_rate": collisions / len(rows),
        "all_nav2_actions_succeeded": all(row["nav2_phase"] == "succeeded" for row in rows),
        "total_clamped_commands": sum(row["clamped_command_count"] for row in rows),
        "total_invalid_commands": sum(row["invalid_command_count"] for row in rows),
        "minimum_clearance_m": min(clearances),
        "median_clearance_m": statistics.median(clearances),
        "wall_elapsed_sec": {
            "minimum": min(wall_times),
            "median": statistics.median(wall_times),
            "maximum": max(wall_times),
            "total": sum(wall_times),
        },
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
