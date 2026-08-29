#!/usr/bin/env python3
"""Emit the preregistered disjoint simulation world groups.

Thin CLI over ``livifuser_sim.world_generator``; every decision that affects
what is generated lives in that module, where it is unit tested.

Writes nothing outside the chosen output directory and refuses to overwrite an
existing world unless ``--force`` is given, because a regenerated world that
silently replaces one already used for training breaks the reproducibility
chain that `docs/RESEARCH_RECORD.md` requires.

Development worlds may be generated before the preregistration freeze; section 4
excludes them from every confirmatory result by construction. Confirmatory
groups are refused unless ``--allow-confirmatory`` is passed, so the freeze is a
deliberate act rather than a default.

    python3 scripts/generate_sim_worlds.py --group dev --output artifacts/simulation/worlds
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"))

from livifuser_sim.world_generator import (  # noqa: E402
    GROUP_SEED_BLOCKS,
    WorldGenerationError,
    derive_condition,
    generate_world,
)

#: §5 world counts. `test_ood` is absent because it reuses `test_id` geometry.
GROUP_COUNTS = {"dev": 2, "train": 6, "val_id": 2, "test_id": 2}

CONFIRMATORY_GROUPS = ("train", "val_id", "test_id")


def build_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        choices=sorted(GROUP_SEED_BLOCKS),
        help="world group; repeatable. Defaults to dev only.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="override the section 5 world count for the selected groups",
    )
    parser.add_argument(
        "--allow-confirmatory",
        action="store_true",
        help="permit train/val_id/test_id generation (requires a frozen protocol)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing worlds")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_arguments().parse_args(argv)
    groups = arguments.group or ["dev"]

    confirmatory = [group for group in groups if group in CONFIRMATORY_GROUPS]
    if confirmatory and not arguments.allow_confirmatory:
        print(
            f"refusing to generate confirmatory groups {confirmatory} without "
            "--allow-confirmatory; the preregistration freeze gates these "
            "(PREREGISTRATION_SIM_SENSOR_FAILURE_V1.md section 16)",
            file=sys.stderr,
        )
        return 2

    arguments.output.mkdir(parents=True, exist_ok=True)
    durations: list[float] = []
    written = 0

    for group in groups:
        count = arguments.count if arguments.count is not None else GROUP_COUNTS[group]
        for index in range(count):
            try:
                payload = generate_world(group, index)
            except WorldGenerationError as error:
                print(f"FAILED {group}[{index}]: {error}", file=sys.stderr)
                return 1

            conditions = (
                ("C0", "C1", "C4") if group in {"dev", "test_id"} else ("C0",)
            )
            for condition in conditions:
                variant = (
                    payload if condition == "C0" else derive_condition(payload, condition)
                )
                suffix = "" if condition == "C0" else f".{condition}"
                destination = arguments.output / f"{payload['name']}{suffix}.json"
                if destination.exists() and not arguments.force:
                    print(f"refusing to overwrite {destination}", file=sys.stderr)
                    return 3
                destination.write_text(
                    json.dumps(variant, indent=2) + "\n", encoding="utf-8", newline="\n"
                )
                written += 1

            validation = payload["validation"]
            durations.append(validation["expert_episode_duration_s"])
            print(
                f"{payload['name']:34s} seed={payload['seed']} "
                f"draw={payload['draw']} "
                f"dur={validation['expert_episode_duration_s']:6.1f}s "
                f"clearance={validation['expert_minimum_clearance_m']:.3f}m "
                f"ratio={validation['path_excess_ratio']:.3f}"
            )

    if durations:
        print(
            f"\n{written} files, {len(durations)} worlds. "
            f"Expert episode duration: min {min(durations):.1f}s "
            f"mean {statistics.mean(durations):.1f}s "
            f"median {statistics.median(durations):.1f}s "
            f"max {max(durations):.1f}s"
        )
        print(
            "Feed the mean back into PREREGISTRATION section 12.2 before "
            "freezing counts; "
            "it is the measured replacement for the 45 s estimate."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
