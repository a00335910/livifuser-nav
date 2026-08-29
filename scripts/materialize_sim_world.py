#!/usr/bin/env python3
"""Materialize one generated schema-2 world as a Gazebo Fortress SDF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_sim.world_sdf import render_world_sdf  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world_json", type=Path)
    parser.add_argument("output_sdf", type=Path)
    parser.add_argument(
        "--template",
        type=Path,
        default=PACKAGE_ROOT / "worlds" / "livifuser_lab.sdf",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_sdf.exists() and not args.overwrite:
        print(f"refusing to overwrite {args.output_sdf}", file=sys.stderr)
        return 3
    payload = json.loads(args.world_json.read_text(encoding="utf-8"))
    rendered = render_world_sdf(payload, args.template)
    args.output_sdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_sdf.write_text(rendered, encoding="utf-8")
    print(args.output_sdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
