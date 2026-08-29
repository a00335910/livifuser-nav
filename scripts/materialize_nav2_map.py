#!/usr/bin/env python3
"""Materialize the permanent-geometry map used by the bounded Nav2 probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_sim.nav2_map import rasterize_nav2_map  # noqa: E402
from livifuser_sim.world_layers import load_world  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world_json", type=Path)
    parser.add_argument("output_yaml", type=Path)
    parser.add_argument("--resolution-m", type=float, default=0.025)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_pgm = args.output_yaml.with_suffix(".pgm")
    output_manifest = args.output_yaml.with_suffix(".manifest.json")
    targets = (args.output_yaml, output_pgm, output_manifest)
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.overwrite:
        print(f"refusing to overwrite Nav2 map artifacts: {existing}", file=sys.stderr)
        return 3

    nav_map = rasterize_nav2_map(load_world(args.world_json), args.resolution_m)
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_pgm.write_bytes(nav_map.pgm_bytes())
    args.output_yaml.write_text(nav_map.yaml_text(output_pgm.name), encoding="utf-8")
    output_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "world_json": str(args.world_json.resolve()),
                "map_yaml": str(args.output_yaml.resolve()),
                "map_pgm": str(output_pgm.resolve()),
                "resolution_m": nav_map.resolution_m,
                "width": nav_map.width,
                "height": nav_map.height,
                "origin_xy_m": list(nav_map.origin_xy_m),
                "included_obstacles": list(nav_map.included_obstacles),
                "excluded_switchable_obstacles": list(
                    nav_map.excluded_switchable_obstacles
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
