#!/usr/bin/env python3
"""Verify, then safely unpack, the sealed RunPod input into a new directory."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from livifuser_nav.runpod_handoff import BUNDLE_ROOT, verify_runpod_handoff  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    report = verify_runpod_handoff(args.bundle)
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace destination: {destination}")
    destination.mkdir(parents=True)
    with zipfile.ZipFile(args.bundle) as archive:
        prefix = f"{BUNDLE_ROOT}/"
        for info in archive.infolist():
            if not info.filename.startswith(prefix):
                raise ValueError(f"member escaped bundle root: {info.filename}")
            relative = info.filename[len(prefix) :]
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    print(
        json.dumps(
            {**report, "status": "verified_and_unpacked", "destination": str(destination)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
