#!/usr/bin/env python3
"""Verify a returned official DINOv3 S+/16 backbone bundle without extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livifuser_nav.backbone_handoff import json_bytes, verify_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_bundle(args.bundle)
    payload = json_bytes(report)
    if args.report is not None:
        if args.report.exists():
            raise FileExistsError(f"refusing to overwrite verification report: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(payload)
    print(payload.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
