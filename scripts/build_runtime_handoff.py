#!/usr/bin/env python3
"""Seal the immutable closed-loop runtime handoff (amendment section 11)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from livifuser_nav.runtime_handoff import build_runtime_handoff  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=ROOT / "artifacts/runtime")
    parser.add_argument(
        "--backbone",
        type=Path,
        default=ROOT / "artifacts/livifuser_dinov3_vits16plus_backbone_c93d816_bundle.zip",
    )
    parser.add_argument(
        "--policies",
        type=Path,
        default=ROOT / "artifacts/livifuser_sim_closed_loop_policy_payload_v1_bundle.zip",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/livifuser_runtime_v1_bundle.zip"
    )
    args = parser.parse_args()
    report = build_runtime_handoff(
        ROOT,
        args.evidence_root,
        args.output,
        backbone_bundle=args.backbone,
        policy_payload=args.policies,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
