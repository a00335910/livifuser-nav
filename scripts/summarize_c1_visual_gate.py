#!/usr/bin/env python3
"""Evaluate paired C0/C1 simulator verifier outputs against the frozen gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_sim.visual_conditions import evaluate_c1_development_gate  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("c0_verifier", type=Path)
    parser.add_argument("c1_verifier", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")

    c0 = json.loads(args.c0_verifier.read_text(encoding="utf-8"))
    c1 = json.loads(args.c1_verifier.read_text(encoding="utf-8"))
    if not c0.get("valid") or not c1.get("valid"):
        raise ValueError("both simulator contract verifiers must be valid")
    result = evaluate_c1_development_gate(
        c0.get("first_image_statistics", {}),
        c1.get("first_image_statistics", {}),
    )
    result["inputs"] = {
        "c0": {"path": str(args.c0_verifier), "sha256": _sha256(args.c0_verifier)},
        "c1": {"path": str(args.c1_verifier), "sha256": _sha256(args.c1_verifier)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
