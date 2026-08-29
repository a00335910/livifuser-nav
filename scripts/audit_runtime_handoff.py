#!/usr/bin/env python3
"""Independently audit a sealed runtime handoff. Read-only; never builds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from livifuser_nav.runtime_handoff import audit_runtime_handoff  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    report = audit_runtime_handoff(args.bundle)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "audit_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
