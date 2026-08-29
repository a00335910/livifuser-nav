#!/usr/bin/env python3
"""Verify every source/export/cache file extracted from a cloud bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from livifuser_nav.cloud_bundle import verify_cloud_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args()
    result = verify_cloud_bundle(args.repository_root.resolve())
    print(json.dumps({"valid": True, **result}, indent=2))


if __name__ == "__main__":
    main()
