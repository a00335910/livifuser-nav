#!/usr/bin/env python3
"""Download and seal the exact official DINOv3 S+/16 runtime snapshot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livifuser_nav.backbone_handoff import (  # noqa: E402
    BUNDLE_FILENAME,
    download_snapshot,
    seal_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    token = os.environ.get(args.token_env, "")
    snapshot = download_snapshot(token=token, cache_dir=args.cache_dir)
    report = seal_snapshot(snapshot, args.output_dir / BUNDLE_FILENAME)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

