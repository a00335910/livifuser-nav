#!/usr/bin/env python3
"""Cache frozen temporary DINOv3 ViT-S/16 features for one accepted export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from livifuser_nav.dino_cache import cache_dino_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=Path(r"D:\dinov3_onnx\dinov3_small_224.onnx")
    )
    parser.add_argument(
        "--view",
        choices=("policy", "sensor"),
        default="policy",
        help="sensor permits action-free exports such as ood_probe recordings",
    )
    args = parser.parse_args()
    result = cache_dino_features(args.export, args.output, args.model, expected_view=args.view)
    print(json.dumps({"run_id": result["run_id"], "timing": result["timing"]}, indent=2))


if __name__ == "__main__":
    main()

