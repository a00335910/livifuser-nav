#!/usr/bin/env python3
"""Independently verify the sealed 12-checkpoint policy payload in place."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livifuser_nav.backbone_handoff import json_bytes  # noqa: E402
from livifuser_nav.policy_payload import verify_policy_payload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    print(json_bytes(verify_policy_payload(args.bundle)).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
