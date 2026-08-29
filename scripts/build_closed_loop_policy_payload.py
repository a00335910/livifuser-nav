#!/usr/bin/env python3
"""Build the deterministic 12-checkpoint closed-loop policy payload."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livifuser_nav.backbone_handoff import json_bytes  # noqa: E402
from livifuser_nav.policy_payload import seal_policy_payload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts/livifuser_simulation_sweep_v1_results.zip"),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("artifacts/livifuser_sim_validation_score_freeze_v1_bundle.zip"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/simulation_sweep_v1.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/livifuser_sim_closed_loop_policy_payload_v1_bundle.zip"),
    )
    args = parser.parse_args()
    print(
        json_bytes(seal_policy_payload(args.results, args.scores, args.config, args.output))
        .decode("utf-8"),
        end="",
    )


if __name__ == "__main__":
    main()
