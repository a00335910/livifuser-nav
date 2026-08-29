#!/usr/bin/env python3
"""Non-scientific device smoke test using the real cached training batches."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import numpy as np  # noqa: E402
import run_baseline_sweep as sweep  # noqa: E402
import torch  # noqa: E402

TRAIN_STEMS = (
    "train_lab_s1_center_002b_policy_git_3f47712",
    "train_lab_s1_rightblock_003b_policy_git_3f47712",
    "train_lab_s1_leftblock_004b_policy_git_3f47712",
    "train_lab_s1_gap_005_policy_git_3f47712",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "baseline_sweep_pilot5_v1.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    smoke_config = copy.deepcopy(config)
    smoke_config["warmup_steps"] = args.steps
    smoke_config["nll_steps"] = 0
    smoke_config["log_every_steps"] = args.steps
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(int(config["torch_threads"]))
    device = sweep.resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    export_root = REPOSITORY_ROOT / "artifacts" / "export" / "protocol_clean_30"
    cache_root = REPOSITORY_ROOT / "artifacts" / "features" / "protocol_clean_30"
    exports = [export_root / stem for stem in TRAIN_STEMS]
    caches = [cache_root / f"{stem}_dino_s16" for stem in TRAIN_STEMS]
    dataset, feature_caches, tokens = sweep.load_split(
        exports, caches, smoke_config, "smoke_train"
    )
    _, record = sweep.train_one(
        config["runs"][0],
        int(config["seeds"][0]),
        smoke_config,
        dataset,
        feature_caches,
        tokens,
        device,
    )
    seconds_per_step = record["seconds"] / args.steps
    planned_results = len(config["runs"]) * len(config["seeds"]) * 5
    full_steps = int(config["warmup_steps"]) + int(config["nll_steps"])
    result = {
        "disposition": "non-scientific CUDA smoke benchmark; never an experiment result",
        "device": sweep.device_provenance(device),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "smoke_steps": args.steps,
        "seconds": record["seconds"],
        "seconds_per_step": seconds_per_step,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "conservative_full_model_equivalent_hours_t4x2": (
            seconds_per_step * full_steps * planned_results / 2.0 / 3600.0
        ),
        "warning": (
            "Estimate assumes every ablation costs as much as the full model; "
            "actual sweep should be faster."
        ),
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
