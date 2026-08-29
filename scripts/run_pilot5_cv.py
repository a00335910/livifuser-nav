#!/usr/bin/env python3
"""Run the frozen five-episode leave-one-out sweep with safe resumption."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from livifuser_nav.cloud_bundle import verify_cloud_bundle  # noqa: E402

EPISODES = (
    ("clear_001b", "train_lab_s1_clear_001b_policy_git_3f47712"),
    ("center_002b", "train_lab_s1_center_002b_policy_git_3f47712"),
    ("rightblock_003b", "train_lab_s1_rightblock_003b_policy_git_3f47712"),
    ("leftblock_004b", "train_lab_s1_leftblock_004b_policy_git_3f47712"),
    ("gap_005", "train_lab_s1_gap_005_policy_git_3f47712"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_identity() -> tuple[str, bool, dict[str, Any] | None]:
    bundle_manifest = REPOSITORY_ROOT / "cloud_bundle_manifest.json"
    if bundle_manifest.is_file():
        verification = verify_cloud_bundle(REPOSITORY_ROOT)
        return str(verification["git_revision"]), True, verification
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, not bool(status), None


def fold_command(
    config: Path,
    held_out: tuple[str, str],
    output: Path,
    device: str,
) -> list[str]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_baseline_sweep.py"),
        "--config",
        str(config),
        "--device",
        device,
    ]
    export_root = REPOSITORY_ROOT / "artifacts" / "export" / "protocol_clean_30"
    cache_root = REPOSITORY_ROOT / "artifacts" / "features" / "protocol_clean_30"
    for episode in EPISODES:
        export = export_root / episode[1]
        cache = cache_root / f"{episode[1]}_dino_s16"
        role = "val" if episode == held_out else "train"
        command.extend([f"--{role}-export", str(export)])
        command.extend([f"--{role}-cache", str(cache)])
    command.extend(["--output", str(output)])
    if output.exists() and any(output.iterdir()):
        command.append("--resume")
    return command


def run_fold(
    config: Path,
    held_out: tuple[str, str],
    output_root: Path,
    print_lock: threading.Lock,
    accelerator_queue: queue.Queue[str | None],
) -> dict[str, str | int]:
    accelerator = accelerator_queue.get()
    try:
        environment = os.environ.copy()
        if accelerator is None:
            device = "cpu"
            accelerator_label = "cpu"
        else:
            environment["CUDA_VISIBLE_DEVICES"] = accelerator
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            device = "cuda:0"
            accelerator_label = f"physical_cuda:{accelerator}"
        fold_output = output_root / f"held_out_{held_out[0]}"
        command = fold_command(config, held_out, fold_output, device)
        log_path = output_root / f"held_out_{held_out[0]}.log"
        with log_path.open("a", encoding="utf-8", newline="\n") as log:
            log.write(
                f"ACCELERATOR {accelerator_label}\n"
                "COMMAND " + subprocess.list2cmdline(command) + "\n"
            )
            log.flush()
            with print_lock:
                print(
                    f"starting fold held_out={held_out[0]} "
                    f"accelerator={accelerator_label} log={log_path}",
                    flush=True,
                )
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                with print_lock:
                    print(f"[{held_out[0]}] {line}", end="", flush=True)
            returncode = process.wait()
        with print_lock:
            print(f"fold held_out={held_out[0]} exit={returncode}", flush=True)
        return {
            "held_out": held_out[0],
            "returncode": returncode,
            "accelerator": accelerator_label,
            "output": str(fold_output),
            "log": str(log_path),
        }
    finally:
        accelerator_queue.put(accelerator)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "baseline_sweep_pilot5_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "artifacts"
            / "experiments"
            / "pilot5_leave_one_episode_out_kaggle_t4x2_v1"
        ),
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--cuda-device",
        action="append",
        help=(
            "Physical CUDA index assigned exclusively to one worker. "
            "Repeat for dual-GPU execution; omit for CPU."
        ),
    )
    args = parser.parse_args()
    config = args.config.resolve()
    output_root = args.output_root.resolve()
    configuration = json.loads(config.read_text(encoding="utf-8"))
    if args.max_workers <= 0:
        raise ValueError("max-workers must be positive")
    cuda_devices = [] if args.cuda_device is None else args.cuda_device
    if len(set(cuda_devices)) != len(cuda_devices):
        raise ValueError("CUDA device assignments must be unique")
    if cuda_devices:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA devices were requested but CUDA is unavailable")
        if args.max_workers != len(cuda_devices):
            raise ValueError("max-workers must equal the number of CUDA devices")
        for value in cuda_devices:
            index = int(value)
            if index < 0 or index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device is unavailable: {value}")
    else:
        required_threads = args.max_workers * int(configuration["torch_threads"])
        available_threads = os.cpu_count() or 1
        if required_threads > available_threads:
            raise ValueError(
                f"workers require {required_threads} threads but host reports "
                f"{available_threads}"
            )
    revision, clean, bundle_verification = source_identity()
    if not clean:
        raise RuntimeError("refusing to start scientific sweep from a dirty worktree")

    for _, stem in EPISODES:
        export = (
            REPOSITORY_ROOT / "artifacts" / "export" / "protocol_clean_30" / stem
        )
        cache = (
            REPOSITORY_ROOT
            / "artifacts"
            / "features"
            / "protocol_clean_30"
            / f"{stem}_dino_s16"
        )
        for required in (export / "manifest.json", cache / "manifest.json"):
            if not required.is_file():
                raise FileNotFoundError(required)

    output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "sweep_id": configuration["sweep_id"],
        "config": str(config),
        "config_sha256": sha256_file(config),
        "git_revision": revision,
        "folds": [episode[0] for episode in EPISODES],
        "model_seed_results_per_fold": len(configuration["runs"])
        * len(configuration["seeds"]),
        "max_workers": args.max_workers,
        "torch_threads_per_worker": int(configuration["torch_threads"]),
        "execution_backend": "cuda_isolated_processes" if cuda_devices else "cpu",
        "cuda_devices": [
            {
                "physical_index": int(value),
                "name": torch.cuda.get_device_name(int(value)),
                "capability": list(torch.cuda.get_device_capability(int(value))),
            }
            for value in cuda_devices
        ],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "cloud_bundle": bundle_verification,
    }
    plan_path = output_root / "cv_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise RuntimeError("existing CV plan differs from this invocation")
    else:
        write_json_atomic(plan_path, plan)

    print_lock = threading.Lock()
    accelerator_queue: queue.Queue[str | None] = queue.Queue()
    for value in cuda_devices or [None] * args.max_workers:
        accelerator_queue.put(value)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(
                run_fold,
                config,
                episode,
                output_root,
                print_lock,
                accelerator_queue,
            )
            for episode in EPISODES
        ]
        results = [future.result() for future in futures]
    summary = {"plan": plan, "folds": results}
    write_json_atomic(output_root / "cv_execution_summary.json", summary)
    failures = [item for item in results if item["returncode"] != 0]
    if failures:
        print(json.dumps({"failures": failures}, indent=2))
        return 1
    print(json.dumps({"summary": str(output_root / "cv_execution_summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
