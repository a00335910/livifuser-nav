#!/usr/bin/env python3
"""Run the frozen 24-result simulation sweep on two isolated Kaggle GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
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

CONFIG_SHA256 = "76680BCE45A67D5D91F42660D5EC25F90450B281838CE311E329477C4E36F09E"
RUN_PARTITIONS = (
    ("full", "rgb_only", "no_fov_mask", "no_temporal"),
    ("lidar_only", "concat", "no_gate", "full_mean_only"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_data_plan(plan: dict[str, Any]) -> None:
    if plan.get("heldout_attached") is not False:
        raise ValueError("data plan does not prove held-out exclusion")
    if plan.get("excluded_splits") != ["test_id", "test_ood"]:
        raise ValueError("data plan held-out split guard drifted")
    expected = {
        "train": (120, 56_128, 41_367),
        "validation": (30, 13_125, 9_459),
    }
    identities: dict[str, set[str]] = {}
    for split, counts in expected.items():
        record = plan[split]
        observed = (
            int(record["episode_count"]),
            int(record["accepted_samples"]),
            int(record["windows_k8_h8"]),
        )
        if observed != counts:
            raise ValueError(f"{split} count drift: {observed}")
        ids = [str(value) for value in record["episode_ids"]]
        if len(ids) != len(set(ids)) or len(ids) != counts[0]:
            raise ValueError(f"{split} episode identity drift")
        if len(record["exports"]) != counts[0] or len(record["caches"]) != counts[0]:
            raise ValueError(f"{split} path count drift")
        identities[split] = set(ids)
    if identities["train"].intersection(identities["validation"]):
        raise ValueError("train and validation episodes overlap")
    if any(
        value.startswith(("test_id_", "test_ood_"))
        for values in identities.values()
        for value in values
    ):
        raise ValueError("held-out episode entered the data plan")


def worker_command(
    config: Path,
    plan: dict[str, Any],
    run_names: tuple[str, ...],
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
    for split, argument in (("train", "train"), ("validation", "val")):
        for export, cache in zip(plan[split]["exports"], plan[split]["caches"], strict=True):
            command.extend([f"--{argument}-export", str(export)])
            command.extend([f"--{argument}-cache", str(cache)])
    for name in run_names:
        command.extend(["--run-name", name])
    command.extend(["--output", str(output)])
    if output.exists() and any(output.iterdir()):
        command.append("--resume")
    return command


def run_worker(
    index: int,
    config: Path,
    plan: dict[str, Any],
    output_root: Path,
    physical_cuda: str,
    print_lock: threading.Lock,
) -> dict[str, Any]:
    run_names = RUN_PARTITIONS[index]
    worker_output = output_root / f"worker_{index}"
    worker_output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = physical_cuda
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    device = "cuda:0"
    accelerator = f"physical_cuda:{physical_cuda}"
    command = worker_command(config, plan, run_names, worker_output, device)
    log_path = output_root / f"worker_{index}.log"
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"ACCELERATOR {accelerator}\nCOMMAND {subprocess.list2cmdline(command)}\n")
        log.flush()
        with print_lock:
            print(f"starting worker={index} accelerator={accelerator}", flush=True)
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
                print(f"[worker {index}] {line}", end="", flush=True)
        returncode = process.wait()
    return {
        "worker": index,
        "run_names": list(run_names),
        "accelerator": accelerator,
        "returncode": returncode,
        "output": str(worker_output),
        "log": str(log_path),
    }


def merge_results(output_root: Path, workers: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    summaries = []
    for worker in workers:
        if worker["returncode"] != 0:
            raise RuntimeError(f"training worker failed: {worker}")
        summary_path = Path(worker["output"]) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(
            {
                "worker": worker["worker"],
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            }
        )
        results.extend(summary["results"])
    identities = [(row["name"], int(row["seed"])) for row in results]
    expected = {
        (name, seed)
        for partition in RUN_PARTITIONS
        for name in partition
        for seed in (20260805, 20260806, 20260807)
    }
    if len(identities) != 24 or set(identities) != expected:
        raise RuntimeError("merged 24-result identity set mismatch")
    combined = {
        "schema_version": 1,
        "sweep_id": "confirmatory_v3_simulation_train_val_v1",
        "config_sha256": CONFIG_SHA256,
        "result_count": 24,
        "worker_summaries": summaries,
        "results": sorted(results, key=lambda row: (row["name"], int(row["seed"]))),
    }
    write_json_atomic(output_root / "summary.json", combined)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-plan", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "simulation_sweep_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/livifuser_simulation_sweep_v1"),
    )
    parser.add_argument("--cuda-device", action="append", required=True)
    args = parser.parse_args()
    config = args.config.resolve()
    if sha256_file(config) != CONFIG_SHA256:
        raise RuntimeError("frozen simulation config hash mismatch")
    plan_path = args.data_plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_data_plan(plan)
    cloud = verify_cloud_bundle(REPOSITORY_ROOT)
    cuda_devices = args.cuda_device
    if len(cuda_devices) != 2 or len(set(cuda_devices)) != 2:
        raise ValueError("exactly two unique CUDA devices are required")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Kaggle T4x2 accelerator is unavailable")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    execution_plan = {
        "schema_version": 1,
        "config": str(config),
        "config_sha256": CONFIG_SHA256,
        "data_plan": str(plan_path),
        "data_plan_sha256": sha256_file(plan_path),
        "cloud_bundle": cloud,
        "partitions": [list(value) for value in RUN_PARTITIONS],
        "cuda_devices": cuda_devices,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }
    plan_output = output_root / "execution_plan.json"
    if plan_output.exists():
        if json.loads(plan_output.read_text(encoding="utf-8")) != execution_plan:
            raise RuntimeError("existing execution plan differs")
    else:
        write_json_atomic(plan_output, execution_plan)
    print_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        for index in range(2):
            futures.append(
                pool.submit(
                    run_worker,
                    index,
                    config,
                    plan,
                    output_root,
                    cuda_devices[index],
                    print_lock,
                )
            )
        workers = [future.result() for future in futures]
    write_json_atomic(output_root / "execution_summary.json", {"workers": workers})
    failures = [worker for worker in workers if worker["returncode"] != 0]
    if failures:
        print(json.dumps({"failures": failures}, indent=2))
        return 1
    combined = merge_results(output_root, workers)
    print(
        json.dumps(
            {"summary": str(output_root / "summary.json"), "results": combined["result_count"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
