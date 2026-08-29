#!/usr/bin/env python3
"""Replay validation scores on Kaggle T4x2 and seal the frozen score bundle."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import io
import json
import os
import platform
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from replay_sim_validation_scores import (  # noqa: E402
    CLOSED_LOOP_NAMES,
    CONFIG_SHA256,
    HETEROSCEDASTIC_PARTITIONS,
    RESULT_ARCHIVE_SHA256,
    SEEDS,
    deterministic_npz,
    sha256_bytes,
    sha256_file,
    validate_plan,
    validate_result_source,
)

from livifuser_nav.cloud_bundle import verify_cloud_bundle  # noqa: E402

AMENDMENT = Path(
    "docs/experiments/PREREGISTRATION_SIM_EVALUATION_EXECUTION_AMENDMENT_2026-08-24.md"
)
AMENDMENT_SHA256 = "8760474F1CCC6269BD23A28489DD01076891ECBF9E66A6F39BBF8E2838F6DCD7"
AUDIT_REPORT_SHA256 = "4D1CEA8F2D61EF76E1A48770FB6228F14683DAF6943C4932C06FCE0FB46611B3"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, indent=2) + chr(10)
    if path.exists():
        require(path.read_text(encoding="utf-8") == raw, f"existing JSON drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(raw, encoding="utf-8", newline=chr(10))
    temporary.replace(path)


def self_hash(payload: dict[str, Any], field: str) -> str:
    copy_payload = copy.deepcopy(payload)
    copy_payload.pop(field, None)
    raw = json.dumps(copy_payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(raw)


def worker_command(
    data_plan: Path,
    results_archive: Path,
    audit_report: Path,
    config: Path,
    output: Path,
    names: tuple[str, ...],
) -> list[str]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "replay_sim_validation_scores.py"),
        "--data-plan",
        str(data_plan),
        "--results-archive",
        str(results_archive),
        "--audit-report",
        str(audit_report),
        "--config",
        str(config),
        "--output",
        str(output),
        "--device",
        "cuda:0",
    ]
    for name in names:
        command.extend(["--run-name", name])
    return command


def run_worker(
    index: int,
    physical_cuda: str,
    data_plan: Path,
    results_archive: Path,
    audit_report: Path,
    config: Path,
    output_root: Path,
    print_lock: threading.Lock,
) -> dict[str, Any]:
    names = HETEROSCEDASTIC_PARTITIONS[index]
    output = output_root / f"worker_{index}"
    output.mkdir(parents=True, exist_ok=True)
    command = worker_command(data_plan, results_archive, audit_report, config, output, names)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = physical_cuda
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    accelerator = f"physical_cuda:{physical_cuda}"
    log_path = output_root / f"worker_{index}.log"
    with log_path.open("a", encoding="utf-8", newline=chr(10)) as log:
        log.write(f"ACCELERATOR {accelerator}{chr(10)}")
        log.write(f"COMMAND {subprocess.list2cmdline(command)}{chr(10)}")
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
        "run_names": list(names),
        "accelerator": accelerator,
        "returncode": returncode,
        "output": str(output),
        "log": str(log_path),
    }


def load_score(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def collect_records(
    output_root: Path, workers: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, str]]:
    records: list[dict[str, Any]] = []
    members: dict[str, bytes] = {}
    common: dict[str, np.ndarray] | None = None
    common_fields = (
        "episode_ids",
        "origin_rows",
        "mahalanobis_distance",
        "mahalanobis_cdf_sorted",
        "z_m",
        "episode_ids_unique",
        "mahalanobis_episode_max",
    )
    for worker in sorted(workers, key=lambda value: int(value["worker"])):
        require(worker["returncode"] == 0, f"score worker failed: {worker}")
        index = int(worker["worker"])
        summary_path = output_root / f"worker_{index}" / "worker_score_summary.json"
        summary_raw = summary_path.read_bytes()
        summary = json.loads(summary_raw)
        require(
            summary["run_names"] == list(HETEROSCEDASTIC_PARTITIONS[index]),
            f"worker {index} partition drift",
        )
        worker["summary_sha256"] = sha256_bytes(summary_raw)
        for record in summary["records"]:
            relative = str(record["score_file"])
            source = output_root / f"worker_{index}" / Path(relative)
            payload = source.read_bytes()
            require(
                len(payload) == int(record["score_size_bytes"])
                and sha256_bytes(payload) == record["score_sha256"],
                f"score file integrity failed: {relative}",
            )
            require(relative not in members, f"duplicate score member: {relative}")
            arrays = load_score(source)
            if common is None:
                common = {field: arrays[field] for field in common_fields}
            else:
                for field in common_fields:
                    require(
                        np.array_equal(common[field], arrays[field]),
                        f"common validation reference drift: {relative}:{field}",
                    )
            members[relative] = payload
            records.append(record)
    expected = {
        (name, seed)
        for names in HETEROSCEDASTIC_PARTITIONS.values()
        for name in names
        for seed in SEEDS
    }
    identities = {(row["name"], int(row["seed"])) for row in records}
    require(len(records) == 21 and identities == expected, "21-score identity drift")
    require(
        sum(bool(row["closed_loop_shortlist"]) for row in records) == 12,
        "closed-loop threshold identity drift",
    )
    require(common is not None, "no score records")
    common_hashes = {
        field: sha256_bytes(deterministic_npz({field: common[field]})) for field in common_fields
    }
    return (
        sorted(records, key=lambda row: (row["name"], int(row["seed"]))),
        members,
        common_hashes,
    )


def zip_payload(members: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, members[name])
    return output.getvalue()


def seal_bundle(
    bundle_output: Path,
    records: list[dict[str, Any]],
    score_members: dict[str, bytes],
    common_hashes: dict[str, str],
    plan_path: Path,
    plan: dict[str, Any],
    audit_report: dict[str, Any],
    cloud: dict[str, Any],
    workers: list[dict[str, Any]],
) -> dict[str, Any]:
    score_member_records = [
        {
            "name": name,
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
        for name, payload in sorted(score_members.items())
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_VALIDATION_ONLY",
        "purpose": "simulation_uncertainty_score_freeze_before_heldout",
        "amendment": {
            "path": AMENDMENT.as_posix(),
            "sha256": AMENDMENT_SHA256,
            "approval_date": "2026-08-24",
        },
        "identities": {
            "result_archive_sha256": RESULT_ARCHIVE_SHA256,
            "result_audit_report_sha256": AUDIT_REPORT_SHA256,
            "simulation_config_sha256": CONFIG_SHA256,
            "validation_data_plan_sha256": sha256_file(plan_path),
            "source_handoff_self_sha256": plan["source_handoff_self_sha256"],
            "cache_manifest_sha256": plan["cache_manifest_sha256"],
            "cache_manifest_self_sha256": plan["cache_manifest_self_sha256"],
            "backbone_contract_sha256": plan["backbone_contract_sha256"],
            "training_data_plan_sha256": audit_report["frozen_provenance"]["data_plan_sha256"],
            "validation_code_cloud_manifest_sha256": cloud["manifest_sha256"].upper(),
            "training_code_cloud_manifest_sha256": audit_report["frozen_provenance"][
                "cloud_manifest_sha256"
            ],
            "cloud_git_revision": cloud["git_revision"],
        },
        "heldout": {
            "attached": False,
            "opened": False,
            "hashed": False,
            "excluded_splits": ["test_id", "test_ood"],
        },
        "validation": {
            "episode_count": 30,
            "window_count": 9459,
            "episode_ids": plan["validation"]["episode_ids"],
            "common_array_sha256": common_hashes,
        },
        "score_contract": {
            "aleatoric": "mean(exp(clip(log_var,-5,2))) over Hx2",
            "cdf": "right_continuous; count(reference<=x)/N",
            "combined": "max(z_a,z_m)",
            "episode_reduction": "maximum_window_score",
            "threshold": "29th_order_statistic_of_30; strict_greater_than",
            "full_mean_only_excluded": True,
            "heteroscedastic_record_count": 21,
            "closed_loop_threshold_record_count": 12,
            "closed_loop_names": sorted(CLOSED_LOOP_NAMES),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "workers": [
                {
                    "worker": row["worker"],
                    "accelerator": row["accelerator"],
                    "run_names": row["run_names"],
                    "summary_sha256": row["summary_sha256"],
                }
                for row in sorted(workers, key=lambda value: int(value["worker"]))
            ],
        },
        "members": score_member_records,
        "records": records,
    }
    field = "manifest_sha256_excludes_self"
    manifest[field] = self_hash(manifest, field)
    manifest_payload = (json.dumps(manifest, indent=2) + chr(10)).encode()
    completion = {
        "schema_version": 1,
        "status": "COMPLETE",
        "manifest_file_sha256": sha256_bytes(manifest_payload),
        "manifest_sha256_excludes_self": manifest[field],
        "score_member_count": len(score_members),
        "heteroscedastic_record_count": len(records),
        "closed_loop_threshold_record_count": sum(
            bool(row["closed_loop_shortlist"]) for row in records
        ),
        "exact_bundle_member_count": len(score_members) + 2,
    }
    bundle_members = dict(score_members)
    bundle_members["SCORE_FREEZE_MANIFEST.json"] = manifest_payload
    bundle_members["SCORE_FREEZE_COMPLETE.json"] = (
        json.dumps(completion, indent=2) + chr(10)
    ).encode()
    payload = zip_payload(bundle_members)
    bundle_output.parent.mkdir(parents=True, exist_ok=True)
    if bundle_output.exists():
        require(bundle_output.read_bytes() == payload, "existing bundle drift")
    else:
        temporary = bundle_output.with_suffix(bundle_output.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(bundle_output)
    return {
        "bundle": str(bundle_output),
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "member_count": len(bundle_members),
        "manifest_file_sha256": sha256_bytes(manifest_payload),
        "manifest_self_sha256": manifest[field],
        "records": len(records),
        "closed_loop_thresholds": completion["closed_loop_threshold_record_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-plan", type=Path, required=True)
    parser.add_argument("--results-archive", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "simulation_sweep_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/livifuser_sim_validation_score_freeze_v1"),
    )
    parser.add_argument(
        "--bundle-output",
        type=Path,
        default=Path("/kaggle/working/livifuser_sim_validation_score_freeze_v1_bundle.zip"),
    )
    parser.add_argument("--cuda-device", action="append", required=True)
    args = parser.parse_args()

    amendment_path = REPOSITORY_ROOT / AMENDMENT
    require(sha256_file(amendment_path) == AMENDMENT_SHA256, "amendment hash drift")
    config = args.config.resolve()
    require(sha256_file(config) == CONFIG_SHA256, "simulation config hash drift")
    audit_path = args.audit_report.resolve()
    require(sha256_file(audit_path) == AUDIT_REPORT_SHA256, "audit report hash drift")
    audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
    require(
        audit_report["status"] == "PASS" and audit_report["integrity"]["heldout_excluded"] is True,
        "result audit is not a held-out-safe PASS",
    )
    results_archive = args.results_archive.resolve()
    result_source = validate_result_source(results_archive, audit_report)
    plan_path = args.data_plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    cloud = verify_cloud_bundle(REPOSITORY_ROOT)
    devices = [str(value) for value in args.cuda_device]
    require(
        len(devices) == 2 and len(set(devices)) == 2,
        "exactly two unique CUDA devices are required",
    )
    require(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "Kaggle T4x2 accelerator is unavailable",
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    execution = {
        "schema_version": 1,
        "amendment_sha256": AMENDMENT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "result_archive_sha256": RESULT_ARCHIVE_SHA256,
        "result_source": result_source,
        "audit_report_sha256": AUDIT_REPORT_SHA256,
        "data_plan": str(plan_path),
        "data_plan_sha256": sha256_file(plan_path),
        "heldout_attached": False,
        "partitions": {str(key): list(value) for key, value in HETEROSCEDASTIC_PARTITIONS.items()},
        "cuda_devices": devices,
        "cloud_bundle": cloud,
    }
    write_json_atomic(output_root / "execution_plan.json", execution)
    print_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                run_worker,
                index,
                devices[index],
                plan_path,
                results_archive,
                audit_path,
                config,
                output_root,
                print_lock,
            )
            for index in range(2)
        ]
        workers = [future.result() for future in futures]
    write_json_atomic(output_root / "execution_summary.json", {"workers": workers})
    records, score_members, common_hashes = collect_records(output_root, workers)
    report = seal_bundle(
        args.bundle_output.resolve(),
        records,
        score_members,
        common_hashes,
        plan_path,
        plan,
        audit_report,
        cloud,
        workers,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
