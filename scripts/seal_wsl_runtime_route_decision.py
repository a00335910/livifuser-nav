#!/usr/bin/env python3
"""Supersede the native route record with actual ROS/WSL CPU evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from livifuser_nav.backbone_handoff import json_bytes, sha256_bytes, sha256_file  # noqa: E402

VARIANTS = ("full", "lidar_only", "concat", "rgb_only")
THREADS = (1, 2, 4, 6)
NATIVE_DECISION_SHA256 = "758BD09F40B65CE4BA0190BAFF9392D8B31CB10E614AFFCA7B319740EAAB8D9D"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_self_hash(report: dict[str, Any]) -> str:
    copy = dict(report)
    declared = copy.pop("report_sha256_excludes_self")
    require(sha256_bytes(json_bytes(copy)) == declared, "benchmark report self-hash failed")
    return declared


def report_record(path: Path, expected_count: int) -> dict[str, Any]:
    raw = path.read_bytes()
    report = json.loads(raw)
    declared = verify_self_hash(report)
    values = np.asarray(report["raw_latency_ms"]["complete_path"], dtype=np.float64)
    require(values.shape == (expected_count,), f"benchmark count drifted: {path}")
    p99 = float(np.percentile(values, 99))
    require(
        abs(p99 - float(report["latency"]["complete_path"]["p99_ms"])) < 1e-12,
        f"benchmark p99 drifted: {path}",
    )
    misses = int(np.count_nonzero(values >= float(report["deadline_ms"])))
    require(misses == int(report["missed_deadline_count"]), f"miss count drifted: {path}")
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "file_sha256": sha256_bytes(raw),
        "report_sha256_excludes_self": declared,
        "variant": report["variant"],
        "threads": report["threads"],
        "timed_iterations": expected_count,
        "complete_path_median_ms": report["latency"]["complete_path"]["median_ms"],
        "complete_path_p99_ms": p99,
        "complete_path_maximum_ms": report["latency"]["complete_path"]["maximum_ms"],
        "missed_deadline_count": misses,
        "deadline_pass": bool(report["deadline_pass"]),
        "software": report["software"],
        "host": report["host"],
    }


def main() -> None:
    native_path = ROOT / "artifacts/runtime/RUNTIME_ROUTE_DECISION.json"
    output = ROOT / "artifacts/runtime/RUNTIME_ROUTE_DECISION_V2.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite WSL route decision: {output}")
    require(
        sha256_file(native_path) == NATIVE_DECISION_SHA256, "native route decision identity drifted"
    )
    native = json.loads(native_path.read_text("utf-8"))
    native_copy = dict(native)
    native_self = native_copy.pop("decision_sha256_excludes_self")
    require(
        sha256_bytes(json_bytes(native_copy)) == native_self,
        "native route decision self-hash failed",
    )

    benchmark_root = ROOT / "artifacts/runtime/benchmarks"
    screen = [
        report_record(benchmark_root / f"wsl_screen_full_threads_{threads}.json", 40)
        for threads in THREADS
    ]
    selected = min(screen, key=lambda row: (row["complete_path_p99_ms"], row["threads"]["intraop"]))
    require(selected["threads"]["intraop"] == 4, "WSL thread selection drifted")
    final = [
        report_record(benchmark_root / f"wsl_final_{variant}_threads_4.json", 200)
        for variant in VARIANTS
    ]
    require({row["variant"] for row in final} == set(VARIANTS), "WSL variant set drifted")
    require(
        all(row["missed_deadline_count"] == 200 for row in final),
        "WSL evidence no longer matches the frozen decision branch",
    )

    decision: dict[str, Any] = {
        "schema_version": "2.0.0",
        "status": "FROZEN_NOT_CONFIRMATORY_READY",
        "supersedes": {
            "path": "artifacts/runtime/RUNTIME_ROUTE_DECISION.json",
            "file_sha256": NATIVE_DECISION_SHA256,
            "decision_sha256_excludes_self": native_self,
            "reason": "actual simulator runtime is ROS 2 Humble inside Ubuntu-TB3",
        },
        "decision": {
            "selected_correctness_route": "wsl_exact_pytorch_cpu_float32_eager",
            "selected_cpu_intraop_threads": 4,
            "selected_cpu_interop_threads": 1,
            "directml_accepted": False,
            "directml_status": (
                "native Windows parity passed, but the frozen zero-miss timing gate failed; "
                "DirectML is not directly available to the WSL ROS process"
            ),
            "ten_hz_wall_latency_ready": False,
            "confirmatory_launch_authorized": False,
            "proposed_simulation_only_deviation": (
                "run Gazebo below real time while retaining the frozen 10 Hz simulation-time "
                "control clock; report wall latency as a failed architecture target"
            ),
            "deviation_approved": False,
            "next_action_requires_user_decision": True,
        },
        "wsl_environment": {
            "distribution": "Ubuntu-TB3",
            "python": "3.10.12",
            "torch": "2.13.0+cpu",
            "transformers": "4.56.0",
            "huggingface_hub": "0.34.4",
            "safetensors": "0.6.2",
            "numpy": "2.0.2",
            "pillow": "11.3.0",
            "scipy_compatibility_override": "1.15.3",
            "ros": "Humble",
            "rclpy_import_verified": True,
            "venv": "/home/a00335910/.venvs/livifuser-runtime-v1",
        },
        "thread_screen": screen,
        "selected_thread_record": selected,
        "final_reports": final,
        "native_windows_evidence": native,
        "identities": {
            "backbone_bundle_sha256": (
                "0F9F7CB99A955AE0B817762CC08565F0D3BD820CDD1692D71DC6B05E2CD9E9F3"
            ),
            "policy_payload_sha256": (
                "3A989EADD0DB8D995993D2042124E34C2C51FAAFC1A9C74EC60106E8A182C162"
            ),
            "selector_code_sha256": sha256_file(Path(__file__)),
        },
    }
    decision["decision_sha256_excludes_self"] = sha256_bytes(json_bytes(decision))
    output.write_bytes(json_bytes(decision))
    print(
        json_bytes(
            {
                "output": str(output.resolve()),
                "file_sha256": sha256_file(output),
                "decision_sha256_excludes_self": decision["decision_sha256_excludes_self"],
                "status": decision["status"],
                "selected_correctness_route": decision["decision"]["selected_correctness_route"],
                "selected_threads": 4,
                "ten_hz_wall_latency_ready": False,
                "deviation_approved": False,
                "confirmatory_launch_authorized": False,
            }
        ).decode(),
        end="",
    )


if __name__ == "__main__":
    main()
