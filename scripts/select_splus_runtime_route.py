#!/usr/bin/env python3
"""Seal the frozen S+/16 runtime-route decision from benchmark evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from livifuser_nav.backbone_handoff import json_bytes, sha256_bytes, sha256_file  # noqa: E402

VARIANTS = ("full", "lidar_only", "concat", "rgb_only")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_report(
    path: Path, route: str, config_sha256: str, config_identity_key: str
) -> dict[str, Any]:
    raw = path.read_bytes()
    report = json.loads(raw)
    declared = report.pop("report_sha256_excludes_self")
    require(sha256_bytes(json_bytes(report)) == declared, f"report self-hash failed: {path}")
    report["report_sha256_excludes_self"] = declared
    require(
        report["status"] == "complete" and report["route"] == route,
        f"report route/status drifted: {path}",
    )
    require(report["timed_iterations"] == 200, f"report count drifted: {path}")
    require(
        report["identities"][config_identity_key] == config_sha256,
        f"report config drifted: {path}",
    )
    values = report["raw_latency_ms"]["complete_path"]
    require(len(values) == 200, f"raw complete-path count drifted: {path}")
    deadline = float(report["deadline_ms"])
    misses = sum(float(value) >= deadline for value in values)
    p99 = float(__import__("numpy").percentile(values, 99))
    require(misses == int(report["missed_deadline_count"]), f"miss count drifted: {path}")
    require(
        abs(p99 - float(report["latency"]["complete_path"]["p99_ms"])) < 1e-12,
        f"p99 drifted: {path}",
    )
    expected_pass = misses == 0 and p99 < deadline
    require(bool(report["deadline_pass"]) == expected_pass, f"deadline result drifted: {path}")
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "file_sha256": sha256_bytes(raw),
        "report_sha256_excludes_self": declared,
        "variant": report["variant"],
        "timed_iterations": 200,
        "complete_path_median_ms": report["latency"]["complete_path"]["median_ms"],
        "complete_path_p99_ms": p99,
        "complete_path_maximum_ms": report["latency"]["complete_path"]["maximum_ms"],
        "missed_deadline_count": misses,
        "deadline_pass": expected_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu-config",
        type=Path,
        default=ROOT / "config/dinov3_splus_cpu_benchmark_v1.json",
    )
    parser.add_argument(
        "--directml-config",
        type=Path,
        default=ROOT / "config/dinov3_splus_directml_benchmark_v1.json",
    )
    parser.add_argument(
        "--parity-report",
        type=Path,
        default=ROOT / "artifacts/runtime/directml_parity_v1.json",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "artifacts/runtime/benchmarks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/runtime/RUNTIME_ROUTE_DECISION.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite route decision: {args.output}")

    cpu_config_raw = args.cpu_config.read_bytes()
    dml_config_raw = args.directml_config.read_bytes()
    cpu_config_sha = sha256_bytes(cpu_config_raw)
    dml_config_sha = sha256_bytes(dml_config_raw)
    dml_config = json.loads(dml_config_raw)
    parity_raw = args.parity_report.read_bytes()
    parity = json.loads(parity_raw)
    require(
        sha256_bytes(parity_raw) == dml_config["parity_report_sha256"],
        "parity report file identity drifted",
    )
    require(parity["parity_pass"] is True, "DirectML parity did not pass")

    cpu = []
    directml = []
    for variant in VARIANTS:
        cpu.append(
            verify_report(
                args.benchmark_root / f"final_{variant}_threads_6.json",
                "exact_pytorch_cpu_float32_eager",
                cpu_config_sha,
                "benchmark_config_sha256",
            )
        )
        directml.append(
            verify_report(
                args.benchmark_root / f"directml_{variant}.json",
                "parity_qualified_splus_directml_rx5500m_plus_exact_pytorch_cpu_policy",
                dml_config_sha,
                "config_sha256",
            )
        )
    require({row["variant"] for row in cpu} == set(VARIANTS), "CPU variant set drifted")
    require({row["variant"] for row in directml} == set(VARIANTS), "DirectML variant set drifted")
    cpu_pass = all(row["deadline_pass"] for row in cpu)
    directml_pass = parity["parity_pass"] and all(row["deadline_pass"] for row in directml)
    require(
        not cpu_pass and not directml_pass,
        "recorded evidence no longer matches the frozen fallback branch",
    )

    decision: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "FROZEN_NOT_CONFIRMATORY_READY",
        "decision": {
            "selected_correctness_route": "exact_pytorch_cpu_float32_eager",
            "selected_cpu_intraop_threads": 6,
            "selected_cpu_interop_threads": 1,
            "directml_accepted": False,
            "directml_rejection_reason": (
                "parity passed, but every variant had at least one >=100 ms complete-path "
                "sample under the prospectively frozen zero-miss acceptance rule"
            ),
            "ten_hz_readiness": False,
            "confirmatory_launch_authorized": False,
            "next_action": (
                "implement and recorded-input-test the exact CPU fallback, preserve the failed "
                "deadline evidence, and do not launch confirmatory evaluation"
            ),
        },
        "rules": {
            "cpu_acceptance": json.loads(cpu_config_raw)["final_route_evidence"]["acceptance"],
            "directml_acceptance": dml_config["benchmark"]["acceptance"],
            "fallback": dml_config["route_selection"]["otherwise"],
            "deadline_ms": 100.0,
        },
        "directml_parity": {
            "path": str(args.parity_report.relative_to(ROOT)).replace("\\", "/"),
            "file_sha256": sha256_bytes(parity_raw),
            "report_sha256_excludes_self": parity["report_sha256_excludes_self"],
            "pass": True,
            "maxima": parity["maxima"],
            "onnx": parity["onnx"],
        },
        "cpu_reports": cpu,
        "directml_reports": directml,
        "identities": {
            "cpu_config_sha256": cpu_config_sha,
            "directml_config_sha256": dml_config_sha,
            "parity_report_sha256": sha256_bytes(parity_raw),
            "selector_code_sha256": sha256_file(Path(__file__)),
        },
    }
    decision["decision_sha256_excludes_self"] = sha256_bytes(json_bytes(decision))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json_bytes(decision))
    print(
        json_bytes(
            {
                "output": str(args.output.resolve()),
                "file_sha256": sha256_file(args.output),
                "decision_sha256_excludes_self": decision["decision_sha256_excludes_self"],
                "status": decision["status"],
                "selected_correctness_route": decision["decision"]["selected_correctness_route"],
                "ten_hz_readiness": decision["decision"]["ten_hz_readiness"],
                "confirmatory_launch_authorized": False,
            }
        ).decode(),
        end="",
    )


if __name__ == "__main__":
    main()
