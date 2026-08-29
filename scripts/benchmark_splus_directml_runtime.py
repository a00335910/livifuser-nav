#!/usr/bin/env python3
"""Benchmark the parity-qualified DirectML backbone plus exact CPU policy."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_splus_cpu_runtime import (
    load_policy_material,
    percentile_summary,
    verify_sources,
)

from livifuser_nav.backbone_handoff import json_bytes, sha256_bytes, sha256_file
from livifuser_nav.evaluation import mahalanobis_distances
from livifuser_nav.heldout_evaluation import right_continuous_cdf
from livifuser_nav.learning_data import preprocess_rgb, tokenize_lidar
from livifuser_nav.model import LiViFuserPolicy

STAGES = (
    "rgb_preprocess",
    "splus_directml_forward_and_copy",
    "lidar_tokenize",
    "policy_stack_and_forward",
    "uncertainty_and_supervisor",
    "complete_path",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def dml_context(
    session: Any, input_name: str, run: Any, row: int
) -> tuple[tuple[np.ndarray, ...], np.ndarray, list[float]]:
    start = time.perf_counter_ns()
    pixel_values = preprocess_rgb(np.asarray(run.rgb[row]))[None]
    t1 = time.perf_counter_ns()
    patches, pooled = session.run(None, {input_name: pixel_values})
    require(
        patches.shape == (1, 49, 384) and pooled.shape == (1, 384), "DirectML output shape drifted"
    )
    patches = np.asarray(patches[0], dtype=np.float32)
    pooled = np.asarray(pooled[0], dtype=np.float32)
    t2 = time.perf_counter_ns()
    tokens = tokenize_lidar(
        run.scan_ranges[row],
        int(run.vectors["scan_beam_count"][row]),
        float(run.vectors["scan_angle_increment_rad"][row]),
        run.manifest,
        sectors=80,
        range_clip_m=10.0,
        visual_radius=1,
    )
    t3 = time.perf_counter_ns()
    require(
        np.all(np.isfinite(patches)) and np.all(np.isfinite(pooled)),
        "DirectML output is non-finite",
    )
    context = (
        patches,
        tokens.features,
        tokens.visual_mask,
        tokens.in_fov,
        np.asarray(run.vectors["goal"][row], dtype=np.float32),
        np.asarray(run.vectors["robot_state"][row], dtype=np.float32),
    )
    return (
        context,
        pooled,
        [
            (t1 - start) / 1e6,
            (t2 - t1) / 1e6,
            (t3 - t2) / 1e6,
        ],
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import onnxruntime as ort
    import PIL
    import torch

    config_payload = args.config.read_bytes()
    config = json.loads(config_payload)
    require(
        config["status"] == "FROZEN_AFTER_PARITY_PASS_BEFORE_FIRST_DIRECTML_TIMING",
        "DirectML timing config is not frozen",
    )
    require(args.variant in config["benchmark"]["variants"], "variant is not frozen")
    require(args.seed == int(config["benchmark"]["seed"]), "seed is not frozen")
    require(
        args.iterations == int(config["benchmark"]["timed_iterations_per_variant"]),
        "timed iteration count drifted",
    )
    require(
        args.warmup == int(config["benchmark"]["warmup_complete_path_iterations"]),
        "warmup iteration count drifted",
    )
    require(sha256_file(args.onnx) == config["onnx_sha256"], "ONNX identity drifted")
    require(args.onnx.stat().st_size == int(config["onnx_size_bytes"]), "ONNX size drifted")
    parity = json.loads(args.parity_report.read_text("utf-8"))
    require(
        sha256_file(args.parity_report) == config["parity_report_sha256"],
        "parity report identity drifted",
    )
    require(
        parity["parity_pass"] is True
        and parity["report_sha256_excludes_self"] == config["parity_report_self_sha256"],
        "DirectML route lacks a bound parity pass",
    )
    source_payload = args.source_config.read_bytes()
    require(
        sha256_bytes(source_payload) == config["source_config_sha256"],
        "source benchmark config drifted",
    )
    source_config = json.loads(source_payload)
    require(sha256_file(args.policies) == config["policy_payload_sha256"], "policy payload drifted")
    require(
        ort.__version__ == "1.24.4" and "DmlExecutionProvider" in ort.get_available_providers(),
        "pinned DirectML runtime is unavailable",
    )

    intraop = int(config["route"]["policy_intraop_threads"])
    interop = int(config["route"]["policy_interop_threads"])
    torch.set_num_threads(intraop)
    torch.set_num_interop_threads(interop)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    runs = verify_sources(source_config)
    material = load_policy_material(args.policies, args.variant, args.seed)
    checkpoint = torch.load(
        __import__("io").BytesIO(material["checkpoint"]), map_location="cpu", weights_only=True
    )
    policy = LiViFuserPolicy(variant=args.variant)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()

    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(args.onnx),
        sess_options=options,
        providers=[
            ("DmlExecutionProvider", {"device_id": str(config["route"]["device_id"])}),
            "CPUExecutionProvider",
        ],
    )
    require(session.get_providers()[0] == "DmlExecutionProvider", "DirectML is not primary")
    input_name = session.get_inputs()[0].name
    histories = [deque(maxlen=8), deque(maxlen=8)]
    previous_actions = [np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)]
    for run_index, run_object in enumerate(runs):
        for row in range(7):
            context, _pooled, _parts = dml_context(session, input_name, run_object, row)
            histories[run_index].append(context)

    timings: dict[str, list[float]] = {name: [] for name in STAGES}
    checksums = []
    for iteration in range(args.warmup + args.iterations):
        run_index = iteration % len(runs)
        source_step = iteration // len(runs)
        run_object = runs[run_index]
        row = 7 + source_step % (run_object.count - 7)
        total_start = time.perf_counter_ns()
        context, pooled, parts = dml_context(session, input_name, run_object, row)
        histories[run_index].append(context)

        policy_start = time.perf_counter_ns()
        history = list(histories[run_index])
        arrays = [np.stack([item[index] for item in history]) for index in range(6)]
        inputs = {
            "visual_tokens": torch.from_numpy(arrays[0]).unsqueeze(0),
            "lidar_features": torch.from_numpy(arrays[1]).unsqueeze(0),
            "visual_mask": torch.from_numpy(arrays[2]).unsqueeze(0),
            "in_fov": torch.from_numpy(arrays[3]).unsqueeze(0),
            "goal": torch.from_numpy(arrays[4]).unsqueeze(0),
            "robot_state": torch.from_numpy(arrays[5]).unsqueeze(0),
        }
        with torch.inference_mode():
            outputs = policy(**inputs)
        mean = outputs["mean"].numpy()
        log_variance = outputs["log_variance"].numpy()
        policy_end = time.perf_counter_ns()

        aleatoric = float(np.mean(np.exp(np.clip(log_variance, -5.0, 2.0))))
        mahalanobis = float(
            mahalanobis_distances(
                pooled[None], material["mahalanobis_mean"], material["mahalanobis_precision"]
            )[0]
        )
        z_a = float(right_continuous_cdf(material["aleatoric_cdf"], np.asarray([aleatoric]))[0])
        z_m = float(right_continuous_cdf(material["mahalanobis_cdf"], np.asarray([mahalanobis]))[0])
        combined = max(z_a, z_m)
        clipped = np.clip(np.asarray(mean[0, 0], dtype=np.float64), [-0.1, -0.5], [0.1, 0.5])
        limits = np.asarray([0.05, 0.1], dtype=np.float64)
        command = previous_actions[run_index] + np.clip(
            clipped - previous_actions[run_index], -limits, limits
        )
        if combined > float(material["thresholds"]["combined"]):
            command[:] = 0.0
        previous_actions[run_index] = command
        uncertainty_end = time.perf_counter_ns()

        if iteration >= args.warmup:
            timings["rgb_preprocess"].append(parts[0])
            timings["splus_directml_forward_and_copy"].append(parts[1])
            timings["lidar_tokenize"].append(parts[2])
            timings["policy_stack_and_forward"].append((policy_end - policy_start) / 1e6)
            timings["uncertainty_and_supervisor"].append((uncertainty_end - policy_end) / 1e6)
            timings["complete_path"].append((uncertainty_end - total_start) / 1e6)
            payload = (
                np.concatenate(
                    (
                        context[0].reshape(-1),
                        pooled,
                        mean.reshape(-1),
                        log_variance.reshape(-1),
                        command,
                    )
                )
                .astype(np.float32)
                .tobytes()
            )
            checksums.append(sha256_bytes(payload))

    summaries = {name: percentile_summary(values) for name, values in timings.items()}
    deadline = float(config["benchmark"]["deadline_ms"])
    misses = sum(value >= deadline for value in timings["complete_path"])
    report = {
        "schema_version": "1.0.0",
        "status": "complete",
        "route": "parity_qualified_splus_directml_rx5500m_plus_exact_pytorch_cpu_policy",
        "variant": args.variant,
        "seed": args.seed,
        "timed_iterations": args.iterations,
        "warmup_iterations": args.warmup,
        "latency": summaries,
        "raw_latency_ms": timings,
        "deadline_ms": deadline,
        "missed_deadline_count": misses,
        "deadline_pass": misses == 0 and summaries["complete_path"]["p99_ms"] < deadline,
        "output_checksum_sha256": sha256_bytes("\n".join(checksums).encode()),
        "runtime": {
            "providers": session.get_providers(),
            "onnxruntime": ort.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "device_id": int(config["route"]["device_id"]),
            "device_identity": config["route"]["device_identity"],
            "policy_intraop_threads": intraop,
            "policy_interop_threads": interop,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "identities": {
            "config_sha256": sha256_bytes(config_payload),
            "source_config_sha256": sha256_bytes(source_payload),
            "onnx_sha256": sha256_file(args.onnx),
            "parity_report_sha256": sha256_file(args.parity_report),
            "policy_payload_sha256": sha256_file(args.policies),
            "code_sha256": sha256_file(Path(__file__)),
        },
        "forbidden_inputs_used": {"heldout": False, "confirmatory": False},
    }
    report["report_sha256_excludes_self"] = sha256_bytes(json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/dinov3_splus_directml_benchmark_v1.json",
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=ROOT / "config/dinov3_splus_cpu_benchmark_v1.json",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=ROOT / "artifacts/runtime/models/dinov3_vits16plus_features_opset17.onnx",
    )
    parser.add_argument(
        "--parity-report",
        type=Path,
        default=ROOT / "artifacts/runtime/directml_parity_v1.json",
    )
    parser.add_argument(
        "--policies",
        type=Path,
        default=ROOT / "artifacts/livifuser_sim_closed_loop_policy_payload_v1_bundle.zip",
    )
    parser.add_argument(
        "--variant", required=True, choices=("full", "lidar_only", "concat", "rgb_only")
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite DirectML benchmark: {args.output}")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json_bytes(report))
    print(
        json_bytes(
            {
                "output": str(args.output.resolve()),
                "variant": report["variant"],
                "complete_path": report["latency"]["complete_path"],
                "missed_deadline_count": report["missed_deadline_count"],
                "deadline_pass": report["deadline_pass"],
                "report_sha256_excludes_self": report["report_sha256_excludes_self"],
            }
        ).decode(),
        end="",
    )


if __name__ == "__main__":
    main()
