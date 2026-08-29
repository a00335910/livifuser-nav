#!/usr/bin/env python3
"""Benchmark the exact float32 PyTorch CPU live-context path."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import tempfile
import time
import zipfile
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from livifuser_nav.backbone_handoff import (  # noqa: E402
    BUNDLE_ROOT as BACKBONE_ROOT,
)
from livifuser_nav.backbone_handoff import (
    EXPECTED_MODEL_FILES,
    json_bytes,
    sha256_bytes,
    sha256_file,
    verify_bundle,
)
from livifuser_nav.evaluation import mahalanobis_distances  # noqa: E402
from livifuser_nav.heldout_evaluation import right_continuous_cdf  # noqa: E402
from livifuser_nav.learning_data import (  # noqa: E402
    ExportRun,
    preprocess_rgb,
    tokenize_lidar,
)
from livifuser_nav.model import LiViFuserPolicy  # noqa: E402
from livifuser_nav.policy_payload import (  # noqa: E402
    BUNDLE_ROOT as POLICY_ROOT,
)
from livifuser_nav.policy_payload import (
    verify_policy_payload,
)

STAGE_NAMES = (
    "rgb_preprocess",
    "splus_forward_and_pool",
    "lidar_tokenize",
    "policy_stack_and_forward",
    "uncertainty_and_supervisor",
    "complete_path",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def percentile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    require(
        array.ndim == 1 and array.size > 0 and np.all(np.isfinite(array)),
        "latency vector is invalid",
    )
    return {
        "count": int(array.size),
        "minimum_ms": float(np.min(array)),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "maximum_ms": float(np.max(array)),
        "mean_ms": float(np.mean(array)),
    }


def extract_backbone(bundle: Path, destination: Path) -> Path:
    verify_bundle(bundle)
    snapshot = destination / "snapshot"
    snapshot.mkdir(parents=True)
    with zipfile.ZipFile(bundle) as archive:
        for name in sorted(EXPECTED_MODEL_FILES):
            payload = archive.read(f"{BACKBONE_ROOT}/{name}")
            target = snapshot / name
            target.write_bytes(payload)
            expected = EXPECTED_MODEL_FILES[name]
            require(
                target.stat().st_size == expected["size_bytes"], f"extracted size drift: {name}"
            )
            require(sha256_file(target) == expected["sha256"], f"extracted hash drift: {name}")
    return snapshot


def load_policy_material(bundle: Path, variant: str, seed: int) -> dict[str, Any]:
    verify_policy_payload(bundle)
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read(f"{POLICY_ROOT}/POLICY_PAYLOAD_MANIFEST.json"))
        matches = [
            row
            for row in manifest["records"]
            if row["variant"] == variant and int(row["seed"]) == seed
        ]
        require(len(matches) == 1, "policy identity is not unique")
        record = matches[0]
        checkpoint = archive.read(f"{POLICY_ROOT}/{record['checkpoint']['name']}")
        score_payload = archive.read(f"{POLICY_ROOT}/{record['score']['name']}")
        mean_payload = archive.read(f"{POLICY_ROOT}/mahalanobis/mean.npy")
        precision_payload = archive.read(f"{POLICY_ROOT}/mahalanobis/precision.npy")
    with np.load(io.BytesIO(score_payload), allow_pickle=False) as score:
        aleatoric_cdf = np.asarray(score["aleatoric_cdf_sorted"], dtype=np.float64).copy()
        mahalanobis_cdf = np.asarray(score["mahalanobis_cdf_sorted"], dtype=np.float64).copy()
    return {
        "checkpoint": checkpoint,
        "aleatoric_cdf": aleatoric_cdf,
        "mahalanobis_cdf": mahalanobis_cdf,
        "mahalanobis_mean": np.load(io.BytesIO(mean_payload), allow_pickle=False),
        "mahalanobis_precision": np.load(io.BytesIO(precision_payload), allow_pickle=False),
        "thresholds": record["thresholds"],
    }


def verify_sources(config: dict[str, Any]) -> list[ExportRun]:
    runs: list[ExportRun] = []
    for source in config["sources"]:
        root = ROOT / source["root"]
        require(
            sha256_file(root / "manifest.json") == source["manifest_sha256"],
            f"source manifest drifted: {root}",
        )
        run = ExportRun(root)
        hashes = run.verify_output_hashes()
        require(hashes["rgb_320x240_rgb8.npy"] == source["rgb_sha256"], "source RGB drifted")
        require(hashes["scan_ranges.npy"] == source["scan_sha256"], "source scan drifted")
        require(hashes["vectors.npz"] == source["vectors_sha256"], "source vectors drifted")
        require(run.count == int(source["rows"]), "source row count drifted")
        runs.append(run)
    require(len(runs) == 2 and all(run.count >= 60 for run in runs), "benchmark sources drifted")
    return runs


def feature_context(
    backbone: Any,
    run: ExportRun,
    row: int,
    torch: Any,
) -> tuple[np.ndarray, np.ndarray, Any, list[float]]:
    start = time.perf_counter_ns()
    rgb = preprocess_rgb(np.asarray(run.rgb[row]))
    t1 = time.perf_counter_ns()
    pixel_values = torch.from_numpy(rgb).unsqueeze(0)
    with torch.inference_mode():
        outputs = backbone(pixel_values=pixel_values, return_dict=False)
    hidden, pooled = outputs[0], outputs[1]
    require(tuple(hidden.shape) == (1, 201, 384), "S+ hidden shape drifted")
    require(tuple(pooled.shape) == (1, 384), "S+ pooler shape drifted")
    spatial = hidden[:, 5:, :].reshape(1, 14, 14, 384)
    patches = (
        spatial.reshape(1, 7, 2, 7, 2, 384)
        .mean(dim=(2, 4))
        .reshape(49, 384)
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    pooled_array = pooled[0].cpu().numpy().astype(np.float32, copy=False)
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
        np.all(np.isfinite(patches)) and np.all(np.isfinite(pooled_array)), "non-finite S+ output"
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
        pooled_array,
        tokens,
        [
            (t1 - start) / 1e6,
            (t2 - t1) / 1e6,
            (t3 - t2) / 1e6,
        ],
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import PIL
    import torch
    import transformers

    config_payload = args.config.read_bytes()
    config = json.loads(config_payload)
    require(
        config["status"] == "FROZEN_BEFORE_FIRST_OFFICIAL_BACKBONE_MODEL_LOAD",
        "benchmark config is not frozen",
    )
    require(
        sha256_file(args.backbone) == config["backbone_bundle_sha256"],
        "backbone bundle identity drifted from benchmark freeze",
    )
    require(
        sha256_file(args.policies) == config["policy_payload_sha256"],
        "policy payload identity drifted from benchmark freeze",
    )
    require(args.variant in config["final_route_evidence"]["variants"], "variant is not frozen")
    require(args.seed == int(config["final_route_evidence"]["seed"]), "seed is not frozen")
    require(
        args.threads in config["thread_screen"]["intraop_thread_candidates"],
        "thread count is not frozen",
    )
    require(args.iterations > 0 and args.warmup >= 0, "iteration counts must be non-negative")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(int(config["common"]["interop_threads"]))
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    runs = verify_sources(config)
    material = load_policy_material(args.policies, args.variant, args.seed)
    checkpoint = torch.load(
        io.BytesIO(material["checkpoint"]), map_location="cpu", weights_only=True
    )
    policy = LiViFuserPolicy(variant=args.variant)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()

    temp_parent = Path(os.environ.get("LIVIFUSER_BENCHMARK_TEMP", tempfile.gettempdir()))
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="livifuser_splus_cpu_", dir=temp_parent) as raw:
        snapshot = extract_backbone(args.backbone, Path(raw))
        backbone = transformers.AutoModel.from_pretrained(
            snapshot,
            local_files_only=True,
            attn_implementation="eager",
            torch_dtype=torch.float32,
        )
        backbone.eval()
        require(next(backbone.parameters()).device.type == "cpu", "backbone is not on CPU")
        require(next(backbone.parameters()).dtype == torch.float32, "backbone is not float32")

        histories = [deque(maxlen=8), deque(maxlen=8)]
        previous_actions = [np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)]
        for run_index, run in enumerate(runs):
            for row in range(7):
                context, _pooled, _tokens, _timings = feature_context(backbone, run, row, torch)
                histories[run_index].append(context)

        timings: dict[str, list[float]] = {name: [] for name in STAGE_NAMES}
        output_checksums: list[str] = []
        total_loops = args.warmup + args.iterations
        for iteration in range(total_loops):
            run_index = iteration % len(runs)
            source_step = iteration // len(runs)
            run = runs[run_index]
            row = 7 + (source_step % (run.count - 7))
            total_start = time.perf_counter_ns()
            context, pooled, _tokens, first_timings = feature_context(backbone, run, row, torch)
            histories[run_index].append(context)
            require(len(histories[run_index]) == 8, "context history did not reach K=8")

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
            mean = outputs["mean"].cpu().numpy()
            log_variance = outputs["log_variance"].cpu().numpy()
            policy_end = time.perf_counter_ns()

            uncertainty_start = policy_end
            aleatoric = float(np.mean(np.exp(np.clip(log_variance, -5.0, 2.0))))
            mahalanobis = float(
                mahalanobis_distances(
                    pooled[None], material["mahalanobis_mean"], material["mahalanobis_precision"]
                )[0]
            )
            z_a = float(right_continuous_cdf(material["aleatoric_cdf"], np.asarray([aleatoric]))[0])
            z_m = float(
                right_continuous_cdf(material["mahalanobis_cdf"], np.asarray([mahalanobis]))[0]
            )
            combined = max(z_a, z_m)
            proposed = np.asarray(mean[0, 0], dtype=np.float64)
            clipped = np.clip(proposed, [-0.1, -0.5], [0.1, 0.5])
            delta_limit = np.asarray([0.05, 0.1], dtype=np.float64)
            command = previous_actions[run_index] + np.clip(
                clipped - previous_actions[run_index], -delta_limit, delta_limit
            )
            if combined > float(material["thresholds"]["combined"]):
                command[:] = 0.0
            previous_actions[run_index] = command
            require(np.all(np.isfinite(command)), "supervised command is non-finite")
            uncertainty_end = time.perf_counter_ns()
            total_end = uncertainty_end

            if iteration >= args.warmup:
                timings["rgb_preprocess"].append(first_timings[0])
                timings["splus_forward_and_pool"].append(first_timings[1])
                timings["lidar_tokenize"].append(first_timings[2])
                timings["policy_stack_and_forward"].append((policy_end - policy_start) / 1e6)
                timings["uncertainty_and_supervisor"].append(
                    (uncertainty_end - uncertainty_start) / 1e6
                )
                timings["complete_path"].append((total_end - total_start) / 1e6)
                checksum_payload = (
                    np.concatenate(
                        (
                            history[-1][0].reshape(-1),
                            pooled,
                            mean.reshape(-1),
                            log_variance.reshape(-1),
                            command,
                        )
                    )
                    .astype(np.float32)
                    .tobytes()
                )
                output_checksums.append(sha256_bytes(checksum_payload))

    summaries = {name: percentile_summary(values) for name, values in timings.items()}
    deadline = float(config["final_route_evidence"]["deadline_ms"])
    missed = sum(value >= deadline for value in timings["complete_path"])
    report = {
        "schema_version": "1.0.0",
        "status": "complete",
        "route": "exact_pytorch_cpu_float32_eager",
        "variant": args.variant,
        "seed": args.seed,
        "threads": {"intraop": args.threads, "interop": int(config["common"]["interop_threads"])},
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "deadline_ms": deadline,
        "missed_deadline_count": missed,
        "deadline_pass": missed == 0 and summaries["complete_path"]["p99_ms"] < deadline,
        "latency": summaries,
        "raw_latency_ms": timings,
        "output_checksum_sha256": sha256_bytes("\n".join(output_checksums).encode()),
        "identities": {
            "benchmark_config_sha256": sha256_bytes(config_payload),
            "backbone_bundle_sha256": sha256_file(args.backbone),
            "policy_payload_sha256": sha256_file(args.policies),
            "benchmark_code_sha256": sha256_file(Path(__file__)),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "logical_cpu_count": os.cpu_count(),
            "device": "cpu",
            "thermal_and_power_telemetry": "not_available_on_host",
        },
        "forbidden_inputs_used": {
            "heldout": False,
            "confirmatory": False,
            "cached_dino_features": False,
        },
    }
    report["report_sha256_excludes_self"] = sha256_bytes(json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/dinov3_splus_cpu_benchmark_v1.json"
    )
    parser.add_argument(
        "--backbone",
        type=Path,
        default=ROOT / "artifacts/livifuser_dinov3_vits16plus_backbone_c93d816_bundle.zip",
    )
    parser.add_argument(
        "--policies",
        type=Path,
        default=ROOT / "artifacts/livifuser_sim_closed_loop_policy_payload_v1_bundle.zip",
    )
    parser.add_argument(
        "--variant", choices=("full", "lidar_only", "concat", "rgb_only"), required=True
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark report: {args.output}")
    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json_bytes(report))
    print(
        json_bytes(
            {
                "output": str(args.output.resolve()),
                "variant": report["variant"],
                "threads": report["threads"],
                "timed_iterations": report["timed_iterations"],
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
