#!/usr/bin/env python3
"""Prospective CPU/CUDA parity and complete-path RTX 3090 runtime gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

from livifuser_nav.backbone_handoff import json_bytes, sha256_bytes, sha256_file  # noqa: E402
from livifuser_nav.learning_data import preprocess_rgb, tokenize_lidar  # noqa: E402
from livifuser_nav.live_runtime import (  # noqa: E402
    ExactPolicyRuntime,
    LiveObservation,
    configure_deterministic_torch,
    extract_verified_backbone,
    load_policy_materials,
)
from livifuser_nav.model import LiViFuserPolicy  # noqa: E402
from livifuser_nav.policy_payload import SEEDS  # noqa: E402
from livifuser_nav.simulation_supervision import (  # noqa: E402
    PrivilegedState,
    ProposalInput,
    SimulationSupervisor,
)

VARIANTS = ("full", "lidar_only", "concat", "rgb_only")
IDENTITIES = tuple((variant, seed) for variant in VARIANTS for seed in SEEDS)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_self_hash(value: dict[str, Any], field: str) -> str:
    copied = copy.deepcopy(value)
    copied.pop(field, None)
    return sha256_bytes(json.dumps(copied, sort_keys=True, separators=(",", ":")).encode())


def cosine_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    require(denominator > 0.0, "cosine denominator is zero")
    return float(1.0 - np.dot(left, right) / denominator)


def load_parity_inputs(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    require(
        manifest["status"] == "FROZEN_EXCLUDED_DEVELOPMENT_PARITY_INPUTS",
        "parity input status drifted",
    )
    require(
        manifest["heldout_or_confirmatory_inputs_present"] is False,
        "forbidden parity input scope",
    )
    require(
        manifest["manifest_sha256_excludes_self"]
        == canonical_self_hash(manifest, "manifest_sha256_excludes_self"),
        "parity manifest self-hash failed",
    )
    values = {
        "manifest": manifest,
        "rgb": np.load(root / "rgb.npy", allow_pickle=False),
        "scan_ranges": np.load(root / "scan_ranges.npy", allow_pickle=False),
        "scan_beam_count": np.load(root / "scan_beam_count.npy", allow_pickle=False),
        "scan_angle_increment_rad": np.load(
            root / "scan_angle_increment_rad.npy", allow_pickle=False
        ),
        "goal": np.load(root / "goal.npy", allow_pickle=False),
        "robot_state": np.load(root / "robot_state.npy", allow_pickle=False),
    }
    require(values["rgb"].shape == (32, 240, 320, 3), "parity RGB shape drifted")
    require(values["goal"].shape == (32, 3), "parity goal shape drifted")
    require(values["robot_state"].shape == (32, 2), "parity state shape drifted")
    return values


def extract_features(model: Any, rgb: np.ndarray, torch: Any, device: str):
    pixels = torch.from_numpy(preprocess_rgb(rgb)).unsqueeze(0).to(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    with torch.inference_mode():
        outputs = model(pixel_values=pixels, return_dict=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    hidden, pooled = outputs[0], outputs[1]
    require(tuple(hidden.shape) == (1, 201, 384), "S+/16 hidden shape drifted")
    require(tuple(pooled.shape) == (1, 384), "S+/16 pooled shape drifted")
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
    return np.ascontiguousarray(patches), np.ascontiguousarray(pooled_array)


def build_policy(material: Any, torch: Any, device: str) -> Any:
    checkpoint = torch.load(io.BytesIO(material.checkpoint), map_location="cpu", weights_only=True)
    policy = LiViFuserPolicy(variant=material.variant)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return policy.eval().to(device=device, dtype=torch.float32)


def policy_inputs(
    patch: np.ndarray,
    frame: int,
    parity: dict[str, Any],
    contract: dict[str, Any],
    torch: Any,
    device: str,
) -> dict[str, Any]:
    tokens = tokenize_lidar(
        parity["scan_ranges"][frame],
        int(parity["scan_beam_count"][frame]),
        float(parity["scan_angle_increment_rad"][frame]),
        contract,
        sectors=80,
        range_clip_m=10.0,
        visual_radius=1,
    )

    def repeated(value: np.ndarray) -> Any:
        values = np.repeat(np.asarray(value)[None], 8, axis=0)
        return torch.from_numpy(np.ascontiguousarray(values)).unsqueeze(0).to(device)

    return {
        "visual_tokens": repeated(patch),
        "lidar_features": repeated(tokens.features),
        "visual_mask": repeated(tokens.visual_mask),
        "in_fov": repeated(tokens.in_fov),
        "goal": repeated(parity["goal"][frame].astype(np.float32)),
        "robot_state": repeated(parity["robot_state"][frame].astype(np.float32)),
    }


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum_ms": float(np.min(array)),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "maximum_ms": float(np.max(array)),
        "mean_ms": float(np.mean(array)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import PIL
    import torch
    import transformers

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite CUDA route report: {args.output}")
    config_payload = args.config.read_bytes()
    config = json.loads(config_payload)
    require(
        config["status"] == "PROPOSED_NOT_APPROVED_DEVELOPMENT_ONLY",
        "RunPod route contract status drifted",
    )
    require(
        sha256_file(args.backbone) == config["input_identities"]["backbone_bundle"]["sha256"],
        "backbone bundle identity drifted",
    )
    require(
        sha256_file(args.policies) == config["input_identities"]["policy_payload"]["sha256"],
        "policy payload identity drifted",
    )
    configure_deterministic_torch(torch, "cuda:0")
    require(torch.cuda.device_count() == 1, "frozen candidate requires exactly one visible GPU")
    require("RTX 3090" in torch.cuda.get_device_name(0), "GPU is not the frozen RTX 3090 candidate")
    parity = load_parity_inputs(args.parity_inputs)
    contract = json.loads(args.sensor_contract.read_text(encoding="utf-8"))
    materials = load_policy_materials(args.policies, IDENTITIES)
    snapshot = extract_verified_backbone(args.backbone, args.extract_root)

    cpu_backbone = transformers.AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).eval()
    cuda_backbone = transformers.AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).eval().to("cuda:0")

    references: list[tuple[np.ndarray, np.ndarray]] = []
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    frame_records: list[dict[str, Any]] = []
    maxima = {
        "patch_max_abs_error": 0.0,
        "pooled_max_abs_error": 0.0,
        "patch_one_minus_cosine": 0.0,
        "pooled_one_minus_cosine": 0.0,
        "downstream_first_action_max_abs_error": 0.0,
    }
    for frame in range(32):
        reference = extract_features(cpu_backbone, parity["rgb"][frame], torch, "cpu")
        candidate = extract_features(cuda_backbone, parity["rgb"][frame], torch, "cuda:0")
        references.append(reference)
        candidates.append(candidate)
        metrics = {
            "patch_max_abs_error": float(np.max(np.abs(reference[0] - candidate[0]))),
            "pooled_max_abs_error": float(np.max(np.abs(reference[1] - candidate[1]))),
            "patch_one_minus_cosine": cosine_error(reference[0], candidate[0]),
            "pooled_one_minus_cosine": cosine_error(reference[1], candidate[1]),
            "downstream_first_action_max_abs_error": 0.0,
        }
        for name, value in metrics.items():
            maxima[name] = max(maxima[name], value)
        frame_records.append(
            {
                **parity["manifest"]["rows"][frame],
                "rgb_sha256": hashlib.sha256(parity["rgb"][frame].tobytes()).hexdigest().upper(),
                "metrics": metrics,
                "action_errors": [],
            }
        )

    for identity in IDENTITIES:
        material = materials[identity]
        cpu_policy = build_policy(material, torch, "cpu")
        cuda_policy = build_policy(material, torch, "cuda:0")
        for frame in range(32):
            cpu_inputs = policy_inputs(
                references[frame][0], frame, parity, contract, torch, "cpu"
            )
            cuda_inputs = policy_inputs(
                candidates[frame][0], frame, parity, contract, torch, "cuda:0"
            )
            with torch.inference_mode():
                cpu_action = cpu_policy(**cpu_inputs)["mean"][0, 0].cpu().numpy()
                cuda_action = cuda_policy(**cuda_inputs)["mean"][0, 0].cpu().numpy()
            error = float(np.max(np.abs(cpu_action - cuda_action)))
            maxima["downstream_first_action_max_abs_error"] = max(
                maxima["downstream_first_action_max_abs_error"], error
            )
            frame_records[frame]["metrics"][
                "downstream_first_action_max_abs_error"
            ] = max(
                frame_records[frame]["metrics"][
                    "downstream_first_action_max_abs_error"
                ],
                error,
            )
            frame_records[frame]["action_errors"].append(
                {"variant": identity[0], "seed": identity[1], "max_abs_error": error}
            )
        del cpu_policy, cuda_policy
        torch.cuda.empty_cache()

    acceptance = config["prospective_acceptance"]
    parity_checks = {
        "patch_max_abs_error": maxima["patch_max_abs_error"]
        <= acceptance["patch_max_abs_error_lte"],
        "pooled_max_abs_error": maxima["pooled_max_abs_error"]
        <= acceptance["pooled_max_abs_error_lte"],
        "patch_one_minus_cosine": maxima["patch_one_minus_cosine"]
        <= acceptance["patch_one_minus_cosine_lte"],
        "pooled_one_minus_cosine": maxima["pooled_one_minus_cosine"]
        <= acceptance["pooled_one_minus_cosine_lte"],
        "downstream_first_action_max_abs_error": maxima[
            "downstream_first_action_max_abs_error"
        ]
        <= acceptance["downstream_first_action_max_abs_error_lte"],
    }
    parity_pass = all(parity_checks.values())
    timing_reports: list[dict[str, Any]] = []
    if parity_pass:
        del cpu_backbone
        for variant in VARIANTS:
            material = materials[(variant, 20260805)]
            policy = build_policy(material, torch, "cuda:0")
            runtime = ExactPolicyRuntime(
                backbone=cuda_backbone,
                policy=policy,
                material=material,
                sensor_contract=contract,
                torch=torch,
                device="cuda:0",
            )
            supervisor = SimulationSupervisor(scientific_deadline_sec=120.0)
            total_ms: list[float] = []
            stage_values: dict[str, list[float]] = {}
            for iteration in range(args.warmup + args.iterations):
                frame = iteration % 32
                observation = LiveObservation(
                    rgb=parity["rgb"][frame],
                    scan_ranges=parity["scan_ranges"][frame],
                    scan_beam_count=int(parity["scan_beam_count"][frame]),
                    scan_angle_increment_rad=float(
                        parity["scan_angle_increment_rad"][frame]
                    ),
                    goal=parity["goal"][frame].astype(np.float32),
                    robot_state=parity["robot_state"][frame].astype(np.float32),
                )
                start_ns = time.perf_counter_ns()
                runtime_decision = runtime.accept(observation)
                supervisor.step(
                    ProposalInput(
                        stamp_ns=1_000_000_000 + iteration * 100_000_000,
                        linear_x=float(runtime_decision.proposed_action[0]),
                        angular_z=float(runtime_decision.proposed_action[1]),
                        valid=True,
                        inference_ready=runtime_decision.ready,
                        status=runtime_decision.status,
                        combined_intervention=runtime_decision.combined_intervention,
                    ),
                    PrivilegedState(True, False, 1.0, 1.0),
                )
                end_ns = time.perf_counter_ns()
                if iteration >= args.warmup:
                    total_ms.append((end_ns - start_ns) / 1e6)
                    for name, value in runtime_decision.stage_ms.items():
                        stage_values.setdefault(name, []).append(float(value))
            missed = sum(value >= 100.0 for value in total_ms)
            total_summary = summarize(total_ms)
            timing_reports.append(
                {
                    "variant": variant,
                    "seed": 20260805,
                    "warmup_iterations": args.warmup,
                    "timed_iterations": args.iterations,
                    "complete_path_including_supervisor": total_summary,
                    "stages": {
                        name: summarize(values) for name, values in sorted(stage_values.items())
                    },
                    "raw_complete_path_ms": total_ms,
                    "missed_deadline_count": missed,
                    "pass": total_summary["p99_ms"] < 100.0 and missed == 0,
                }
            )
            del runtime, policy
            torch.cuda.empty_cache()
    timing_pass = parity_pass and len(timing_reports) == 4 and all(
        row["pass"] for row in timing_reports
    )
    properties = torch.cuda.get_device_properties(0)
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "CUDA_ROUTE_ACCEPTED" if timing_pass else "CUDA_ROUTE_REJECTED",
        "scope": "excluded_development_route_selection_only",
        "confirmatory_inference_performed": False,
        "heldout_inference_performed": False,
        "inputs": {
            "config_sha256": sha256_bytes(config_payload),
            "backbone_bundle_sha256": sha256_file(args.backbone),
            "policy_payload_sha256": sha256_file(args.policies),
            "parity_manifest_self_sha256": parity["manifest"][
                "manifest_sha256_excludes_self"
            ],
        },
        "device": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "software": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "execution": {
            "float32": True,
            "autocast": False,
            "tf32": False,
            "deterministic_algorithms": True,
            "cuda_synchronized": True,
        },
        "parity": {
            "frame_count": 32,
            "policy_identity_count": 12,
            "maxima": maxima,
            "checks": parity_checks,
            "pass": parity_pass,
            "frames": frame_records,
        },
        "timing": {"reports": timing_reports, "pass": timing_pass},
        "route_selected": timing_pass,
    }
    report["report_sha256_excludes_self"] = canonical_self_hash(
        report, "report_sha256_excludes_self"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/runpod_rtx3090_runtime_v1.proposed.json"
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
        "--parity-inputs",
        type=Path,
        default=ROOT / "artifacts/runtime/parity_inputs_v1",
    )
    parser.add_argument(
        "--sensor-contract",
        type=Path,
        default=ROOT / "config/simulation_live_sensor_contract_v1.json",
    )
    parser.add_argument(
        "--extract-root", type=Path, default=Path("/workspace/livifuser/runtime/backbone")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/livifuser/evidence/cuda_route_benchmark_v1.json"),
    )
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    require(args.warmup >= 8, "warmup must populate K=8")
    require(args.iterations >= 200, "frozen timing gate requires at least 200 iterations")
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "parity_pass": report["parity"]["pass"],
                "timing_pass": report["timing"]["pass"],
                "route_selected": report["route_selected"],
                "report": str(args.output.resolve()),
                "report_sha256_excludes_self": report[
                    "report_sha256_excludes_self"
                ],
            },
            indent=2,
        )
    )
    return 0 if report["route_selected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
