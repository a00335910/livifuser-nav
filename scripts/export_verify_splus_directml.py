#!/usr/bin/env python3
"""Export S+/16 features to ONNX and enforce frozen DirectML parity."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from livifuser_nav.backbone_handoff import (
    BUNDLE_ROOT as BACKBONE_ROOT,
)
from livifuser_nav.backbone_handoff import (
    EXPECTED_MODEL_FILES,
    json_bytes,
    sha256_bytes,
    sha256_file,
    verify_bundle,
)
from livifuser_nav.learning_data import ExportRun, preprocess_rgb, tokenize_lidar
from livifuser_nav.model import LiViFuserPolicy
from livifuser_nav.policy_payload import BUNDLE_ROOT as POLICY_ROOT
from livifuser_nav.policy_payload import verify_policy_payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def cosine_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    require(denominator > 0.0, "cosine denominator is zero")
    return float(1.0 - np.dot(left, right) / denominator)


def extract_backbone(bundle: Path, destination: Path) -> Path:
    verify_bundle(bundle)
    snapshot = destination / "snapshot"
    snapshot.mkdir(parents=True)
    with zipfile.ZipFile(bundle) as archive:
        for name in sorted(EXPECTED_MODEL_FILES):
            payload = archive.read(f"{BACKBONE_ROOT}/{name}")
            target = snapshot / name
            target.write_bytes(payload)
            require(
                sha256_file(target) == EXPECTED_MODEL_FILES[name]["sha256"],
                f"extracted backbone drifted: {name}",
            )
    return snapshot


def load_policies(bundle: Path, torch: Any) -> list[tuple[str, int, LiViFuserPolicy]]:
    verify_policy_payload(bundle)
    policies = []
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read(f"{POLICY_ROOT}/POLICY_PAYLOAD_MANIFEST.json"))
        for row in manifest["records"]:
            variant = str(row["variant"])
            seed = int(row["seed"])
            checkpoint = torch.load(
                io.BytesIO(archive.read(f"{POLICY_ROOT}/{row['checkpoint']['name']}")),
                map_location="cpu",
                weights_only=True,
            )
            policy = LiViFuserPolicy(variant=variant)
            policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
            policy.eval()
            policies.append((variant, seed, policy))
    require(len(policies) == 12, "policy identity count drifted")
    return policies


def source_inputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for source in config["parity_sources"]:
        root = ROOT / source["root"]
        require(
            sha256_file(root / "manifest.json") == source["manifest_sha256"],
            f"parity manifest drifted: {root}",
        )
        run = ExportRun(root)
        hashes = run.verify_output_hashes()
        require(hashes["rgb_320x240_rgb8.npy"] == source["rgb_sha256"], "RGB hash drifted")
        require(hashes["scan_ranges.npy"] == source["scan_sha256"], "scan hash drifted")
        require(hashes["vectors.npz"] == source["vectors_sha256"], "vectors hash drifted")
        for row in source["rows"]:
            require(0 <= int(row) < run.count, "parity row is out of range")
            output.append({"source": source, "run": run, "row": int(row)})
    require(len(output) == int(config["parity_frame_count"]) == 32, "parity frame count drifted")
    return output


def policy_input(patch_tokens: np.ndarray, run: ExportRun, row: int, torch: Any) -> dict[str, Any]:
    tokens = tokenize_lidar(
        run.scan_ranges[row],
        int(run.vectors["scan_beam_count"][row]),
        float(run.vectors["scan_angle_increment_rad"][row]),
        run.manifest,
        sectors=80,
        range_clip_m=10.0,
        visual_radius=1,
    )

    def repeated(value: np.ndarray) -> Any:
        return torch.from_numpy(np.repeat(value[None], 8, axis=0)).unsqueeze(0)

    return {
        "visual_tokens": repeated(np.asarray(patch_tokens, dtype=np.float32)),
        "lidar_features": repeated(tokens.features),
        "visual_mask": repeated(tokens.visual_mask),
        "in_fov": repeated(tokens.in_fov),
        "goal": repeated(np.asarray(run.vectors["goal"][row], dtype=np.float32)),
        "robot_state": repeated(np.asarray(run.vectors["robot_state"][row], dtype=np.float32)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort
    import PIL
    import torch
    import transformers

    config_payload = args.config.read_bytes()
    config = json.loads(config_payload)
    require(
        config["status"] == "FROZEN_BEFORE_FIRST_SPLUS_ONNX_EXPORT",
        "DirectML parity config is not frozen",
    )
    require(
        sha256_file(args.backbone) == config["backbone_bundle_sha256"], "backbone identity drifted"
    )
    require(
        sha256_file(args.policies) == config["policy_payload_sha256"],
        "policy payload identity drifted",
    )
    require(
        "DmlExecutionProvider" in ort.get_available_providers(),
        "DirectML execution provider is unavailable",
    )
    require(
        ort.__version__ == config["runtime"]["onnxruntime_directml"],
        "DirectML runtime version drifted",
    )
    require(onnx.__version__ == config["runtime"]["onnx"], "ONNX version drifted")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    frames = source_inputs(config)
    policies = load_policies(args.policies, torch)

    temp_parent = Path(os.environ.get("LIVIFUSER_BENCHMARK_TEMP", tempfile.gettempdir()))
    temp_parent.mkdir(parents=True, exist_ok=True)
    args.onnx_output.parent.mkdir(parents=True, exist_ok=True)
    if args.onnx_output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite DirectML output or report")
    with tempfile.TemporaryDirectory(prefix="livifuser_splus_dml_", dir=temp_parent) as raw:
        temporary = Path(raw)
        snapshot = extract_backbone(args.backbone, temporary)
        backbone = transformers.AutoModel.from_pretrained(
            snapshot,
            local_files_only=True,
            attn_implementation="eager",
            torch_dtype=torch.float32,
        )
        backbone.eval()

        class FeatureWrapper(torch.nn.Module):
            def __init__(self, model: Any) -> None:
                super().__init__()
                self.model = model

            def forward(self, pixel_values: Any) -> tuple[Any, Any]:
                hidden, pooled = self.model(pixel_values=pixel_values, return_dict=False)[:2]
                spatial = hidden[:, 5:, :].reshape(1, 14, 14, 384)
                patches = spatial.reshape(1, 7, 2, 7, 2, 384).mean(dim=(2, 4))
                return patches.reshape(1, 49, 384), pooled

        wrapper = FeatureWrapper(backbone).eval()
        sample = torch.from_numpy(preprocess_rgb(np.asarray(frames[0]["run"].rgb[0]))).unsqueeze(0)
        temporary_onnx = temporary / "dinov3_splus_features.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (sample,),
                temporary_onnx,
                input_names=[config["export"]["input_name"]],
                output_names=config["export"]["output_names"],
                opset_version=int(config["export"]["opset"]),
                do_constant_folding=True,
                dynamo=False,
            )
        model = onnx.load(temporary_onnx)
        onnx.checker.check_model(model)
        onnx_payload = temporary_onnx.read_bytes()
        args.onnx_output.write_bytes(onnx_payload)

        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            str(args.onnx_output),
            sess_options=options,
            providers=[
                ("DmlExecutionProvider", {"device_id": str(config["runtime"]["device_id"])}),
                "CPUExecutionProvider",
            ],
        )
        require(
            session.get_providers()[0] == "DmlExecutionProvider",
            "DirectML was not selected as primary provider",
        )

        rows = []
        maxima = {
            "patch_max_abs_error": 0.0,
            "pooled_max_abs_error": 0.0,
            "patch_one_minus_cosine": 0.0,
            "pooled_one_minus_cosine": 0.0,
            "downstream_first_action_max_abs_error": 0.0,
        }
        for frame_index, frame in enumerate(frames):
            run_object, row = frame["run"], frame["row"]
            pixel_values = preprocess_rgb(np.asarray(run_object.rgb[row]))[None]
            tensor = torch.from_numpy(pixel_values)
            with torch.inference_mode():
                reference_patch_t, reference_pool_t = wrapper(tensor)
            reference_patch = reference_patch_t.cpu().numpy()
            reference_pool = reference_pool_t.cpu().numpy()
            candidate_patch, candidate_pool = session.run(
                None, {config["export"]["input_name"]: pixel_values}
            )
            metrics = {
                "patch_max_abs_error": float(np.max(np.abs(reference_patch - candidate_patch))),
                "pooled_max_abs_error": float(np.max(np.abs(reference_pool - candidate_pool))),
                "patch_one_minus_cosine": cosine_error(reference_patch, candidate_patch),
                "pooled_one_minus_cosine": cosine_error(reference_pool, candidate_pool),
                "downstream_first_action_max_abs_error": 0.0,
            }
            reference_inputs = policy_input(reference_patch[0], run_object, row, torch)
            candidate_inputs = policy_input(candidate_patch[0], run_object, row, torch)
            action_errors = []
            for variant, seed, policy in policies:
                with torch.inference_mode():
                    reference_action = policy(**reference_inputs)["mean"][0, 0].numpy()
                    candidate_action = policy(**candidate_inputs)["mean"][0, 0].numpy()
                error = float(np.max(np.abs(reference_action - candidate_action)))
                action_errors.append({"variant": variant, "seed": seed, "max_abs_error": error})
            metrics["downstream_first_action_max_abs_error"] = max(
                item["max_abs_error"] for item in action_errors
            )
            for name, value in metrics.items():
                maxima[name] = max(maxima[name], value)
            rows.append(
                {
                    "frame_index": frame_index,
                    "condition": frame["source"]["condition"],
                    "source_root": frame["source"]["root"],
                    "row": row,
                    "preprocessed_rgb_sha256": sha256_bytes(pixel_values.tobytes()),
                    "metrics": metrics,
                    "action_errors": action_errors,
                }
            )

    tolerances = config["tolerances"]
    checks = {
        "patch_max_abs_error": maxima["patch_max_abs_error"] <= tolerances["patch_max_abs_error"],
        "pooled_max_abs_error": maxima["pooled_max_abs_error"]
        <= tolerances["pooled_max_abs_error"],
        "patch_one_minus_cosine": maxima["patch_one_minus_cosine"]
        <= tolerances["patch_one_minus_cosine_max"],
        "pooled_one_minus_cosine": maxima["pooled_one_minus_cosine"]
        <= tolerances["pooled_one_minus_cosine_max"],
        "downstream_first_action_max_abs_error": maxima["downstream_first_action_max_abs_error"]
        <= tolerances["downstream_first_action_max_abs_error"],
    }
    report = {
        "schema_version": "1.0.0",
        "status": "complete",
        "parity_pass": all(checks.values()),
        "checks": checks,
        "maxima": maxima,
        "tolerances": tolerances,
        "frame_count": len(rows),
        "policy_identity_count": len(policies),
        "rows": rows,
        "onnx": {
            "path": str(args.onnx_output.resolve()),
            "size_bytes": args.onnx_output.stat().st_size,
            "sha256": sha256_file(args.onnx_output),
            "opset": int(config["export"]["opset"]),
        },
        "runtime": {
            "providers": session.get_providers(),
            "device_id": int(config["runtime"]["device_id"]),
            "device_identity": config["runtime"]["device_identity"],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "identities": {
            "config_sha256": sha256_bytes(config_payload),
            "backbone_bundle_sha256": sha256_file(args.backbone),
            "policy_payload_sha256": sha256_file(args.policies),
            "code_sha256": sha256_file(Path(__file__)),
        },
        "forbidden_inputs_used": {"heldout": False, "confirmatory": False},
    }
    report["report_sha256_excludes_self"] = sha256_bytes(json_bytes(report))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/dinov3_splus_directml_parity_v1.json",
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
        "--onnx-output",
        type=Path,
        default=ROOT / "artifacts/runtime/models/dinov3_vits16plus_features_opset17.onnx",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "artifacts/runtime/directml_parity_v1.json",
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json_bytes(
            {
                "parity_pass": report["parity_pass"],
                "maxima": report["maxima"],
                "onnx": report["onnx"],
                "report": str(args.report.resolve()),
                "report_sha256_excludes_self": report["report_sha256_excludes_self"],
            }
        ).decode(),
        end="",
    )


if __name__ == "__main__":
    main()
