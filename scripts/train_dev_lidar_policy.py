#!/usr/bin/env python3
"""Train and export the development-only LiDAR policy for the C3 vacuity gate."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "livifuser_sim"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from livifuser_sim.lidar_policy_features import lidar_only_features  # noqa: E402
from torch import nn  # noqa: E402

from livifuser_nav.learning_data import WindowDataset, sha256_file  # noqa: E402
from livifuser_nav.model import ACTION_SCALE, LiViFuserPolicy  # noqa: E402


class LidarPolicyOnnx(nn.Module):
    """Three-input deployment surface for the structurally LiDAR-only model."""

    def __init__(self, model: LiViFuserPolicy) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        lidar_features: torch.Tensor,
        goal: torch.Tensor,
        robot_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = lidar_features.shape[0]
        unused_visual = lidar_features.new_zeros((batch, 8, 1, 1))
        unused_mask = torch.zeros((batch, 8, 1, 1), dtype=torch.bool)
        unused_fov = torch.zeros((batch, 8, 1), dtype=torch.bool)
        outputs = self.model(
            visual_tokens=unused_visual,
            lidar_features=lidar_features,
            visual_mask=unused_mask,
            in_fov=unused_fov,
            goal=goal,
            robot_state=robot_state,
        )
        return outputs["mean"], outputs["log_variance"]


def _git_identity() -> dict[str, str]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"revision": revision, "state": "dirty" if status else "clean"}


def _batch(
    dataset: WindowDataset,
    tokens: list[np.ndarray],
    indices: np.ndarray,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    lidar: list[np.ndarray] = []
    goal: list[np.ndarray] = []
    state: list[np.ndarray] = []
    target: list[np.ndarray] = []
    for index in indices:
        ref = dataset.windows[int(index)]
        rows = list(ref.context_rows)
        run = dataset.runs[ref.run_index]
        lidar.append(tokens[ref.run_index][rows])
        goal.append(np.asarray(run.vectors["goal"][rows], dtype=np.float32))
        state.append(np.asarray(run.vectors["robot_state"][rows], dtype=np.float32))
        target.append(dataset.targets(ref).astype(np.float32))
    inputs = {
        "lidar_features": torch.from_numpy(np.stack(lidar)),
        "goal": torch.from_numpy(np.stack(goal)),
        "robot_state": torch.from_numpy(np.stack(state)),
    }
    return inputs, torch.from_numpy(np.stack(target))


def _forward(model: LiViFuserPolicy, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batch = inputs["lidar_features"].shape[0]
    return model(
        visual_tokens=torch.zeros((batch, 8, 1, 1)),
        lidar_features=inputs["lidar_features"],
        visual_mask=torch.zeros((batch, 8, 1, 1), dtype=torch.bool),
        in_fov=torch.zeros((batch, 8, 1), dtype=torch.bool),
        goal=inputs["goal"],
        robot_state=inputs["robot_state"],
    )


def _normalized_mse(mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = target.new_tensor(ACTION_SCALE)
    return torch.mean(((mean - target) / scale) ** 2)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tokenize_lidar_only_run(run: Any, config: dict[str, Any]) -> np.ndarray:
    geometry = run.manifest["calibration"]["lidar_geometry"]["angular_frame"]
    features = np.empty(
        (run.count, int(config["lidar_sectors"]), 4), dtype=np.float32
    )
    for row in range(run.count):
        beam_count = int(run.vectors["scan_beam_count"][row])
        features[row] = lidar_only_features(
            run.scan_ranges[row, :beam_count],
            angle_min_rad=float(geometry["angle_min_rad"]),
            angle_increment_rad=float(run.vectors["scan_angle_increment_rad"][row]),
            range_min_m=float(geometry["range_min_m"]),
            range_max_m=float(geometry["range_max_m"]),
            sectors=int(config["lidar_sectors"]),
            range_clip_m=float(config["lidar_range_clip_m"]),
        )
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--export", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty result {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if len(args.export) < 2:
        raise ValueError("C3 development training requires both dev geometries")

    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(config["torch_threads"]))
    dataset = WindowDataset(
        args.export,
        context_k=int(config["context_k"]),
        horizon_h=int(config["horizon_h"]),
    )
    if len(dataset) < 100:
        raise ValueError("development dataset has too few valid windows")
    run_tokens = [_tokenize_lidar_only_run(run, config) for run in dataset.runs]
    window_counts = {
        run.run_id: sum(ref.run_index == index for ref in dataset.windows)
        for index, run in enumerate(dataset.runs)
    }

    model = LiViFuserPolicy(variant="lidar_only")
    initial_checkpoint_sha256 = None
    if args.initial_checkpoint is not None:
        checkpoint = torch.load(
            args.initial_checkpoint, map_location="cpu", weights_only=True
        )
        if checkpoint.get("variant") != "lidar_only":
            raise ValueError("initial checkpoint is not a lidar_only model")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        initial_checkpoint_sha256 = sha256_file(args.initial_checkpoint)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(dataset))
    cursor = 0
    sampling_probabilities = None
    near_goal_factor = float(config.get("near_goal_oversample_factor", 1.0))
    near_goal_threshold = float(config.get("near_goal_threshold_m", 0.5))
    if near_goal_factor < 1.0:
        raise ValueError("near_goal_oversample_factor must be at least one")
    if near_goal_factor > 1.0:
        weights = np.ones(len(dataset), dtype=np.float64)
        for index, ref in enumerate(dataset.windows):
            run = dataset.runs[ref.run_index]
            rho_m = float(run.vectors["goal"][ref.origin_row, 0])
            if rho_m < near_goal_threshold:
                weights[index] = near_goal_factor
        sampling_probabilities = weights / weights.sum()
    steps = int(config["steps"])
    batch_size = int(config["batch_size"])
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        if sampling_probabilities is None:
            if cursor + batch_size > len(order):
                order = generator.permutation(len(dataset))
                cursor = 0
            indices = order[cursor : cursor + batch_size]
            cursor += batch_size
        else:
            indices = generator.choice(
                len(dataset), size=batch_size, replace=True, p=sampling_probabilities
            )
        inputs, target = _batch(dataset, run_tokens, indices)
        progress = step / max(1, steps - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        rate = float(config["learning_rate"]) * (
            float(config["lr_floor_fraction"])
            + (1.0 - float(config["lr_floor_fraction"])) * cosine
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward(model, inputs)
        loss = _normalized_mse(outputs["mean"], target)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["gradient_clip_norm"])
        )
        optimizer.step()
        if not math.isfinite(float(loss.detach())):
            raise RuntimeError(f"non-finite loss at step {step + 1}")
        if step == 0 or (step + 1) % int(config["log_every_steps"]) == 0 or step + 1 == steps:
            record = {
                "step": step + 1,
                "learning_rate": rate,
                "normalized_mse": float(loss.detach()),
                "gradient_norm_before_clip": float(gradient_norm),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    training_seconds = time.perf_counter() - started

    model.eval()
    all_indices = np.arange(len(dataset))
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(dataset), 64):
            inputs, target = _batch(dataset, run_tokens, all_indices[start : start + 64])
            predictions.append(_forward(model, inputs)["mean"].numpy())
            targets.append(target.numpy())
    predicted = np.concatenate(predictions)
    target_values = np.concatenate(targets)
    normalized_mse = float(
        np.mean(((predicted - target_values) / np.asarray(ACTION_SCALE)) ** 2)
    )

    checkpoint_path = args.output / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "variant": "lidar_only",
            "seed": seed,
            "config": config,
        },
        checkpoint_path,
    )
    onnx_path = args.output / "lidar_policy.onnx"
    wrapper = LidarPolicyOnnx(model)
    example = {
        "lidar_features": torch.zeros((1, 8, 80, 4), dtype=torch.float32),
        "goal": torch.zeros((1, 8, 3), dtype=torch.float32),
        "robot_state": torch.zeros((1, 8, 2), dtype=torch.float32),
    }
    torch.onnx.export(
        wrapper,
        tuple(example.values()),
        onnx_path,
        input_names=list(example),
        output_names=["mean", "log_variance"],
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_mean, torch_log_variance = wrapper(*example.values())
    onnx_mean, onnx_log_variance = session.run(
        None, {name: value.numpy() for name, value in example.items()}
    )
    np.testing.assert_allclose(onnx_mean, torch_mean.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        onnx_log_variance, torch_log_variance.numpy(), rtol=1e-5, atol=1e-6
    )

    result = {
        "experiment_id": config["experiment_id"],
        "disposition": config["disposition"],
        "git": _git_identity(),
        "config": config,
        "config_sha256": sha256_file(args.config),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
        },
        "dataset": {
            "exports": [str(path.resolve()) for path in args.export],
            "manifest_sha256": [sha256_file(path / "manifest.json") for path in args.export],
            "window_counts": window_counts,
            "total_windows": len(dataset),
            "sampling": {
                "near_goal_threshold_m": near_goal_threshold,
                "near_goal_oversample_factor": near_goal_factor,
            },
        },
        "training": {
            "seconds": training_seconds,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "history": history,
            "in_sample_normalized_mse": normalized_mse,
        },
        "deployment": {
            "inputs": {item.name: item.shape for item in session.get_inputs()},
            "outputs": {item.name: item.shape for item in session.get_outputs()},
            "onnx_parity": {"rtol": 1e-5, "atol": 1e-6, "passed": True},
        },
        "outputs": {
            checkpoint_path.name: sha256_file(checkpoint_path),
            onnx_path.name: sha256_file(onnx_path),
        },
        "initial_checkpoint": {
            "path": None if args.initial_checkpoint is None else str(args.initial_checkpoint),
            "sha256": initial_checkpoint_sha256,
        },
    }
    _write_json(args.output / "manifest.json", result)
    print(json.dumps(result["training"], indent=2))


if __name__ == "__main__":
    main()
