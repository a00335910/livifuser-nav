#!/usr/bin/env python3
"""Run the deterministic `_05` + `_06` Stage 2 end-to-end overfit gate."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from livifuser_nav.dino_cache import DINOFeatureCache
from livifuser_nav.learning_data import WindowDataset, WindowRef, sha256_file, tokenize_lidar
from livifuser_nav.model import (
    ACTION_SCALE,
    LiViFuserPolicy,
    heteroscedastic_nll,
    mean_warmup_loss,
)
from livifuser_nav.training import (
    action_class,
    evaluate_acceptance,
    per_window_errors,
    phase_learning_rate,
    select_tiny_windows,
)


def build_batch(
    dataset: WindowDataset,
    caches: list[DINOFeatureCache],
    refs: list[WindowRef],
    config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    visual_batch: list[np.ndarray] = []
    lidar_batch: list[np.ndarray] = []
    mask_batch: list[np.ndarray] = []
    fov_batch: list[np.ndarray] = []
    goal_batch: list[np.ndarray] = []
    state_batch: list[np.ndarray] = []
    target_batch: list[np.ndarray] = []
    selection_records: list[dict[str, Any]] = []
    token_cache: dict[tuple[int, int], Any] = {}
    for ref in refs:
        run = dataset.runs[ref.run_index]
        rows = list(ref.context_rows)
        visual_batch.append(np.asarray(caches[ref.run_index].patch_tokens[rows], dtype=np.float32))
        lidar_context = []
        mask_context = []
        fov_context = []
        for row in rows:
            key = (ref.run_index, row)
            if key not in token_cache:
                token_cache[key] = tokenize_lidar(
                    run.scan_ranges[row],
                    int(run.vectors["scan_beam_count"][row]),
                    float(run.vectors["scan_angle_increment_rad"][row]),
                    run.manifest,
                    sectors=int(config["lidar_sectors"]),
                    range_clip_m=float(config["lidar_range_clip_m"]),
                    visual_radius=int(config["visual_mask_radius_tokens"]),
                )
            tokens = token_cache[key]
            lidar_context.append(tokens.features)
            mask_context.append(tokens.visual_mask)
            fov_context.append(tokens.in_fov)
        lidar_batch.append(np.stack(lidar_context))
        mask_batch.append(np.stack(mask_context))
        fov_batch.append(np.stack(fov_context))
        goal_batch.append(np.asarray(run.vectors["goal"][rows], dtype=np.float32))
        state_batch.append(np.asarray(run.vectors["robot_state"][rows], dtype=np.float32))
        target = dataset.targets(ref).astype(np.float32)
        target_batch.append(target)
        selection_records.append(
            {
                "run_id": run.run_id,
                "origin_row": ref.origin_row,
                "context_rows": rows,
                "action_rows": list(ref.action_rows),
                "first_action_class": action_class(target[0]),
                "target": target.tolist(),
            }
        )
    arrays = {
        "visual_tokens": np.stack(visual_batch),
        "lidar_features": np.stack(lidar_batch),
        "visual_mask": np.stack(mask_batch),
        "in_fov": np.stack(fov_batch),
        "goal": np.stack(goal_batch),
        "robot_state": np.stack(state_batch),
        "target": np.stack(target_batch),
    }
    tensors = {
        name: torch.from_numpy(value if value.flags.writeable else value.copy())
        for name, value in arrays.items()
    }
    return tensors, selection_records


def metrics(mean: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    error = mean - target
    return {
        "mse_normalized": float(
            ((error / error.new_tensor(ACTION_SCALE)).square()).mean().item()
        ),
        "mae_linear_mps": float(error[..., 0].abs().mean().item()),
        "mae_angular_radps": float(error[..., 1].abs().mean().item()),
        "rmse_linear_mps": float(error[..., 0].square().mean().sqrt().item()),
        "rmse_angular_radps": float(error[..., 1].square().mean().sqrt().item()),
        "max_abs_linear_mps": float(error[..., 0].abs().max().item()),
        "max_abs_angular_radps": float(error[..., 1].abs().max().item()),
    }


def git_state() -> dict[str, str]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], check=True, capture_output=True, text=True
    ).stdout
    return {"revision": revision, "state": "dirty" if status else "clean"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/tiny_overfit.json"))
    parser.add_argument("--export", type=Path, action="append", required=True)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.export) != 2 or len(args.cache) != 2:
        raise ValueError(
            "the pilot gate requires exactly the accepted `_05` and `_06` exports/caches"
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty result {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text("utf-8"))
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(config["torch_threads"]))

    dataset = WindowDataset(
        args.export, context_k=int(config["context_k"]), horizon_h=int(config["horizon_h"])
    )
    expected_counts = [1400, 948]
    actual_counts = [
        sum(1 for ref in dataset.windows if ref.run_index == run_index)
        for run_index in range(2)
    ]
    if actual_counts != expected_counts:
        raise ValueError(f"unexpected `_05`/`_06` window counts: {actual_counts}")
    caches = [
        DINOFeatureCache(cache_root, dataset.runs[index])
        for index, cache_root in enumerate(args.cache)
    ]
    refs = select_tiny_windows(dataset, int(config["windows_per_action_class_per_run"]))
    batch, selections = build_batch(dataset, caches, refs, config)
    target = batch.pop("target")
    model = LiViFuserPolicy()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["warmup_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    model.train()
    with torch.no_grad():
        initial_outputs = model(**batch)
        initial_metrics = metrics(initial_outputs["mean"], target)
    history: list[dict[str, Any]] = []
    warmup_steps = int(config["warmup_steps"])
    nll_steps = int(config["nll_steps"])
    if warmup_steps < 1:
        raise ValueError("the mean-MSE phase must run at least one step to be scored")
    total_steps = warmup_steps + nll_steps
    log_every = int(config.get("log_every_steps", 10))
    started = time.perf_counter()
    for step in range(total_steps):
        warmup = step < warmup_steps
        if warmup:
            rate = phase_learning_rate(
                step, 0, warmup_steps, float(config["warmup_learning_rate"]), config
            )
        else:
            rate = phase_learning_rate(
                step, warmup_steps, nll_steps, float(config["nll_learning_rate"]), config
            )
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = mean_warmup_loss(outputs, target) if warmup else heteroscedastic_nll(outputs, target)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["gradient_clip_norm"])
        )
        optimizer.step()
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == total_steps:
            current = metrics(outputs["mean"].detach(), target)
            history.append(
                {
                    "step": step + 1,
                    "phase": "mean_mse_warmup" if warmup else "heteroscedastic_nll",
                    "learning_rate": rate,
                    "loss": float(loss.detach().item()),
                    "gradient_norm_before_clip": float(gradient_norm),
                    **current,
                }
            )
        if not math.isfinite(float(loss.detach())):
            raise RuntimeError(f"non-finite loss at step {step + 1}")
        if step + 1 == warmup_steps:
            # The mean-fit gate is scored here: the phase that follows optimizes
            # a different objective and may trade mean accuracy for variance.
            with torch.no_grad():
                mean_phase_outputs = model(**batch)
                mean_phase_metrics = metrics(mean_phase_outputs["mean"], target)
                mean_phase_windows = per_window_errors(
                    mean_phase_outputs["mean"].numpy(), target.numpy(), selections
                )
    training_seconds = time.perf_counter() - started

    model.eval()
    with torch.no_grad():
        final_outputs = model(**batch)
        final_metrics = metrics(final_outputs["mean"], target)
        window_errors = per_window_errors(
            final_outputs["mean"].numpy(), target.numpy(), selections
        )
        clipped_log_variance = final_outputs["log_variance"].clamp(-5.0, 2.0)
        for index, record in enumerate(window_errors):
            record["max_log_variance_linear"] = float(
                clipped_log_variance[index, :, 0].max()
            )
            record["max_log_variance_angular"] = float(
                clipped_log_variance[index, :, 1].max()
            )
        gate = final_outputs["gate"]
        front = batch["in_fov"].unsqueeze(-1).expand_as(gate)
        timings = []
        single = {name: value[:1] for name, value in batch.items()}
        for _ in range(30):
            tick = time.perf_counter()
            model(**single)
            timings.append((time.perf_counter() - tick) * 1000.0)

    checkpoint_path = args.output / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_path)
    result = {
        "experiment_id": config["experiment_id"],
        "disposition": "pipeline-development tiny-subset overfit; not policy efficacy evidence",
        "git": git_state(),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "backbone_deviation": config["temporary_backbone"],
        "acceptance": (
            {
                "declared_in_config_before_the_run": True,
                **evaluate_acceptance(
                    config["acceptance"],
                    initial=initial_metrics,
                    mean_phase=mean_phase_metrics,
                    mean_phase_windows=mean_phase_windows,
                    uncertainty_phase=final_metrics,
                ),
            }
            if "acceptance" in config
            else {
                "declared_in_config_before_the_run": False,
                "note": "This configuration predates the preregistered gate criteria.",
            }
        ),
        "dataset": {
            "run_ids": [run.run_id for run in dataset.runs],
            "exports": [str(run.root) for run in dataset.runs],
            "feature_caches": [str(cache.root) for cache in caches],
            "window_counts": actual_counts,
            "total_window_count": len(dataset),
            "selected_window_count": len(refs),
            "selection": (
                "deterministic uniformly-spaced origins per first-action class per run"
            ),
            "selected_windows": selections,
            "limitations": [
                "Both episodes are diagnostic pipeline pilots, not final "
                "protocol-clean training data.",
                "The recorded goal is the constant baseline [1.0, 0.0, 1.0].",
                "The keyboard interface was latched and physical operator "
                "observations were not recorded.",
                "Episode `_07` is rejected and was not loaded or counted.",
            ],
        },
        "shapes": {name: list(value.shape) for name, value in {**batch, "target": target}.items()},
        "model": {
            "parameter_count": parameter_count,
            "common_width": 256,
            "cross_attention_heads": 4,
            "visual_tokens": 49,
            "lidar_tokens": 80,
            "context_k": 8,
            "horizon_h": 8,
            "output": "H x 4: mean linear/angular plus log variance linear/angular",
        },
        "training": {
            "seconds": training_seconds,
            "steps": total_steps,
            "history": history,
            "initial_metrics": initial_metrics,
            "mean_phase_final_metrics": mean_phase_metrics,
            "mean_phase_per_window_errors": mean_phase_windows,
            "final_metrics": final_metrics,
            "per_window_final_errors": window_errors,
            "phase_note": (
                "`mean_phase_*` is the end of the mean-MSE phase and scores the "
                "memorization gate. `final_*` is the end of the heteroscedastic "
                "phase, which optimizes calibrated variance and may raise the "
                "mean error where the recorded command is discontinuous."
            ),
            "final_log_variance": {
                "min": float(clipped_log_variance.min()),
                "mean": float(clipped_log_variance.mean()),
                "max": float(clipped_log_variance.max()),
            },
            "front_gate": {
                "element_count": int(front.sum()),
                "mean": float(gate[front].mean()),
                "min": float(gate[front].min()),
                "max": float(gate[front].max()),
            },
        },
        "predictions": final_outputs["mean"].tolist(),
        "timing": {
            "cached_feature_policy_batch1_ms": {
                "median": float(np.median(timings)),
                "p95": float(np.percentile(timings, 95)),
                "max": float(np.max(timings)),
                "samples": len(timings),
            },
            "scope": (
                "CPU policy path from cached DINO tokens; excludes capture, RGB "
                "preprocess, DINO backbone, safety, and publish"
            ),
        },
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(result_path),
                "initial_metrics": initial_metrics,
                "final_metrics": final_metrics,
                "acceptance": result["acceptance"],
                "training_seconds": training_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

