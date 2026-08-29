#!/usr/bin/env python3
"""Train and evaluate the Architecture v1.1 §8.1 offline baseline sweep.

Every configured run (model variant × loss mode) trains once per seed on the
train-split exports and is evaluated on the validation-split exports, with
episodes as the unit of analysis. The Mahalanobis fit uses train-split pooled
features only. Test-split exports must not be passed to this script; the
sealed test evaluation is a separate, single, preregistered run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from livifuser_nav.batching import RunTokens, tokenize_run, window_arrays  # noqa: E402
from livifuser_nav.dino_cache import DINOFeatureCache  # noqa: E402
from livifuser_nav.evaluation import (  # noqa: E402
    fit_gaussian,
    mahalanobis_distances,
    per_episode_summary,
    per_horizon_normalized_mse,
    risk_coverage,
    sigma_coverage,
    window_nll,
    window_normalized_mse,
)
from livifuser_nav.learning_data import WindowDataset, sha256_file  # noqa: E402
from livifuser_nav.model import (  # noqa: E402
    LiViFuserPolicy,
    heteroscedastic_nll,
    mean_warmup_loss,
)
from livifuser_nav.training import phase_learning_rate  # noqa: E402

TENSOR_KEYS = (
    "visual_tokens",
    "lidar_features",
    "visual_mask",
    "in_fov",
    "goal",
    "robot_state",
)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_frozen_execution(config: dict[str, Any]) -> None:
    """Refuse a deliberately proposed, not-yet-approved scientific config."""

    execution_freeze = config.get("execution_freeze")
    if execution_freeze is not None and execution_freeze.get("status") != "frozen":
        raise RuntimeError(
            "training execution config is not frozen; record explicit approval before launch"
        )


def summary_row(result: dict[str, Any]) -> dict[str, Any]:
    run = result["run"]
    training = result["training"]
    validation = result["validation"]
    return {
        "name": run["name"],
        "variant": run["variant"],
        "loss": run["loss"],
        "seed": int(result["seed"]),
        "parameter_count": int(training["parameter_count"]),
        "training_seconds": float(training["seconds"]),
        "macro_episode_nll": float(validation["macro_episode_nll"]),
        "macro_episode_normalized_mse": float(
            validation["macro_episode_normalized_mse"]
        ),
    }


def completed_result(
    run_dir: Path, run_config: dict[str, Any], seed: int
) -> dict[str, Any] | None:
    """Return a validated completed result, or None for a new empty run."""

    result_path = run_dir / "result.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    if not run_dir.exists():
        return None
    entries = list(run_dir.iterdir())
    if not entries:
        return None
    if not result_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError(f"refusing ambiguous partial run directory: {run_dir}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("run") != run_config or int(result.get("seed", -1)) != seed:
        raise RuntimeError(f"completed result identity mismatch: {run_dir}")
    summary_row(result)
    return result


def ensure_run_context(path: Path, provenance: dict[str, Any]) -> None:
    """Pin resumed work to the exact original code, config, host, and split."""

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != provenance:
            raise RuntimeError("resume context differs from the original sweep context")
        return
    write_json_atomic(path, provenance)


def git_state() -> dict[str, str]:
    try:
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
    except (FileNotFoundError, subprocess.CalledProcessError):
        manifest_path = REPO_ROOT / "cloud_bundle_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                "neither Git identity nor cloud bundle identity is available"
            ) from None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "revision": str(manifest["git_revision"]),
            "state": "verified_cloud_bundle",
        }


def load_split(
    exports: list[Path], caches: list[Path], config: dict[str, Any], label: str
) -> tuple[WindowDataset, list[DINOFeatureCache], list[RunTokens]]:
    if len(exports) != len(caches):
        raise ValueError(f"{label}: one cache is required per export, in the same order")
    dataset = WindowDataset(
        exports,
        context_k=int(config["context_k"]),
        horizon_h=int(config["horizon_h"]),
        load_rgb=False,
    )
    feature_caches = [
        DINOFeatureCache(cache_root, dataset.runs[index])
        for index, cache_root in enumerate(caches)
    ]
    tokens = [tokenize_run(run, config) for run in dataset.runs]
    return dataset, feature_caches, tokens


def common_cache_identity(caches: list[DINOFeatureCache]) -> dict[str, Any]:
    if not caches:
        raise ValueError("at least one feature cache is required")
    identity = caches[0].cache_identity
    if any(cache.cache_identity != identity for cache in caches[1:]):
        raise ValueError("feature caches do not share one backbone contract")
    return identity


def selected_runs(config: dict[str, Any], requested: list[str] | None) -> list[dict[str, Any]]:
    runs = list(config["runs"])
    if requested is None:
        return runs
    if len(requested) != len(set(requested)):
        raise ValueError("run-name selections must be unique")
    by_name = {str(run["name"]): run for run in runs}
    unknown = set(requested).difference(by_name)
    if unknown:
        raise ValueError(f"unknown run-name selections: {sorted(unknown)}")
    return [by_name[name] for name in requested]


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = 0 if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index is unavailable: {index}")
        return torch.device("cuda", index)
    if device.type != "cpu":
        raise ValueError(f"unsupported training device: {requested}")
    return device


def device_provenance(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "device_index": index,
                "device_name": torch.cuda.get_device_name(index),
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
            }
        )
    return result


def batch_tensors(
    arrays: dict[str, Any], device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    tensors = {
        key: torch.from_numpy(arrays[key]).to(device=device, non_blocking=True)
        for key in TENSOR_KEYS
    }
    target = torch.from_numpy(arrays["target"]).to(
        device=device, non_blocking=True
    )
    return tensors, target


def evaluate_model(
    model: LiViFuserPolicy,
    dataset: WindowDataset,
    caches: list[DINOFeatureCache],
    tokens: list[RunTokens],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray | list[str] | list[int]]:
    """Forward every window of `dataset` in evaluation batches."""

    eval_batch = int(config["eval_batch_size"])
    means: list[np.ndarray] = []
    log_variances: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    episode_ids: list[str] = []
    origin_rows: list[int] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset.windows), eval_batch):
            refs = list(dataset.windows[start : start + eval_batch])
            arrays = window_arrays(dataset, caches, tokens, refs)
            tensors, target = batch_tensors(arrays, device)
            outputs = model(**tensors)
            means.append(outputs["mean"].detach().cpu().numpy())
            log_variances.append(outputs["log_variance"].detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            episode_ids.extend(arrays["episode_ids"])
            origin_rows.extend(arrays["origin_rows"])
    return {
        "mean": np.concatenate(means),
        "log_variance": np.concatenate(log_variances),
        "target": np.concatenate(targets),
        "episode_ids": episode_ids,
        "origin_rows": origin_rows,
    }


def train_one(
    run_config: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    train_dataset: WindowDataset,
    train_caches: list[DINOFeatureCache],
    train_tokens: list[RunTokens],
    device: torch.device,
) -> tuple[LiViFuserPolicy, dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    model = LiViFuserPolicy(variant=str(run_config["variant"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["warmup_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    warmup_steps = int(config["warmup_steps"])
    nll_steps = int(config["nll_steps"])
    total_steps = warmup_steps + nll_steps
    batch_size = int(config["batch_size"])
    mean_only = str(run_config["loss"]) == "mean_only"
    if str(run_config["loss"]) not in ("mean_only", "heteroscedastic"):
        raise ValueError(f"unknown loss mode {run_config['loss']!r}")
    shuffle = np.random.default_rng(seed)
    order = shuffle.permutation(len(train_dataset.windows))
    cursor = 0
    history: list[dict[str, Any]] = []
    log_every = int(config.get("log_every_steps", 100))
    model.train()
    started = time.perf_counter()
    for step in range(total_steps):
        if cursor + batch_size > len(order):
            order = shuffle.permutation(len(train_dataset.windows))
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += batch_size
        refs = [train_dataset.windows[int(index)] for index in indices]
        arrays = window_arrays(train_dataset, train_caches, train_tokens, refs)
        tensors, target = batch_tensors(arrays, device)
        warmup = mean_only or step < warmup_steps
        if step < warmup_steps:
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
        outputs = model(**tensors)
        loss = (
            mean_warmup_loss(outputs, target)
            if warmup
            else heteroscedastic_nll(outputs, target)
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["gradient_clip_norm"])
        )
        optimizer.step()
        loss_value = float(loss.detach().item())
        if not np.isfinite(loss_value):
            raise RuntimeError(f"non-finite loss at step {step + 1}")
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == total_steps:
            history.append(
                {
                    "step": step + 1,
                    "phase": "mean" if warmup else "heteroscedastic_nll",
                    "learning_rate": rate,
                    "loss": loss_value,
                    "gradient_norm_before_clip": float(gradient_norm),
                }
            )
    record = {
        "seconds": time.perf_counter() - started,
        "steps": total_steps,
        "parameter_count": sum(item.numel() for item in model.parameters()),
        "history": history,
    }
    return model, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-export", type=Path, action="append", required=True)
    parser.add_argument("--train-cache", type=Path, action="append", required=True)
    parser.add_argument("--val-export", type=Path, action="append", required=True)
    parser.add_argument("--val-cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device. Each Kaggle worker receives an isolated cuda:0.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip only model/seed runs with validated result and checkpoint files.",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        help="Train only this configured run name; repeat for an isolated GPU partition.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text("utf-8"))
    require_frozen_execution(config)
    execution_runs = selected_runs(config, args.run_name)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite non-empty result {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(int(config["torch_threads"]))
    device = resolve_device(args.device)

    print("loading train split ...", flush=True)
    train_dataset, train_caches, train_tokens = load_split(
        args.train_export, args.train_cache, config, "train"
    )
    print("loading validation split ...", flush=True)
    val_dataset, val_caches, val_tokens = load_split(
        args.val_export, args.val_cache, config, "validation"
    )
    overlap = {run.run_id for run in train_dataset.runs} & {
        run.run_id for run in val_dataset.runs
    }
    if overlap:
        raise ValueError(f"episodes appear in both splits: {sorted(overlap)}")
    backbone = common_cache_identity([*train_caches, *val_caches])
    configured_backbone = config.get("backbone")
    configured_deviation = config.get("temporary_backbone")
    if configured_backbone is not None and configured_backbone != backbone:
        raise ValueError("configured backbone identity does not match feature caches")
    if configured_deviation is not None and backbone["status"] != "temporary_deviation":
        raise ValueError("temporary-backbone config cannot train on official S+ caches")
    if configured_backbone is None and configured_deviation is None:
        raise ValueError("config must declare backbone or temporary_backbone")

    print("fitting Mahalanobis Gaussian on train pooled features ...", flush=True)
    train_pooled = np.concatenate(
        [np.asarray(cache.pooled_features, dtype=np.float64) for cache in train_caches]
    )
    shrinkage = float(config["mahalanobis_shrinkage"])
    gaussian_mean, precision = fit_gaussian(train_pooled, shrinkage)
    train_distances = mahalanobis_distances(train_pooled, gaussian_mean, precision)
    val_origin_pooled = np.asarray(
        [
            val_caches[ref.run_index].pooled_features[ref.origin_row]
            for ref in val_dataset.windows
        ],
        dtype=np.float64,
    )
    val_distances = mahalanobis_distances(val_origin_pooled, gaussian_mean, precision)
    np.save(args.output / "mahalanobis_mean.npy", gaussian_mean)
    np.save(args.output / "mahalanobis_precision.npy", precision)

    provenance = {
        "sweep_id": config["sweep_id"],
        "git": git_state(),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "platform": platform.platform(),
        },
        "device": device_provenance(device),
        "backbone": backbone,
        "backbone_deviation": configured_deviation,
        "execution_run_names": [str(run["name"]) for run in execution_runs],
        "splits": {
            "train": {
                "exports": [str(run.root) for run in train_dataset.runs],
                "export_manifest_sha256": [
                    sha256_file(run.root / "manifest.json") for run in train_dataset.runs
                ],
                "cache_manifest_sha256": [
                    sha256_file(cache.root / "manifest.json") for cache in train_caches
                ],
                "episode_window_counts": {
                    run.run_id: sum(
                        1
                        for ref in train_dataset.windows
                        if ref.run_index == index
                    )
                    for index, run in enumerate(train_dataset.runs)
                },
            },
            "validation": {
                "exports": [str(run.root) for run in val_dataset.runs],
                "export_manifest_sha256": [
                    sha256_file(run.root / "manifest.json") for run in val_dataset.runs
                ],
                "cache_manifest_sha256": [
                    sha256_file(cache.root / "manifest.json") for cache in val_caches
                ],
                "episode_window_counts": {
                    run.run_id: sum(
                        1 for ref in val_dataset.windows if ref.run_index == index
                    )
                    for index, run in enumerate(val_dataset.runs)
                },
            },
        },
        "mahalanobis": {
            "shrinkage": shrinkage,
            "fit_rows": int(train_pooled.shape[0]),
            "train_distance": {
                "median": float(np.median(train_distances)),
                "p95": float(np.percentile(train_distances, 95)),
                "max": float(train_distances.max()),
            },
            "validation_origin_distance": {
                "median": float(np.median(val_distances)),
                "p95": float(np.percentile(val_distances, 95)),
                "max": float(val_distances.max()),
            },
        },
    }
    ensure_run_context(args.output / "run_context.json", provenance)

    summary_rows: list[dict[str, Any]] = []
    planned_result_count = len(execution_runs) * len(config["seeds"])
    for run_config in execution_runs:
        for seed in (int(value) for value in config["seeds"]):
            name = str(run_config["name"])
            run_dir = args.output / name / f"seed_{seed}"
            existing = completed_result(run_dir, run_config, seed)
            if existing is not None:
                summary_rows.append(summary_row(existing))
                print(f"skipping completed {name} seed {seed}", flush=True)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"training {name} seed {seed} ...", flush=True)
            model, training_record = train_one(
                run_config,
                seed,
                config,
                train_dataset,
                train_caches,
                train_tokens,
                device,
            )
            evaluation = evaluate_model(
                model, val_dataset, val_caches, val_tokens, config, device
            )
            nll = window_nll(
                evaluation["mean"], evaluation["log_variance"], evaluation["target"]
            )
            mse = window_normalized_mse(evaluation["mean"], evaluation["target"])
            episode_nll = per_episode_summary(evaluation["episode_ids"], nll)
            episode_mse = per_episode_summary(evaluation["episode_ids"], mse)
            macro_nll = float(np.mean([item["mean"] for item in episode_nll.values()]))
            macro_mse = float(np.mean([item["mean"] for item in episode_mse.values()]))
            result = {
                "run": run_config,
                "seed": seed,
                "training": training_record,
                "validation": {
                    "window_count": int(mse.shape[0]),
                    "macro_episode_nll": macro_nll,
                    "macro_episode_normalized_mse": macro_mse,
                    "per_episode_nll": episode_nll,
                    "per_episode_normalized_mse": episode_mse,
                    "per_horizon_normalized_mse": per_horizon_normalized_mse(
                        evaluation["mean"], evaluation["target"]
                    ),
                    "sigma_coverage": sigma_coverage(
                        evaluation["mean"],
                        evaluation["log_variance"],
                        evaluation["target"],
                    ),
                    "risk_coverage_by_mahalanobis": risk_coverage(mse, val_distances),
                    "per_window": {
                        "episode_ids": evaluation["episode_ids"],
                        "origin_rows": evaluation["origin_rows"],
                        "normalized_mse": [float(value) for value in mse],
                        "nll": [float(value) for value in nll],
                        "mahalanobis_distance": [float(value) for value in val_distances],
                    },
                },
            }
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "variant": run_config["variant"],
                    "seed": seed,
                    "config_sha256": provenance["config_sha256"],
                },
                run_dir / "checkpoint.pt",
            )
            write_json_atomic(run_dir / "result.json", result)
            summary_rows.append(summary_row(result))
            write_json_atomic(
                args.output / "progress.json",
                {
                    "completed_result_count": len(summary_rows),
                    "planned_result_count": planned_result_count,
                    "completed": [
                        {"name": row["name"], "seed": row["seed"]}
                        for row in summary_rows
                    ],
                },
            )
            print(
                f"  {name} seed {seed}: macro episode NLL {macro_nll:.4f}, "
                f"macro episode normalized MSE {macro_mse:.5f}",
                flush=True,
            )

    summary = {
        **provenance,
        "disposition": config.get(
            "disposition",
            "validation-split comparison only; not policy efficacy or test-split evidence",
        ),
        "results": summary_rows,
    }
    write_json_atomic(args.output / "summary.json", summary)
    print(json.dumps({"summary": str(args.output / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
