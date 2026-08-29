#!/usr/bin/env python3
"""Verify and aggregate the returned five-fold Kaggle pilot sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from livifuser_nav.pilot_analysis import (  # noqa: E402
    metric_summary,
    paired_metric_summary,
    risk_at_coverage,
    spearman_correlation,
)

EXPECTED_FOLDS = (
    "clear_001b",
    "center_002b",
    "rightblock_003b",
    "leftblock_004b",
    "gap_005",
)
EXPECTED_SEEDS = (20260805, 20260806, 20260807)
EXPECTED_NAMES = (
    "full",
    "lidar_only",
    "rgb_only",
    "concat",
    "no_fov_mask",
    "no_gate",
    "no_temporal",
    "full_mean_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Extracted Kaggle result directory")
    parser.add_argument("--archive", type=Path, help="Optional returned ZIP to hash")
    parser.add_argument("--output", type=Path, required=True, help="Analysis JSON")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_result(
    path: Path,
    fold: str,
    name: str,
    seed: int,
    expected_run: dict[str, Any],
    summary_row: dict[str, Any],
) -> dict[str, Any]:
    result = load_json(path)
    require(result["run"] == expected_run, f"run identity mismatch: {path}")
    require(int(result["seed"]) == seed, f"seed mismatch: {path}")
    validation = result["validation"]
    count = int(validation["window_count"])
    per_window = validation["per_window"]
    for key in ("episode_ids", "origin_rows", "normalized_mse", "nll", "mahalanobis_distance"):
        require(len(per_window[key]) == count, f"{key} count mismatch: {path}")
    require(
        all(str(episode).endswith(fold) for episode in per_window["episode_ids"]),
        f"held-out episode mismatch: {path}",
    )
    mse = np.asarray(per_window["normalized_mse"], dtype=np.float64)
    nll = np.asarray(per_window["nll"], dtype=np.float64)
    distance = np.asarray(per_window["mahalanobis_distance"], dtype=np.float64)
    require(np.all(np.isfinite(mse)), f"non-finite MSE: {path}")
    require(np.all(np.isfinite(nll)), f"non-finite NLL: {path}")
    require(np.all(np.isfinite(distance)), f"non-finite Mahalanobis distance: {path}")
    macro_mse = float(validation["macro_episode_normalized_mse"])
    macro_nll = float(validation["macro_episode_nll"])
    require(close(float(mse.mean()), macro_mse), f"macro MSE mismatch: {path}")
    require(close(float(nll.mean()), macro_nll), f"macro NLL mismatch: {path}")
    require(len(validation["per_horizon_normalized_mse"]) == 8, f"horizon mismatch: {path}")
    curve = validation["risk_coverage_by_mahalanobis"]
    require(len(curve) == count, f"risk-coverage length mismatch: {path}")
    require(close(float(curve[-1]["coverage"]), 1.0), f"final coverage mismatch: {path}")
    require(close(float(curve[-1]["risk"]), macro_mse), f"final risk mismatch: {path}")
    require(
        close(float(summary_row["macro_episode_nll"]), macro_nll),
        f"summary NLL mismatch: {path}",
    )
    require(
        close(float(summary_row["macro_episode_normalized_mse"]), macro_mse),
        f"summary MSE mismatch: {path}",
    )
    return {
        "episode": fold,
        "name": name,
        "seed": seed,
        "nll": macro_nll,
        "mse": macro_mse,
        "training_seconds": float(result["training"]["seconds"]),
        "horizon_mse": [float(value) for value in validation["per_horizon_normalized_mse"]],
        "sigma": validation["sigma_coverage"],
        "risk_curve": curve,
        "window_mse": mse,
        "window_distance": distance,
    }


def aggregate(input_root: Path, archive: Path | None) -> dict[str, Any]:
    execution = load_json(input_root / "cv_execution_summary.json")
    plan = execution["plan"]
    require(tuple(plan["folds"]) == EXPECTED_FOLDS, "execution fold plan drift")
    require(int(plan["model_seed_results_per_fold"]) == 24, "result-count plan drift")
    require(plan["execution_backend"] == "cuda_isolated_processes", "backend drift")
    require(len(plan["cuda_devices"]) == 2, "expected exactly two CUDA devices")
    require(all(item["name"] == "Tesla T4" for item in plan["cuda_devices"]), "device drift")
    require(all(int(item["returncode"]) == 0 for item in execution["folds"]), "fold failure")

    config = load_json(REPO_ROOT / "config" / "baseline_sweep_pilot5_v1.json")
    require(tuple(int(value) for value in config["seeds"]) == EXPECTED_SEEDS, "seed drift")
    config_runs = {str(item["name"]): item for item in config["runs"]}
    require(tuple(config_runs) == EXPECTED_NAMES, "model plan drift")

    result_files = list(input_root.rglob("result.json"))
    checkpoint_files = list(input_root.rglob("checkpoint.pt"))
    require(len(result_files) == 120, "expected 120 result files")
    require(len(checkpoint_files) == 120, "expected 120 checkpoint files")

    records: list[dict[str, Any]] = []
    fold_mahalanobis: dict[str, Any] = {}
    revisions: set[str] = set()
    config_hashes: set[str] = set()
    for fold in EXPECTED_FOLDS:
        fold_root = input_root / f"held_out_{fold}"
        summary = load_json(fold_root / "summary.json")
        context = load_json(fold_root / "run_context.json")
        progress = load_json(fold_root / "progress.json")
        revisions.add(str(summary["git"]["revision"]).lower())
        config_hashes.add(str(summary["config_sha256"]).lower())
        require(summary["git"]["state"] == "verified_cloud_bundle", f"unverified source: {fold}")
        require(int(progress["completed_result_count"]) == 24, f"incomplete progress: {fold}")
        require(int(progress["planned_result_count"]) == 24, f"progress plan drift: {fold}")
        validation_counts = summary["splits"]["validation"]["episode_window_counts"]
        require(len(validation_counts) == 1, f"validation split is not one episode: {fold}")
        require(next(iter(validation_counts)).endswith(fold), f"validation identity drift: {fold}")
        require(
            len(summary["splits"]["train"]["episode_window_counts"]) == 4,
            f"train split drift: {fold}",
        )
        require(context["splits"] == summary["splits"], f"context/summary split drift: {fold}")
        summary_rows = {
            (str(row["name"]), int(row["seed"])): row for row in summary["results"]
        }
        require(len(summary_rows) == 24, f"summary result count mismatch: {fold}")
        fold_distances: np.ndarray | None = None
        for name in EXPECTED_NAMES:
            for seed in EXPECTED_SEEDS:
                run_root = fold_root / name / f"seed_{seed}"
                require((run_root / "checkpoint.pt").is_file(), f"missing checkpoint: {run_root}")
                record = validate_result(
                    run_root / "result.json",
                    fold,
                    name,
                    seed,
                    config_runs[name],
                    summary_rows[(name, seed)],
                )
                if fold_distances is None:
                    fold_distances = record["window_distance"]
                else:
                    require(
                        np.array_equal(fold_distances, record["window_distance"]),
                        f"Mahalanobis distances vary by model/seed: {fold}",
                    )
                records.append(record)
        assert fold_distances is not None
        train_p95 = float(summary["mahalanobis"]["train_distance"]["p95"])
        fold_mahalanobis[fold] = {
            **summary["mahalanobis"],
            "validation_fraction_above_train_p95": float(np.mean(fold_distances > train_p95)),
        }

    require(len(revisions) == 1, "source revision drift across folds")
    require(len(config_hashes) == 1, "configuration hash drift across folds")
    by_name = {
        name: [record for record in records if record["name"] == name]
        for name in EXPECTED_NAMES
    }
    variants: dict[str, Any] = {}
    for name, rows in by_name.items():
        horizon = np.asarray([row["horizon_mse"] for row in rows], dtype=np.float64)
        training = np.asarray([row["training_seconds"] for row in rows], dtype=np.float64)
        sigma_fields = ("1_sigma", "2_sigma", "3_sigma")
        variants[name] = {
            "mse": metric_summary(rows, "mse"),
            "nll": metric_summary(rows, "nll"),
            "per_horizon_normalized_mse": [float(value) for value in horizon.mean(axis=0)],
            "training_seconds": {
                "median": float(np.median(training)),
                "p95": float(np.percentile(training, 95)),
                "max": float(training.max()),
                "sum": float(training.sum()),
            },
            "sigma_calibration": {
                "nll_comparable": name != "full_mean_only",
                "clamp_floor_fraction": float(
                    np.mean([row["sigma"]["clamp_floor_fraction"] for row in rows])
                ),
                "expected": rows[0]["sigma"]["expected"],
                "linear": {
                    key: float(np.mean([row["sigma"]["linear"][key] for row in rows]))
                    for key in sigma_fields
                },
                "angular": {
                    key: float(np.mean([row["sigma"]["angular"][key] for row in rows]))
                    for key in sigma_fields
                },
            },
        }

    comparisons: dict[str, Any] = {}
    for name in EXPECTED_NAMES[1:]:
        comparisons[name] = {
            "mse": paired_metric_summary(by_name["full"], by_name[name], "mse"),
            "nll": {
                **paired_metric_summary(by_name["full"], by_name[name], "nll"),
                "comparable": name != "full_mean_only",
            },
        }

    full_risk: dict[str, Any] = {}
    for coverage in (0.25, 0.5, 0.75, 1.0):
        points = [risk_at_coverage(row["risk_curve"], coverage) for row in by_name["full"]]
        full_risk[str(coverage)] = {
            "mean_actual_coverage": float(np.mean([point["coverage"] for point in points])),
            "mean_risk": float(np.mean([point["risk"] for point in points])),
        }
    full_risk_at_one = full_risk["1.0"]["mean_risk"]
    for item in full_risk.values():
        item["risk_ratio_to_full_coverage"] = item["mean_risk"] / full_risk_at_one

    correlations = [
        spearman_correlation(row["window_distance"], row["window_mse"])
        for row in by_name["full"]
    ]
    valid_correlations = [value for value in correlations if value is not None]
    mahalanobis_risk = {
        "full_model_selective_risk": full_risk,
        "full_model_spearman_distance_vs_mse": {
            "mean": float(np.mean(valid_correlations)),
            "min": float(np.min(valid_correlations)),
            "median": float(np.median(valid_correlations)),
            "max": float(np.max(valid_correlations)),
            "record_count": len(valid_correlations),
        },
        "fold_distribution_shift": fold_mahalanobis,
        "ood_auroc": None,
        "ood_auroc_reason": (
            "No purpose-recorded OOD probes were collected before the camera failure; "
            "the July shifted-distribution substitute requires a separate evaluation."
        ),
    }

    archive_record = None
    if archive is not None:
        require(archive.is_file(), f"archive does not exist: {archive}")
        archive_record = {
            "path": str(archive.resolve()),
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        }
    return {
        "schema_version": "1.0.0",
        "disposition": (
            "Five-episode leave-one-episode-out development evidence on the temporary "
            "frozen DINOv3 ViT-S/16 baseline; not a held-out test or policy-efficacy result."
        ),
        "integrity": {
            "valid": True,
            "folds": list(EXPECTED_FOLDS),
            "results": len(result_files),
            "checkpoints": len(checkpoint_files),
            "source_revision": next(iter(revisions)),
            "config_sha256": next(iter(config_hashes)).upper(),
            "archive": archive_record,
            "runtime": plan["runtime"],
            "cuda_devices": plan["cuda_devices"],
        },
        "variants": variants,
        "paired_comparisons_against_full": comparisons,
        "mahalanobis": mahalanobis_risk,
        "known_limitations": [
            "Only five episodes from one room; no significance or unseen-environment claim.",
            "No sealed test set exists.",
            "No purpose-recorded OOD probes exist, so the preregistered OOD AUROC is unavailable.",
            "The backbone is frozen DINOv3 ViT-S/16, not the locked final S+/16 model.",
            (
                "The saved sigma coverage points permit coverage-gap diagnostics but "
                "not a separately defined ECE."
            ),
        ],
    }


def main() -> None:
    args = parse_args()
    payload = aggregate(args.input.resolve(), args.archive.resolve() if args.archive else None)
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps({"analysis": str(args.output.resolve()), "integrity": "valid"}, indent=2))


if __name__ == "__main__":
    main()
