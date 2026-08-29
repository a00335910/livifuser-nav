#!/usr/bin/env python3
"""Inference-only exploratory comparison of pilot uncertainty signals.

The training outputs preregister sigma coverage and Mahalanobis risk-coverage,
but they do not retain per-window predicted variance. This script reconstructs
that missing prediction from the returned full-model checkpoints. No weights
are updated. Scalar reductions of the Hx2 aleatoric tensor are explicitly
post-hoc diagnostics, not preregistered headline metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from run_baseline_sweep import evaluate_model, load_split  # noqa: E402

from livifuser_nav.evaluation import (  # noqa: E402
    auroc,
    mahalanobis_distances,
    risk_coverage,
    window_nll,
    window_normalized_mse,
)
from livifuser_nav.model import LiViFuserPolicy  # noqa: E402
from livifuser_nav.pilot_analysis import (  # noqa: E402
    metric_summary,
    paired_metric_summary,
    risk_at_coverage,
    spearman_correlation,
)

FOLDS = (
    "clear_001b",
    "center_002b",
    "rightblock_003b",
    "leftblock_004b",
    "gap_005",
)
SEEDS = (20260805, 20260806, 20260807)
COVERAGES = (0.25, 0.5, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--shifted-export", type=Path, action="append")
    parser.add_argument("--shifted-cache", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=2e-5, abs_tol=2e-7):
        raise ValueError(f"{label}: reconstructed {observed} != saved {expected}")


def local_split_path(recorded: str, local_root: Path) -> Path:
    resolved = local_root / Path(recorded).name
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def signal_records(
    episode: str,
    seed: int,
    errors: np.ndarray,
    scores: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mean_error = float(errors.mean())
    for name, score in scores.items():
        curve = risk_coverage(errors, score)
        record: dict[str, Any] = {
            "episode": episode,
            "seed": seed,
            "signal": name,
            "spearman": spearman_correlation(score, errors),
        }
        for coverage in COVERAGES:
            point = risk_at_coverage(curve, coverage)
            record[f"risk_ratio_{coverage}"] = float(point["risk"] / mean_error)
            record[f"actual_coverage_{coverage}"] = float(point["coverage"])
        rows.append(record)
    return rows


def aleatoric_scores(log_variance: np.ndarray) -> dict[str, np.ndarray]:
    clipped = np.clip(np.asarray(log_variance, dtype=np.float64), -5.0, 2.0)
    sigma = np.exp(0.5 * clipped)
    variance = np.exp(clipped)
    return {
        "aleatoric_mean_variance_h8x2": variance.mean(axis=(1, 2)),
        "aleatoric_max_sigma_h8x2": sigma.max(axis=(1, 2)),
        "aleatoric_first_step_max_sigma": sigma[:, 0, :].max(axis=1),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = load_json(REPO_ROOT / "config" / "baseline_sweep_pilot5_v1.json")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(config["torch_threads"]))
    device = torch.device("cpu")
    shifted_exports = args.shifted_export or []
    shifted_caches = args.shifted_cache or []
    if len(shifted_exports) != len(shifted_caches):
        raise ValueError("one shifted cache is required per shifted export")
    if shifted_exports:
        shifted_dataset, shifted_feature_caches, shifted_tokens = load_split(
            shifted_exports,
            shifted_caches,
            config,
            "shifted distribution",
        )
    else:
        shifted_dataset = None
        shifted_feature_caches = []
        shifted_tokens = []
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    distribution_records: list[dict[str, Any]] = []
    checkpoint_checks: list[dict[str, Any]] = []

    for fold in FOLDS:
        fold_root = args.sweep_root / f"held_out_{fold}"
        summary = load_json(fold_root / "summary.json")
        export = local_split_path(
            summary["splits"]["validation"]["exports"][0], args.export_root
        )
        # Cache directories use the export basename plus the frozen-backbone suffix.
        cache = args.cache_root / f"{export.name}_dino_s16"
        if not cache.is_dir():
            raise FileNotFoundError(cache)
        dataset, caches, tokens = load_split([export], [cache], config, "validation")

        for seed in SEEDS:
            result_path = fold_root / "full" / f"seed_{seed}" / "result.json"
            checkpoint_path = fold_root / "full" / f"seed_{seed}" / "checkpoint.pt"
            saved = load_json(result_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint["variant"] != "full" or int(checkpoint["seed"]) != seed:
                raise ValueError(f"checkpoint identity mismatch: {checkpoint_path}")
            if str(checkpoint["config_sha256"]).lower() != str(
                summary["config_sha256"]
            ).lower():
                raise ValueError(f"checkpoint configuration mismatch: {checkpoint_path}")
            model = LiViFuserPolicy(variant="full")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            evaluation = evaluate_model(model, dataset, caches, tokens, config, device)
            if evaluation["origin_rows"] != saved["validation"]["per_window"]["origin_rows"]:
                raise ValueError(f"window identity mismatch: {checkpoint_path}")
            mean = np.asarray(evaluation["mean"], dtype=np.float64)
            log_variance = np.asarray(evaluation["log_variance"], dtype=np.float64)
            target = np.asarray(evaluation["target"], dtype=np.float64)
            errors = window_normalized_mse(mean, target)
            nll = window_nll(mean, log_variance, target)
            saved_mse = float(saved["validation"]["macro_episode_normalized_mse"])
            saved_nll = float(saved["validation"]["macro_episode_nll"])
            require_close(float(errors.mean()), saved_mse, f"{fold}/{seed} MSE")
            require_close(float(nll.mean()), saved_nll, f"{fold}/{seed} NLL")
            mahalanobis = np.asarray(
                saved["validation"]["per_window"]["mahalanobis_distance"],
                dtype=np.float64,
            )
            scores = {
                "mahalanobis": mahalanobis,
                **aleatoric_scores(log_variance),
            }
            records.extend(signal_records(fold, seed, errors, scores))
            if shifted_dataset is not None:
                shifted_evaluation = evaluate_model(
                    model,
                    shifted_dataset,
                    shifted_feature_caches,
                    shifted_tokens,
                    config,
                    device,
                )
                shifted_aleatoric = aleatoric_scores(
                    np.asarray(shifted_evaluation["log_variance"], dtype=np.float64)
                )
                mean = np.load(fold_root / "mahalanobis_mean.npy")
                precision = np.load(fold_root / "mahalanobis_precision.npy")
                shifted_pooled = np.asarray(
                    [
                        shifted_feature_caches[ref.run_index].pooled_features[ref.origin_row]
                        for ref in shifted_dataset.windows
                    ],
                    dtype=np.float64,
                )
                shifted_scores = {
                    "mahalanobis": mahalanobis_distances(
                        shifted_pooled, mean, precision
                    ),
                    **shifted_aleatoric,
                }
                shifted_episode_ids = np.asarray(shifted_evaluation["episode_ids"])
                id_episode_ids = np.asarray(evaluation["episode_ids"])
                for signal, id_score in scores.items():
                    probe_score = shifted_scores[signal]
                    record = {
                        "episode": fold,
                        "seed": seed,
                        "signal": signal,
                        "auroc": auroc(probe_score, id_score),
                        "negative_auroc": -auroc(probe_score, id_score),
                        "per_probe_auroc": {},
                    }
                    for probe in sorted(set(shifted_episode_ids)):
                        record["per_probe_auroc"][str(probe)] = auroc(
                            probe_score[shifted_episode_ids == probe], id_score
                        )
                    if set(id_episode_ids) != {next(iter(set(id_episode_ids)))}:
                        raise ValueError(f"ID evaluation spans multiple episodes: {fold}")
                    distribution_records.append(record)
            checkpoint_checks.append(
                {
                    "episode": fold,
                    "seed": seed,
                    "window_count": int(errors.size),
                    "reconstructed_mse": float(errors.mean()),
                    "saved_mse": saved_mse,
                    "reconstructed_nll": float(nll.mean()),
                    "saved_nll": saved_nll,
                }
            )
            print(f"verified and scored {fold} seed {seed}", flush=True)

    signals = sorted({str(row["signal"]) for row in records})
    aggregates: dict[str, Any] = {}
    for signal in signals:
        member = [row for row in records if row["signal"] == signal]
        aggregates[signal] = {
            "spearman_vs_window_mse": metric_summary(member, "spearman"),
            "selective_risk_ratio": {
                str(coverage): metric_summary(member, f"risk_ratio_{coverage}")
                for coverage in COVERAGES
            },
        }

    comparisons: dict[str, Any] = {}
    mahalanobis_rows = [row for row in records if row["signal"] == "mahalanobis"]
    for signal in signals:
        if signal == "mahalanobis":
            continue
        member = [row for row in records if row["signal"] == signal]
        comparisons[signal] = {
            str(coverage): paired_metric_summary(
                mahalanobis_rows, member, f"risk_ratio_{coverage}"
            )
            for coverage in COVERAGES
        }

    distribution_shift: dict[str, Any] | None = None
    if distribution_records:
        distribution_signals: dict[str, Any] = {}
        for signal in signals:
            member = [row for row in distribution_records if row["signal"] == signal]
            probes = sorted(member[0]["per_probe_auroc"])
            distribution_signals[signal] = {
                "combined_probe_auroc": metric_summary(member, "auroc"),
                "per_probe_auroc": {
                    probe: metric_summary(
                        [
                            {
                                **row,
                                "probe_auroc": row["per_probe_auroc"][probe],
                            }
                            for row in member
                        ],
                        "probe_auroc",
                    )
                    for probe in probes
                },
            }
        mahalanobis_distribution = [
            row for row in distribution_records if row["signal"] == "mahalanobis"
        ]
        distribution_pairs = {}
        for signal in signals:
            if signal == "mahalanobis":
                continue
            member = [row for row in distribution_records if row["signal"] == signal]
            distribution_pairs[signal] = {
                **paired_metric_summary(
                    mahalanobis_distribution, member, "negative_auroc"
                ),
                "delta_interpretation": (
                    "mahalanobis AUROC minus aleatoric AUROC; positive favors Mahalanobis"
                ),
            }
        distribution_shift = {
            "probe_kind": "shifted_distribution",
            "claim_limitation": (
                "The probes confound room, layout, and operator interface. Results measure "
                "detection of a different recording context, not novel-object OOD."
            ),
            "signals": distribution_signals,
            "mahalanobis_minus_aleatoric_paired_auroc": distribution_pairs,
        }

    payload = {
        "schema_version": "1.0.0",
        "disposition": (
            "Inference-only exploratory uncertainty comparison on the five real pilot "
            "episodes; no training or checkpoint selection was performed."
        ),
        "post_hoc_score_definitions": {
            "mahalanobis": "Saved distance from the train-fold frozen-feature Gaussian.",
            "aleatoric_mean_variance_h8x2": (
                "Mean exp(clipped log variance) over eight horizons and two actions."
            ),
            "aleatoric_max_sigma_h8x2": (
                "Maximum exp(0.5 * clipped log variance) over all horizons/actions."
            ),
            "aleatoric_first_step_max_sigma": (
                "Maximum normalized sigma over linear/angular at the executed first step."
            ),
        },
        "integrity": {
            "valid": True,
            "checkpoint_count": len(checkpoint_checks),
            "checkpoint_metric_reconstructions": checkpoint_checks,
        },
        "signals": aggregates,
        "aleatoric_minus_mahalanobis_paired_risk": comparisons,
        "shifted_distribution_separation": distribution_shift,
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "Scalar aleatoric reductions were not fixed in the preregistration.",
            "Only five same-room episodes are available.",
            "No purpose-recorded OOD probes exist.",
        ],
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"output": str(args.output), "integrity": "valid"}, indent=2))


if __name__ == "__main__":
    main()
