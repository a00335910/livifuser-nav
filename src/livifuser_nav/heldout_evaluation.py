"""Frozen pure-NumPy statistics for simulation held-out evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

EXPECTED_COVERAGE = np.asarray((0.6827, 0.9545, 0.9973), dtype=np.float64)
REQUESTED_COVERAGES = np.asarray(
    tuple(round(value, 2) for value in np.arange(1.0, 0.09, -0.05)),
    dtype=np.float64,
)


def require_finite(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def right_continuous_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference_array = require_finite(reference, "CDF reference")
    value_array = require_finite(values, "CDF values")
    if reference_array.ndim != 1 or reference_array.size == 0:
        raise ValueError("CDF reference must be a non-empty vector")
    return (
        np.searchsorted(np.sort(reference_array), value_array, side="right") / reference_array.size
    )


def episode_reduce(
    episode_ids: Sequence[str],
    values: np.ndarray,
    reduction: str,
) -> tuple[np.ndarray, np.ndarray]:
    value_array = require_finite(values, "episode values")
    if value_array.ndim != 1 or len(episode_ids) != value_array.size:
        raise ValueError("episode reduction requires one scalar per identity")
    identities = np.asarray(sorted(set(str(value) for value in episode_ids)), dtype=np.str_)
    output = np.empty(identities.size, dtype=np.float64)
    for index, identity in enumerate(identities):
        members = value_array[
            np.asarray([str(value) == identity for value in episode_ids], dtype=bool)
        ]
        if reduction == "mean":
            output[index] = members.mean()
        elif reduction == "max":
            output[index] = members.max()
        else:
            raise ValueError(f"unsupported episode reduction: {reduction}")
    return identities, output


def grouped_average_precision(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> float:
    positive = require_finite(positive_scores, "positive scores").reshape(-1)
    negative = require_finite(negative_scores, "negative scores").reshape(-1)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("both score classes must be non-empty")
    scores = np.concatenate((positive, negative))
    labels = np.concatenate(
        (np.ones(positive.size, dtype=np.int8), np.zeros(negative.size, dtype=np.int8))
    )
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    result = 0.0
    for threshold in np.unique(scores)[::-1]:
        selected = scores == threshold
        true_positive += int(labels[selected].sum())
        false_positive += int(selected.sum() - labels[selected].sum())
        recall = true_positive / positive.size
        precision = true_positive / (true_positive + false_positive)
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return float(result)


def fpr_at_recall(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    recall: float = 0.95,
) -> float:
    positive = require_finite(positive_scores, "positive scores").reshape(-1)
    negative = require_finite(negative_scores, "negative scores").reshape(-1)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("both score classes must be non-empty")
    if not 0.0 < recall <= 1.0:
        raise ValueError("recall must be in (0, 1]")
    scores = np.concatenate((positive, negative))
    candidates = []
    for threshold in np.unique(scores)[::-1]:
        true_positive_rate = float(np.mean(positive >= threshold))
        if true_positive_rate >= recall:
            candidates.append(float(np.mean(negative >= threshold)))
    if not candidates:
        candidates.append(1.0)
    return min(candidates)


def midrank_auroc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    positive = require_finite(positive_scores, "positive scores").reshape(-1)
    negative = require_finite(negative_scores, "negative scores").reshape(-1)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("both score classes must be non-empty")
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def discrimination_metrics(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> dict[str, Any]:
    positive = require_finite(positive_scores, "positive scores").reshape(-1)
    negative = require_finite(negative_scores, "negative scores").reshape(-1)
    return {
        "positive_episodes": int(positive.size),
        "negative_episodes": int(negative.size),
        "auroc": midrank_auroc(positive, negative),
        "average_precision": grouped_average_precision(positive, negative),
        "fpr_at_95_recall": fpr_at_recall(positive, negative, 0.95),
    }


def macro_sigma_calibration(
    episode_ids: Sequence[str],
    mean: np.ndarray,
    log_variance: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    predicted = require_finite(mean, "mean")
    variance_log = require_finite(log_variance, "log variance")
    truth = require_finite(target, "target")
    if predicted.shape != variance_log.shape or predicted.shape != truth.shape:
        raise ValueError("calibration arrays must have matching shapes")
    if predicted.ndim != 3 or predicted.shape[2] != 2:
        raise ValueError("calibration arrays must be [windows,horizon,2]")
    if len(episode_ids) != predicted.shape[0]:
        raise ValueError("one episode id is required per window")
    clamped = np.clip(variance_log, -5.0, 2.0)
    sigma = np.exp(0.5 * clamped)
    scale = np.asarray((0.10, 0.50), dtype=np.float64)
    error = np.abs(predicted - truth) / scale
    identities = sorted(set(str(value) for value in episode_ids))
    episode_coverages = np.empty((len(identities), 2, 3), dtype=np.float64)
    episode_floor = np.empty(len(identities), dtype=np.float64)
    for index, identity in enumerate(identities):
        selected = np.asarray([str(value) == identity for value in episode_ids], dtype=bool)
        episode_floor[index] = np.mean(clamped[selected] <= -5.0)
        for channel in range(2):
            for sigma_index, multiple in enumerate((1, 2, 3)):
                episode_coverages[index, channel, sigma_index] = np.mean(
                    error[selected, :, channel] <= multiple * sigma[selected, :, channel]
                )
    macro = episode_coverages.mean(axis=0)
    return {
        "episode_count": len(identities),
        "expected": EXPECTED_COVERAGE.tolist(),
        "linear": macro[0].tolist(),
        "angular": macro[1].tolist(),
        "clamp_floor_fraction": float(episode_floor.mean()),
        "six_point_calibration_mae": float(np.mean(np.abs(macro - EXPECTED_COVERAGE[None, :]))),
    }


def episode_risk_coverage(
    episode_ids: Sequence[str],
    episode_risk: np.ndarray,
    episode_score: np.ndarray,
) -> list[dict[str, Any]]:
    identities = np.asarray([str(value) for value in episode_ids], dtype=np.str_)
    risk = require_finite(episode_risk, "episode risk").reshape(-1)
    score = require_finite(episode_score, "episode score").reshape(-1)
    if identities.size == 0 or identities.shape != risk.shape or risk.shape != score.shape:
        raise ValueError("risk coverage requires matching non-empty episode vectors")
    if len(set(identities.tolist())) != identities.size:
        raise ValueError("risk coverage episode identities must be unique")
    order = np.lexsort((identities, score))
    result = []
    for requested in REQUESTED_COVERAGES:
        retained = int(math.ceil(float(requested) * identities.size))
        selected = order[:retained]
        result.append(
            {
                "requested_coverage": float(requested),
                "retained_episode_count": retained,
                "realized_coverage": float(retained / identities.size),
                "risk": float(risk[selected].mean()),
                "score_threshold": float(score[selected].max()),
                "retained_episode_ids": identities[selected].tolist(),
            }
        )
    return result


def exact_two_sided_sign_test(differences: np.ndarray) -> dict[str, Any]:
    values = require_finite(differences, "sign-test differences").reshape(-1)
    nonzero = values[values != 0]
    positives = int(np.count_nonzero(nonzero > 0))
    denominator = int(nonzero.size)
    if denominator == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(denominator, k) for k in range(positives + 1))
        opposite = sum(math.comb(denominator, k) for k in range(positives, denominator + 1))
        p_value = min(1.0, 2.0 * min(tail, opposite) / (2**denominator))
    return {
        "positive_worlds": positives,
        "negative_worlds": int(denominator - positives),
        "ties_excluded": int(values.size - denominator),
        "denominator": denominator,
        "two_sided_p_value": float(p_value),
    }


def hierarchical_paired_bootstrap(
    worlds: Sequence[str],
    episode_ids: Sequence[str],
    differences: np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 20_260_824,
) -> dict[str, Any]:
    world_array = np.asarray([str(value) for value in worlds], dtype=np.str_)
    identity_array = np.asarray([str(value) for value in episode_ids], dtype=np.str_)
    values = require_finite(differences, "paired differences").reshape(-1)
    if (
        values.size == 0
        or world_array.shape != values.shape
        or identity_array.shape != values.shape
    ):
        raise ValueError("bootstrap vectors must be matching and non-empty")
    unique_worlds = np.asarray(sorted(set(world_array.tolist())), dtype=np.str_)
    if unique_worlds.size != 2:
        raise ValueError("frozen simulation bootstrap requires exactly two worlds")
    if replicates != 10_000 or seed != 20_260_824:
        raise ValueError("bootstrap replicates or seed differ from the frozen contract")
    by_world = {world: np.flatnonzero(world_array == world) for world in unique_worlds}
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_worlds = rng.choice(unique_worlds, size=unique_worlds.size, replace=True)
        sampled_values = []
        for world in sampled_worlds:
            indices = by_world[str(world)]
            sampled = rng.choice(indices, size=indices.size, replace=True)
            sampled_values.append(values[sampled])
        bootstrap[replicate] = np.concatenate(sampled_values).mean()
    world_means = np.asarray(
        [values[by_world[str(world)]].mean() for world in unique_worlds],
        dtype=np.float64,
    )
    interval = np.quantile(bootstrap, (0.025, 0.975), method="linear")
    return {
        "point_estimate": float(values.mean()),
        "worlds": unique_worlds.tolist(),
        "world_mean_differences": world_means.tolist(),
        "replicates": replicates,
        "rng_seed": seed,
        "quantile_method": "linear",
        "confidence_interval_95": interval.tolist(),
        "bootstrap_replicates": bootstrap.tolist(),
        "sign_test": exact_two_sided_sign_test(world_means),
    }
