"""Held-out evaluation and Mahalanobis OOD scoring for the baseline sweep.

Everything here decides whether a model or uncertainty signal is any good, so
it lives in `src/` and is unit tested. Nothing imports PyTorch: the sweep
runner converts tensors to arrays at the boundary, keeping this importable on
hosts without torch.

Conventions shared with `model.py` (pinned by a cross-check test where torch
is available): actions are normalized by `ACTION_SCALE` per channel, and log
variance is clamped to `LOG_VARIANCE_CLAMP` before use.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ACTION_SCALE = (0.10, 0.50)
LOG_VARIANCE_CLAMP = (-5.0, 2.0)


def normalized_error(mean: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-element error in units of each channel's commanded range."""

    if mean.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    scale = np.asarray(ACTION_SCALE, dtype=np.float64)
    return (np.asarray(mean, dtype=np.float64) - np.asarray(target, dtype=np.float64)) / scale


def window_nll(mean: np.ndarray, log_variance: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-window heteroscedastic Gaussian NLL, matching the training loss.

    Inputs are `[windows, horizon, 2]`; the result is `[windows]`, averaged
    over horizon and channels exactly as `heteroscedastic_nll` averages.
    """

    if mean.shape != target.shape or log_variance.shape != mean.shape:
        raise ValueError("mean, log_variance, and target shapes must match")
    error = normalized_error(mean, target)
    clamped = np.clip(
        np.asarray(log_variance, dtype=np.float64),
        LOG_VARIANCE_CLAMP[0],
        LOG_VARIANCE_CLAMP[1],
    )
    values = 0.5 * (np.exp(-clamped) * np.square(error) + clamped)
    return values.mean(axis=(1, 2))


def window_normalized_mse(mean: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-window normalized MSE `[windows]`, matching the runner's metric."""

    return np.square(normalized_error(mean, target)).mean(axis=(1, 2))


def per_horizon_normalized_mse(mean: np.ndarray, target: np.ndarray) -> list[float]:
    """Normalized MSE per horizon step h=1..H so degradation stays visible."""

    return [
        float(value)
        for value in np.square(normalized_error(mean, target)).mean(axis=(0, 2))
    ]


def per_episode_summary(
    episode_ids: list[str], window_values: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Episode-level aggregation: episodes, not windows, are the analysis unit."""

    values = np.asarray(window_values, dtype=np.float64)
    if len(episode_ids) != values.shape[0]:
        raise ValueError("one episode id is required per window value")
    summary: dict[str, dict[str, Any]] = {}
    for episode in sorted(set(episode_ids)):
        member = values[np.asarray([item == episode for item in episode_ids])]
        summary[episode] = {
            "window_count": int(member.shape[0]),
            "mean": float(member.mean()),
            "median": float(np.median(member)),
            "p95": float(np.percentile(member, 95)),
            "max": float(member.max()),
        }
    return summary


def sigma_coverage(
    mean: np.ndarray, log_variance: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    """Empirical fraction of errors within 1/2/3 predicted sigma per channel.

    A calibrated Gaussian head gives approximately 0.683/0.954/0.997. Also
    reports the fraction of elements pinned at the lower clamp, because a
    clamp-saturated head has no measurable calibration (the tiny overfit had
    14 of 16 windows saturated).
    """

    error = np.abs(
        np.asarray(mean, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    ) / np.asarray(ACTION_SCALE, dtype=np.float64)
    clamped = np.clip(
        np.asarray(log_variance, dtype=np.float64),
        LOG_VARIANCE_CLAMP[0],
        LOG_VARIANCE_CLAMP[1],
    )
    sigma = np.exp(0.5 * clamped)
    result: dict[str, Any] = {
        "clamp_floor_fraction": float(np.mean(clamped <= LOG_VARIANCE_CLAMP[0])),
        "expected": {"1_sigma": 0.6827, "2_sigma": 0.9545, "3_sigma": 0.9973},
    }
    for channel, name in enumerate(("linear", "angular")):
        channel_error = error[..., channel]
        channel_sigma = sigma[..., channel]
        result[name] = {
            f"{k}_sigma": float(np.mean(channel_error <= k * channel_sigma))
            for k in (1, 2, 3)
        }
    return result


def fit_gaussian(
    features: np.ndarray, shrinkage: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and shrunk-precision fit for Mahalanobis distance (PDF §4.2).

    Shrinkage blends the sample covariance toward the scaled identity:
    `(1 - s) * Sigma + s * (trace(Sigma) / d) * I`, keeping the inverse stable
    when rows barely exceed the 384 feature dimensions.
    """

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("features must be [rows, dimensions]")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be within [0, 1]")
    if values.shape[0] < 2:
        raise ValueError("fitting requires at least two rows")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / (values.shape[0] - 1)
    dimensions = covariance.shape[0]
    scaled_identity = (np.trace(covariance) / dimensions) * np.eye(dimensions)
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * scaled_identity
    return mean, np.linalg.inv(shrunk)


def mahalanobis_distances(
    features: np.ndarray, mean: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    centered = np.asarray(features, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    squared = np.einsum("rd,de,re->r", centered, precision, centered)
    return np.sqrt(np.maximum(squared, 0.0))


def auroc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """Rank-based AUROC with tie handling; positives should score higher."""

    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("both score sets must be non-empty")
    combined = np.concatenate((positive, negative))
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1)
    # Average ranks across ties so tied scores contribute 0.5.
    for value in np.unique(combined):
        tied = combined == value
        if np.count_nonzero(tied) > 1:
            ranks[tied] = ranks[tied].mean()
    positive_rank_sum = ranks[: positive.size].sum()
    expected_minimum = positive.size * (positive.size + 1) / 2.0
    return float(
        (positive_rank_sum - expected_minimum) / (positive.size * negative.size)
    )


def risk_coverage(errors: np.ndarray, distances: np.ndarray) -> list[dict[str, float]]:
    """Selective-prediction curve: retain windows in ascending-distance order.

    At each coverage level the risk is the mean error over the retained
    windows. A useful OOD signal yields risk that falls as coverage drops
    (the highest-distance windows were also the worst-predicted ones).
    """

    error_values = np.asarray(errors, dtype=np.float64)
    distance_values = np.asarray(distances, dtype=np.float64)
    if error_values.shape != distance_values.shape or error_values.ndim != 1:
        raise ValueError("errors and distances must be matching 1-D arrays")
    if error_values.size == 0:
        raise ValueError("risk-coverage requires at least one window")
    order = np.argsort(distance_values, kind="mergesort")
    ordered_errors = error_values[order]
    cumulative = np.cumsum(ordered_errors)
    counts = np.arange(1, error_values.size + 1)
    return [
        {
            "coverage": float(count / error_values.size),
            "risk": float(cumulative[count - 1] / count),
            "distance_threshold": float(distance_values[order][count - 1]),
        }
        for count in counts
    ]
