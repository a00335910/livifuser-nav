"""Pure aggregation helpers for the five-episode pilot analysis.

Episodes remain the statistical unit. Seeds are averaged within each held-out
episode before the episode bootstrap, matching the amended preregistration.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def exact_bootstrap_mean_ci(
    values: Sequence[float], confidence: float = 0.95
) -> dict[str, float | int]:
    """Exact percentile CI over every ordered episode bootstrap resample."""

    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("bootstrap values must be a non-empty 1-D sequence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")
    if samples.size > 8:
        raise ValueError("exact bootstrap is intentionally limited to eight units")
    means = np.fromiter(
        (
            float(samples[np.asarray(indices, dtype=np.int64)].mean())
            for indices in itertools.product(range(samples.size), repeat=samples.size)
        ),
        dtype=np.float64,
        count=int(samples.size**samples.size),
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "confidence": confidence,
        "lower": float(np.percentile(means, 100.0 * tail)),
        "upper": float(np.percentile(means, 100.0 * (1.0 - tail))),
        "resample_count": int(means.size),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    for group, count in enumerate(counts):
        if count > 1:
            members = inverse == group
            ranks[members] = ranks[members].mean()
    return ranks


def spearman_correlation(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    """Tie-aware Spearman correlation, or ``None`` for a constant input."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("Spearman inputs must be matching 1-D sequences of length >= 2")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Spearman inputs must be finite")
    ranked_x = _rankdata(x)
    ranked_y = _rankdata(y)
    if np.ptp(ranked_x) == 0.0 or np.ptp(ranked_y) == 0.0:
        return None
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def risk_at_coverage(
    curve: Sequence[Mapping[str, float]], target: float
) -> dict[str, float]:
    """Return the first saved selective-risk point at or above ``target``."""

    if not 0.0 < target <= 1.0:
        raise ValueError("target coverage must be within (0, 1]")
    if not curve:
        raise ValueError("risk-coverage curve must not be empty")
    for row in curve:
        if float(row["coverage"]) >= target:
            return {
                "coverage": float(row["coverage"]),
                "risk": float(row["risk"]),
                "distance_threshold": float(row["distance_threshold"]),
            }
    raise ValueError("risk-coverage curve does not reach the requested coverage")


def metric_summary(
    records: Iterable[Mapping[str, Any]], metric: str
) -> dict[str, Any]:
    """Summarize one metric while keeping episodes and seeds explicit."""

    rows = list(records)
    if not rows:
        raise ValueError("metric summary requires records")
    episodes = sorted({str(row["episode"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    expected = {(episode, seed) for episode in episodes for seed in seeds}
    observed = {(str(row["episode"]), int(row["seed"])) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("records do not form a complete episode-by-seed grid")
    episode_means = {
        episode: float(
            np.mean([float(row[metric]) for row in rows if row["episode"] == episode])
        )
        for episode in episodes
    }
    seed_means = {
        str(seed): float(
            np.mean([float(row[metric]) for row in rows if int(row["seed"]) == seed])
        )
        for seed in seeds
    }
    seed_values = np.asarray(list(seed_means.values()), dtype=np.float64)
    return {
        "mean": float(np.mean(list(episode_means.values()))),
        "episode_means": episode_means,
        "episode_bootstrap_ci95": exact_bootstrap_mean_ci(
            list(episode_means.values())
        ),
        "seed_means": seed_means,
        "seed_spread": {
            "min": float(seed_values.min()),
            "median": float(np.median(seed_values)),
            "max": float(seed_values.max()),
        },
    }


def paired_metric_summary(
    full_records: Iterable[Mapping[str, Any]],
    comparator_records: Iterable[Mapping[str, Any]],
    metric: str,
) -> dict[str, Any]:
    """Paired comparator-minus-full deltas for a lower-is-better metric."""

    full = {
        (str(row["episode"]), int(row["seed"])): float(row[metric])
        for row in full_records
    }
    comparator = {
        (str(row["episode"]), int(row["seed"])): float(row[metric])
        for row in comparator_records
    }
    if not full or full.keys() != comparator.keys():
        raise ValueError("paired records must have the same episode-by-seed keys")
    deltas = {key: comparator[key] - full[key] for key in full}
    episode_deltas = {
        episode: float(np.mean([value for (item, _), value in deltas.items() if item == episode]))
        for episode in sorted({key[0] for key in deltas})
    }
    positive = sum(value > 0.0 for value in deltas.values())
    negative = sum(value < 0.0 for value in deltas.values())
    ties = len(deltas) - positive - negative
    if positive == len(deltas):
        consistency = "full_better_every_fold_and_seed"
    elif negative == len(deltas):
        consistency = "comparator_better_every_fold_and_seed"
    else:
        consistency = "inconclusive"
    return {
        "delta_definition": "comparator_minus_full; positive favors full",
        "mean_delta": float(np.mean(list(episode_deltas.values()))),
        "episode_deltas": episode_deltas,
        "episode_bootstrap_ci95": exact_bootstrap_mean_ci(
            list(episode_deltas.values())
        ),
        "paired_results": len(deltas),
        "full_wins": positive,
        "comparator_wins": negative,
        "ties": ties,
        "consistency": consistency,
    }
