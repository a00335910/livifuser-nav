"""Window batch assembly for the baseline sweep.

`run_tiny_overfit.py` tokenized scans lazily behind a per-row cache because it
touched sixteen windows. The sweep touches every window every epoch, so scans
are tokenized once per run up front. Nothing here imports PyTorch; the runner
converts the returned arrays to tensors at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dino_cache import DINOFeatureCache
from .learning_data import WindowDataset, WindowRef, tokenize_lidar


@dataclass(frozen=True, slots=True)
class RunTokens:
    """Precomputed LiDAR tokens for every accepted row of one run."""

    features: np.ndarray  # [rows, sectors, 4] float32
    visual_mask: np.ndarray  # [rows, sectors, 49] bool
    in_fov: np.ndarray  # [rows, sectors] bool


def tokenize_run(run: Any, config: dict[str, Any]) -> RunTokens:
    """Tokenize every scan of one export run with its per-scan geometry."""

    sectors = int(config["lidar_sectors"])
    features = np.empty((run.count, sectors, 4), dtype=np.float32)
    visual_mask = np.empty((run.count, sectors, 49), dtype=bool)
    in_fov = np.empty((run.count, sectors), dtype=bool)
    for row in range(run.count):
        tokens = tokenize_lidar(
            run.scan_ranges[row],
            int(run.vectors["scan_beam_count"][row]),
            float(run.vectors["scan_angle_increment_rad"][row]),
            run.manifest,
            sectors=sectors,
            range_clip_m=float(config["lidar_range_clip_m"]),
            visual_radius=int(config["visual_mask_radius_tokens"]),
        )
        features[row] = tokens.features
        visual_mask[row] = tokens.visual_mask
        in_fov[row] = tokens.in_fov
    return RunTokens(features, visual_mask, in_fov)


def window_arrays(
    dataset: WindowDataset,
    caches: list[DINOFeatureCache],
    run_tokens: list[RunTokens],
    refs: list[WindowRef],
) -> dict[str, Any]:
    """Stack model inputs for `refs` in order, plus per-window identity.

    `episode_ids` and `origin_rows` travel with the arrays so downstream
    reporting can aggregate by episode and align pooled features for the
    Mahalanobis risk–coverage analysis without re-deriving window layout.
    """

    if len(caches) != len(dataset.runs) or len(run_tokens) != len(dataset.runs):
        raise ValueError("one cache and one token set are required per run")
    visual: list[np.ndarray] = []
    lidar: list[np.ndarray] = []
    mask: list[np.ndarray] = []
    fov: list[np.ndarray] = []
    goal: list[np.ndarray] = []
    state: list[np.ndarray] = []
    target: list[np.ndarray] = []
    pooled: list[np.ndarray] = []
    episode_ids: list[str] = []
    origin_rows: list[int] = []
    for ref in refs:
        run = dataset.runs[ref.run_index]
        rows = list(ref.context_rows)
        tokens = run_tokens[ref.run_index]
        visual.append(
            np.asarray(caches[ref.run_index].patch_tokens[rows], dtype=np.float32)
        )
        lidar.append(tokens.features[rows])
        mask.append(tokens.visual_mask[rows])
        fov.append(tokens.in_fov[rows])
        goal.append(np.asarray(run.vectors["goal"][rows], dtype=np.float32))
        state.append(np.asarray(run.vectors["robot_state"][rows], dtype=np.float32))
        target.append(dataset.targets(ref).astype(np.float32))
        pooled.append(
            np.asarray(
                caches[ref.run_index].pooled_features[ref.origin_row], dtype=np.float32
            )
        )
        episode_ids.append(run.run_id)
        origin_rows.append(ref.origin_row)
    return {
        "visual_tokens": np.stack(visual),
        "lidar_features": np.stack(lidar),
        "visual_mask": np.stack(mask),
        "in_fov": np.stack(fov),
        "goal": np.stack(goal),
        "robot_state": np.stack(state),
        "target": np.stack(target),
        "origin_pooled_features": np.stack(pooled),
        "episode_ids": episode_ids,
        "origin_rows": origin_rows,
    }
