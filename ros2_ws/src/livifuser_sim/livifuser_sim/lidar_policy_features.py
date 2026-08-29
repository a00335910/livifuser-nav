"""Calibration-independent LiDAR-only policy tokenization.

This is the feature subset used by the Stage 2 ``lidar_only`` model. It stays
separate from visual projection so the ROS deployment path neither needs a
camera calibration nor silently invents visual inputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def lidar_only_features(
    ranges: Sequence[float],
    *,
    angle_min_rad: float,
    angle_increment_rad: float,
    range_min_m: float,
    range_max_m: float,
    sectors: int = 80,
    range_clip_m: float = 10.0,
) -> np.ndarray:
    """Aggregate a variable-length scan into ``[range,sin,cos,valid]`` sectors."""

    beam_count = len(ranges)
    if not 1 <= sectors <= beam_count:
        raise ValueError("sectors must be between one and beam count")
    if not math.isfinite(angle_increment_rad) or angle_increment_rad <= 0.0:
        raise ValueError("angle increment must be finite and positive")
    if not 0.0 < range_min_m < range_max_m:
        raise ValueError("range bounds must be positive and ordered")
    if not math.isfinite(range_clip_m) or range_clip_m <= 0.0:
        raise ValueError("range clip must be finite and positive")

    raw = np.asarray(ranges, dtype=np.float64)
    theta = angle_min_rad + np.arange(beam_count, dtype=np.float64) * angle_increment_rad
    valid = np.isfinite(raw) & (raw >= range_min_m) & (raw <= range_max_m)
    features = np.empty((sectors, 4), dtype=np.float32)
    for sector in range(sectors):
        start = sector * beam_count // sectors
        end = (sector + 1) * beam_count // sectors
        sector_valid = valid[start:end]
        fraction = float(np.mean(sector_valid))
        bearing = float(np.mean(theta[start:end]))
        if np.any(sector_valid):
            measured = raw[start:end][sector_valid]
            normalized_range = float(np.mean(np.minimum(measured, range_clip_m)) / range_clip_m)
        else:
            normalized_range = 1.0
        features[sector] = (
            normalized_range,
            math.sin(bearing),
            math.cos(bearing),
            fraction,
        )
    return features
