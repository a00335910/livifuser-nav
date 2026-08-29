"""Read-only live overlay state for the teleop demonstration.

The demonstration drives the robot by hand and runs a trained policy beside it
as a commentator: every tick the policy answers what it *would* command, and the
answer is drawn next to what the operator actually did. Nothing here publishes,
and nothing here imports PyTorch — the runner converts arrays to tensors at the
boundary exactly as `run_baseline_sweep.py` does.

Two rules are load-bearing and are the reason this is a tested module rather
than script code:

1. The K-step window is never padded across a gap. A dropped scan resets the
   window instead of manufacturing continuity the policy never saw in training.
2. The live window must be assembled from the same tokenization the training
   export used, including that export's calibration block, or the policy is
   being shown inputs it was not trained on.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .learning_data import LidarTokens, tokenize_lidar
from .replay_safety import SafetyAuditError, is_forbidden_publisher

#: Grid period of the locked 10 Hz association contract.
TICK_PERIOD_S = 0.1

#: A window is discarded when consecutive accepted ticks are further apart than
#: this. One and a half grid periods tolerates jitter but not a dropped tick.
DEFAULT_MAX_GAP_S = 0.15

#: Visual token geometry, needed to shape the unused vision inputs for the
#: LiDAR-only variant whose forward signature still requires them.
VISUAL_TOKENS = 49
VISUAL_WIDTH = 384


#: Publishers rclpy creates for every node. They carry logging and parameter
#: events, never actuation, and cannot be suppressed, so the overlay's "publishes
#: nothing" claim is stated against exactly this set.
INFRASTRUCTURE_PUBLISHERS: frozenset[str] = frozenset({"/parameter_events", "/rosout"})


def assert_no_command_publishers(names: list[str]) -> None:
    """Refuse to run if any resolved publisher could reach a motor interface.

    The demonstration node is supposed to be motion-incapable by construction.
    This turns that intent into an assertion, so a later edit that adds a
    publisher fails loudly at startup instead of quietly on the robot.
    """

    offenders = sorted({name for name in names if is_forbidden_publisher(name)})
    if offenders:
        raise SafetyAuditError(
            "the live overlay must not publish command topics: " + ", ".join(offenders)
        )


def assert_publishers_are_inert(names: list[str]) -> None:
    """Refuse any publisher beyond the two rclpy creates on its own.

    Stronger than the command-pattern check alone: a new publisher with an
    innocent name still fails here, so the demonstration cannot quietly acquire
    an output channel between now and the robot being switched on.
    """

    assert_no_command_publishers(names)
    unexpected = sorted(set(names) - INFRASTRUCTURE_PUBLISHERS)
    if unexpected:
        raise SafetyAuditError(
            "the live overlay may only hold rclpy infrastructure publishers, found: "
            + ", ".join(unexpected)
        )


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    """Tokenization and window geometry, taken from the training sweep config."""

    context_k: int = 8
    horizon_h: int = 8
    lidar_sectors: int = 80
    lidar_range_clip_m: float = 10.0
    visual_mask_radius_tokens: int = 1
    max_gap_s: float = DEFAULT_MAX_GAP_S

    @classmethod
    def from_sweep_config(
        cls, config: dict[str, Any], *, max_gap_s: float = DEFAULT_MAX_GAP_S
    ) -> OverlayConfig:
        return cls(
            context_k=int(config["context_k"]),
            horizon_h=int(config["horizon_h"]),
            lidar_sectors=int(config["lidar_sectors"]),
            lidar_range_clip_m=float(config["lidar_range_clip_m"]),
            visual_mask_radius_tokens=int(config["visual_mask_radius_tokens"]),
            max_gap_s=max_gap_s,
        )


def calibration_context(manifest_path: str | Path) -> dict[str, Any]:
    """Read the calibration block the live tokenizer must reuse.

    `tokenize_lidar` reads only `manifest["calibration"]`, so the live path
    borrows the calibration of the export the policy was trained on rather than
    re-deriving one. Returning the run identity alongside it keeps the
    demonstration able to state which export's geometry it is standing on.
    """

    manifest = json.loads(Path(manifest_path).read_text("utf-8"))
    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError(f"{manifest_path} has no calibration block")
    for required in ("lidar_geometry", "recorded_camera_info", "static_transforms"):
        if required not in calibration:
            raise ValueError(f"{manifest_path} calibration lacks {required}")
    return {
        "calibration": calibration,
        "run_id": manifest.get("run_id"),
        "source_tree_sha256": manifest.get("code", {}).get("source_tree_sha256"),
    }


def tokenize_live_scan(
    ranges: np.ndarray,
    angle_increment_rad: float,
    context: dict[str, Any],
    config: OverlayConfig,
) -> LidarTokens:
    """Tokenize one live scan exactly as the exporter's rows were tokenized."""

    values = np.asarray(ranges, dtype=np.float64)
    beam_count = int(values.shape[0])
    if beam_count < 1:
        raise ValueError("a live scan must carry at least one beam")
    return tokenize_lidar(
        values,
        beam_count,
        float(angle_increment_rad),
        context,
        sectors=config.lidar_sectors,
        range_clip_m=config.lidar_range_clip_m,
        visual_radius=config.visual_mask_radius_tokens,
    )


@dataclass
class LiveWindow:
    """Rolling K-step policy input window that refuses to bridge a gap."""

    config: OverlayConfig
    _features: list[np.ndarray] = field(default_factory=list, init=False)
    _visual_mask: list[np.ndarray] = field(default_factory=list, init=False)
    _in_fov: list[np.ndarray] = field(default_factory=list, init=False)
    _goal: list[np.ndarray] = field(default_factory=list, init=False)
    _state: list[np.ndarray] = field(default_factory=list, init=False)
    _last_timestamp_s: float | None = field(default=None, init=False)
    resets: int = field(default=0, init=False)

    def reset(self) -> None:
        self._features.clear()
        self._visual_mask.clear()
        self._in_fov.clear()
        self._goal.clear()
        self._state.clear()
        self._last_timestamp_s = None

    @property
    def depth(self) -> int:
        return len(self._features)

    @property
    def ready(self) -> bool:
        return self.depth >= self.config.context_k

    def push(
        self,
        *,
        timestamp_s: float,
        tokens: LidarTokens,
        goal: tuple[float, float, float],
        robot_state: tuple[float, float],
    ) -> bool:
        """Append one tick, resetting first if the previous tick is too old.

        Returns whether the window is ready to be forwarded.
        """

        if not math.isfinite(timestamp_s):
            raise ValueError("tick timestamps must be finite")
        if self._last_timestamp_s is not None:
            gap = timestamp_s - self._last_timestamp_s
            if gap <= 0.0 or gap > self.config.max_gap_s:
                self.reset()
                self.resets += 1
        self._features.append(np.asarray(tokens.features, dtype=np.float32))
        self._visual_mask.append(np.asarray(tokens.visual_mask, dtype=bool))
        self._in_fov.append(np.asarray(tokens.in_fov, dtype=bool))
        self._goal.append(np.asarray(goal, dtype=np.float32))
        self._state.append(np.asarray(robot_state, dtype=np.float32))
        self._last_timestamp_s = timestamp_s
        limit = self.config.context_k
        if len(self._features) > limit:
            del self._features[:-limit]
            del self._visual_mask[:-limit]
            del self._in_fov[:-limit]
            del self._goal[:-limit]
            del self._state[:-limit]
        return self.ready

    def arrays(self) -> dict[str, np.ndarray]:
        """Batch-of-one model inputs in the sweep's key order and dtypes."""

        if not self.ready:
            raise ValueError("the window is not full yet")
        k = self.config.context_k
        return {
            "visual_tokens": np.zeros(
                (1, k, VISUAL_TOKENS, VISUAL_WIDTH), dtype=np.float32
            ),
            "lidar_features": np.stack(self._features)[None],
            "visual_mask": np.stack(self._visual_mask)[None],
            "in_fov": np.stack(self._in_fov)[None],
            "goal": np.stack(self._goal)[None],
            "robot_state": np.stack(self._state)[None],
        }


def sigma_from_log_variance(log_variance: np.ndarray) -> np.ndarray:
    """Per-step predictive standard deviation of the heteroscedastic head."""

    return np.exp(0.5 * np.asarray(log_variance, dtype=np.float64))


def rollout_unicycle(
    commands: np.ndarray, *, dt: float = TICK_PERIOD_S
) -> np.ndarray:
    """Integrate a [steps, 2] command sequence into [steps + 1, 3] poses.

    The pose sequence starts at the robot origin so the drawn path is always in
    the robot's own frame, which is the frame the scan is drawn in.
    """

    values = np.asarray(commands, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("commands must have shape [steps, 2]")
    poses = np.zeros((values.shape[0] + 1, 3), dtype=np.float64)
    x = y = theta = 0.0
    for index, (linear, angular) in enumerate(values):
        x += float(linear) * math.cos(theta) * dt
        y += float(linear) * math.sin(theta) * dt
        theta += float(angular) * dt
        poses[index + 1] = (x, y, theta)
    return poses


def goal_xy(goal: tuple[float, float, float]) -> tuple[float, float]:
    """Convert the exported [rho, sin(alpha), cos(alpha)] goal to robot-frame xy."""

    rho, sin_alpha, cos_alpha = (float(value) for value in goal)
    norm = math.hypot(sin_alpha, cos_alpha)
    if norm == 0.0:
        raise ValueError("goal bearing is degenerate")
    return rho * cos_alpha / norm, rho * sin_alpha / norm


def scan_points(
    ranges: np.ndarray,
    angle_increment_rad: float,
    context: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Robot-frame xy of every valid return in one live scan.

    Bearings use *this* scan's increment. The scanner varies its beam count and
    covaries the increment, so a cached global bearing table would be wrong by
    several degrees at the far beam.
    """

    geometry = context["calibration"]["lidar_geometry"]["angular_frame"]
    angle_min = float(geometry["angle_min_rad"])
    range_min = float(geometry["range_min_m"])
    range_max = float(geometry["range_max_m"])
    values = np.asarray(ranges, dtype=np.float64)
    theta = angle_min + np.arange(values.shape[0], dtype=np.float64) * float(
        angle_increment_rad
    )
    valid = np.isfinite(values) & (values >= range_min) & (values <= range_max)
    return values[valid] * np.cos(theta[valid]), values[valid] * np.sin(theta[valid])


def agreement_error(
    predicted: np.ndarray, commanded: tuple[float, float]
) -> dict[str, float]:
    """First-step disagreement between the policy and the operator.

    Only the first horizon step is comparable to the command in force right now;
    the remaining steps are predictions about a future the operator has not
    reached yet, and scoring those against the current command would be wrong.
    """

    values = np.asarray(predicted, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("predicted commands must have shape [steps, 2]")
    linear, angular = (float(value) for value in commanded)
    return {
        "linear_error_mps": float(values[0, 0]) - linear,
        "angular_error_radps": float(values[0, 1]) - angular,
    }
