"""Subset selection, learning-rate schedule, and gate scoring for Stage 2.

These decide which data a run uses and whether it passed, so they live here and
are unit tested rather than sitting in the runner script. Nothing in this module
imports PyTorch: the runner converts tensors to arrays at the boundary, which
keeps the module importable on the ROS host alongside the rest of `src/`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from .learning_data import WindowDataset, WindowRef

ACTION_CLASSES = ("zero", "straight", "left", "right")

#: Commands below this magnitude are the recorded zero command, not motion.
ZERO_TOLERANCE = 1e-5


def action_class(action: np.ndarray) -> str:
    """Classify one recorded command for balanced subset selection."""

    linear, angular = (float(value) for value in action)
    if abs(linear) <= ZERO_TOLERANCE and abs(angular) <= ZERO_TOLERANCE:
        return "zero"
    if angular > ZERO_TOLERANCE:
        return "left"
    if angular < -ZERO_TOLERANCE:
        return "right"
    return "straight"


def select_tiny_windows(dataset: WindowDataset, per_class_per_run: int) -> list[WindowRef]:
    """Pick uniformly spaced windows per first-action class per run.

    Deterministic by construction: no shuffling and no RNG, so the same exports
    always yield the same subset. Selection is per run because the two pilot
    episodes are separate split units and must both be represented.
    """

    if per_class_per_run <= 0:
        raise ValueError("per_class_per_run must be positive")
    candidates: dict[tuple[int, str], list[WindowRef]] = defaultdict(list)
    for ref in dataset.windows:
        candidates[(ref.run_index, action_class(dataset.targets(ref)[0]))].append(ref)
    selected: list[WindowRef] = []
    for run_index in range(len(dataset.runs)):
        for category in ACTION_CLASSES:
            available = candidates[(run_index, category)]
            if len(available) < per_class_per_run:
                raise ValueError(
                    f"run {dataset.runs[run_index].run_id} has only "
                    f"{len(available)} {category} windows"
                )
            indices = np.linspace(0, len(available) - 1, per_class_per_run, dtype=int)
            selected.extend(available[int(index)] for index in indices)
    return selected


def phase_learning_rate(
    step: int, phase_start: int, phase_steps: int, peak: float, config: dict[str, Any]
) -> float:
    """Learning rate for `step`, decayed within its own phase.

    `constant` reproduces the first run of this gate, which held each phase at
    its peak rate and oscillated instead of settling. `cosine_per_phase` decays
    from the peak to `lr_floor_fraction` of it across the phase.
    """

    schedule = str(config.get("lr_schedule", "constant"))
    if schedule == "constant":
        return peak
    if schedule != "cosine_per_phase":
        raise ValueError(f"unknown lr_schedule {schedule!r}")
    if phase_steps <= 0:
        raise ValueError("phase_steps must be positive")
    floor = peak * float(config["lr_floor_fraction"])
    progress = (step - phase_start) / max(1, phase_steps - 1)
    if not 0.0 <= progress <= 1.0:
        raise ValueError(f"step {step} is outside its phase")
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def per_window_errors(
    mean: np.ndarray, target: np.ndarray, selections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Worst-case absolute error per selected window, in physical units."""

    if mean.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if mean.shape[0] != len(selections):
        raise ValueError("one selection record is required per predicted window")
    error = np.abs(np.asarray(mean, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    return [
        {
            "run_id": record["run_id"],
            "origin_row": record["origin_row"],
            "first_action_class": record["first_action_class"],
            "max_abs_linear_mps": float(error[index, :, 0].max()),
            "max_abs_angular_radps": float(error[index, :, 1].max()),
        }
        for index, record in enumerate(selections)
    ]


ACCEPTANCE_SCOPE = (
    "Memorization of a fixed tiny window subset. Passing shows the export, "
    "feature cache, geometry mask, fusion, and action head form a trainable "
    "path; it says nothing about generalization or policy efficacy."
)

#: The mean-fit criteria are scored at the end of the mean-MSE phase, because
#: the heteroscedastic phase that follows optimizes a different objective and is
#: entitled to trade mean accuracy for calibrated variance. The last criterion
#: bounds how much of that trade is acceptable.
ACCEPTANCE_CRITERIA = (
    ("normalized_mse_reduction_factor", "min_normalized_mse_reduction_factor", ">="),
    ("mean_phase_final_normalized_mse", "max_mean_phase_normalized_mse", "<="),
    ("mean_phase_worst_window_abs_linear_mps", "max_window_abs_linear_mps", "<="),
    ("mean_phase_worst_window_abs_angular_radps", "max_window_abs_angular_radps", "<="),
    ("uncertainty_phase_final_normalized_mse", "max_uncertainty_phase_normalized_mse", "<="),
)


def evaluate_acceptance(
    criteria: dict[str, Any],
    *,
    initial: dict[str, Any],
    mean_phase: dict[str, Any],
    mean_phase_windows: list[dict[str, Any]],
    uncertainty_phase: dict[str, Any],
) -> dict[str, Any]:
    """Score the preregistered gate criteria without softening any of them."""

    if not mean_phase_windows:
        raise ValueError("acceptance requires at least one scored window")
    observed_values = {
        "normalized_mse_reduction_factor": (
            initial["mse_normalized"] / max(mean_phase["mse_normalized"], 1e-12)
        ),
        "mean_phase_final_normalized_mse": mean_phase["mse_normalized"],
        "mean_phase_worst_window_abs_linear_mps": max(
            window["max_abs_linear_mps"] for window in mean_phase_windows
        ),
        "mean_phase_worst_window_abs_angular_radps": max(
            window["max_abs_angular_radps"] for window in mean_phase_windows
        ),
        "uncertainty_phase_final_normalized_mse": uncertainty_phase["mse_normalized"],
    }
    checks = []
    for name, key, comparison in ACCEPTANCE_CRITERIA:
        threshold = float(criteria[key])
        observed = float(observed_values[name])
        passed = observed >= threshold if comparison == ">=" else observed <= threshold
        checks.append(
            {
                "criterion": name,
                "comparison": comparison,
                "threshold": threshold,
                "observed": observed,
                "passed": bool(passed),
            }
        )
    return {
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "scope": ACCEPTANCE_SCOPE,
    }
