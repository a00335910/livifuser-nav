"""Unit tests for subset selection, the LR schedule, and gate scoring.

These decide what the overfit gate trains on and whether it passed, so they are
tested here rather than trusted inside the runner script.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from livifuser_nav.learning_data import WindowDataset
from livifuser_nav.training import (
    ACTION_CLASSES,
    action_class,
    evaluate_acceptance,
    per_window_errors,
    phase_learning_rate,
    select_tiny_windows,
)
from tests.test_learning_data import write_export

CRITERIA = {
    "min_normalized_mse_reduction_factor": 20.0,
    "max_mean_phase_normalized_mse": 0.01,
    "max_window_abs_linear_mps": 0.025,
    "max_window_abs_angular_radps": 0.125,
    "max_uncertainty_phase_normalized_mse": 0.05,
}


def selection_record(index: int) -> dict:
    return {
        "run_id": "run",
        "origin_row": index,
        "first_action_class": "zero",
    }


class ActionClassTests(unittest.TestCase):
    def test_recorded_command_alphabet_maps_to_the_four_classes(self) -> None:
        # `_05` and `_06` only ever recorded these six commands.
        self.assertEqual(action_class(np.asarray([0.0, 0.0])), "zero")
        self.assertEqual(action_class(np.asarray([0.08, 0.0])), "straight")
        self.assertEqual(action_class(np.asarray([0.0, 0.4])), "left")
        self.assertEqual(action_class(np.asarray([0.0, -0.4])), "right")
        self.assertEqual(action_class(np.asarray([0.08, 0.4])), "left")
        self.assertEqual(action_class(np.asarray([0.08, -0.4])), "right")

    def test_turning_dominates_the_class_even_while_driving_forward(self) -> None:
        self.assertEqual(action_class(np.asarray([0.08, 0.4])), "left")

    def test_float_noise_below_tolerance_is_still_zero(self) -> None:
        self.assertEqual(action_class(np.asarray([1e-9, -1e-9])), "zero")

    def test_every_class_name_is_declared(self) -> None:
        produced = {
            action_class(np.asarray(value))
            for value in ([0.0, 0.0], [0.08, 0.0], [0.0, 0.4], [0.0, -0.4])
        }
        self.assertEqual(produced, set(ACTION_CLASSES))


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def build(self, name: str, rows: int, run_id: str) -> WindowDataset:
        path = write_export(self.root / name, segment_lengths=[rows], run_id=run_id)
        # Overwrite the ramp actions with a repeating four-class command cycle.
        loaded = dict(np.load(path / "vectors.npz"))
        cycle = np.asarray(
            [[0.0, 0.0], [0.08, 0.0], [0.0, 0.4], [0.0, -0.4]], dtype=np.float32
        )
        loaded["action"] = np.tile(cycle, (rows // 4 + 1, 1))[:rows]
        np.savez(path / "vectors.npz", **loaded)
        return path

    def test_selection_is_balanced_across_classes_and_runs(self) -> None:
        first = self.build("a", 60, "run_a")
        second = self.build("b", 60, "run_b")
        dataset = WindowDataset([first, second], context_k=8, horizon_h=8)
        selected = select_tiny_windows(dataset, 2)
        self.assertEqual(len(selected), 16)
        counts: dict[tuple[int, str], int] = {}
        for ref in selected:
            key = (ref.run_index, action_class(dataset.targets(ref)[0]))
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(
            counts, {(run, name): 2 for run in (0, 1) for name in ACTION_CLASSES}
        )

    def test_selection_is_deterministic(self) -> None:
        path = self.build("c", 60, "run_c")
        dataset = WindowDataset([path], context_k=8, horizon_h=8)
        self.assertEqual(select_tiny_windows(dataset, 2), select_tiny_windows(dataset, 2))

    def test_selection_spans_the_run_rather_than_clustering_at_the_start(self) -> None:
        path = self.build("d", 200, "run_d")
        dataset = WindowDataset([path], context_k=8, horizon_h=8)
        origins = sorted(ref.origin_row for ref in select_tiny_windows(dataset, 2))
        self.assertLess(origins[0], 20)
        self.assertGreater(origins[-1], 150)

    def test_a_class_with_too_few_windows_fails_loudly(self) -> None:
        path = write_export(self.root / "e", segment_lengths=[40], run_id="run_e")
        loaded = dict(np.load(path / "vectors.npz"))
        loaded["action"] = np.zeros((40, 2), dtype=np.float32)  # every window is zero
        np.savez(path / "vectors.npz", **loaded)
        dataset = WindowDataset([path], context_k=8, horizon_h=8)
        with self.assertRaisesRegex(ValueError, "has only 0 straight windows"):
            select_tiny_windows(dataset, 2)

    def test_non_positive_request_is_refused(self) -> None:
        path = self.build("f", 60, "run_f")
        dataset = WindowDataset([path], context_k=8, horizon_h=8)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            select_tiny_windows(dataset, 0)


class LearningRateScheduleTests(unittest.TestCase):
    def test_constant_schedule_reproduces_the_first_run(self) -> None:
        config = {}
        for step in (0, 37, 149):
            self.assertEqual(phase_learning_rate(step, 0, 150, 0.002, config), 0.002)

    def test_cosine_starts_at_the_peak_and_ends_at_the_floor(self) -> None:
        config = {"lr_schedule": "cosine_per_phase", "lr_floor_fraction": 0.05}
        self.assertAlmostEqual(phase_learning_rate(0, 0, 600, 0.002, config), 0.002)
        self.assertAlmostEqual(phase_learning_rate(599, 0, 600, 0.002, config), 0.0001)

    def test_cosine_decreases_monotonically_within_a_phase(self) -> None:
        config = {"lr_schedule": "cosine_per_phase", "lr_floor_fraction": 0.05}
        rates = [phase_learning_rate(step, 0, 600, 0.002, config) for step in range(600)]
        self.assertTrue(all(b < a for a, b in zip(rates, rates[1:], strict=False)))

    def test_the_second_phase_is_scheduled_from_its_own_start(self) -> None:
        config = {"lr_schedule": "cosine_per_phase", "lr_floor_fraction": 0.05}
        self.assertAlmostEqual(phase_learning_rate(600, 600, 200, 0.0005, config), 0.0005)
        self.assertAlmostEqual(phase_learning_rate(799, 600, 200, 0.0005, config), 0.000025)

    def test_midpoint_is_the_cosine_midpoint(self) -> None:
        config = {"lr_schedule": "cosine_per_phase", "lr_floor_fraction": 0.0}
        rate = phase_learning_rate(50, 0, 101, 0.002, config)
        self.assertAlmostEqual(rate, 0.001, places=9)

    def test_a_single_step_phase_stays_at_the_peak(self) -> None:
        config = {"lr_schedule": "cosine_per_phase", "lr_floor_fraction": 0.05}
        self.assertAlmostEqual(phase_learning_rate(0, 0, 1, 0.002, config), 0.002)

    def test_unknown_schedule_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown lr_schedule"):
            phase_learning_rate(0, 0, 10, 0.002, {"lr_schedule": "linear"})

    def test_a_step_outside_its_phase_is_refused(self) -> None:
        config = {"lr_schedule": "cosine_per_phase", "lr_floor_fraction": 0.05}
        with self.assertRaisesRegex(ValueError, "outside its phase"):
            phase_learning_rate(700, 0, 600, 0.002, config)

    def test_the_schedule_never_reaches_zero(self) -> None:
        config = {"lr_schedule": "cosine_per_phase", "lr_floor_fraction": 0.05}
        rates = [phase_learning_rate(step, 0, 600, 0.002, config) for step in range(600)]
        self.assertGreater(min(rates), 0.0)
        self.assertTrue(all(math.isfinite(rate) for rate in rates))


class PerWindowErrorTests(unittest.TestCase):
    def test_worst_case_per_window_is_reported_in_physical_units(self) -> None:
        target = np.zeros((2, 8, 2))
        mean = np.zeros((2, 8, 2))
        mean[0, 3, 0] = 0.011
        mean[1, 5, 1] = -0.2
        errors = per_window_errors(mean, target, [selection_record(0), selection_record(1)])
        self.assertAlmostEqual(errors[0]["max_abs_linear_mps"], 0.011)
        self.assertAlmostEqual(errors[0]["max_abs_angular_radps"], 0.0)
        self.assertAlmostEqual(errors[1]["max_abs_angular_radps"], 0.2)

    def test_selection_metadata_is_carried_through(self) -> None:
        errors = per_window_errors(np.zeros((1, 8, 2)), np.zeros((1, 8, 2)), [selection_record(42)])
        self.assertEqual(errors[0]["origin_row"], 42)
        self.assertEqual(errors[0]["run_id"], "run")

    def test_mismatched_shapes_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            per_window_errors(np.zeros((1, 8, 2)), np.zeros((1, 4, 2)), [selection_record(0)])
        with self.assertRaisesRegex(ValueError, "one selection record"):
            per_window_errors(np.zeros((2, 8, 2)), np.zeros((2, 8, 2)), [selection_record(0)])


class AcceptanceTests(unittest.TestCase):
    def passing_windows(self) -> list[dict]:
        return [
            {**selection_record(0), "max_abs_linear_mps": 0.004,
             "max_abs_angular_radps": 0.02},
            {**selection_record(1), "max_abs_linear_mps": 0.010,
             "max_abs_angular_radps": 0.05},
        ]

    def score(self, **overrides) -> dict:
        arguments = {
            "initial": {"mse_normalized": 0.194},
            "mean_phase": {"mse_normalized": 0.002},
            "mean_phase_windows": self.passing_windows(),
            "uncertainty_phase": {"mse_normalized": 0.004},
        }
        arguments.update(overrides)
        return evaluate_acceptance(CRITERIA, **arguments)

    def failed_criteria(self, result: dict) -> list[str]:
        return [check["criterion"] for check in result["checks"] if not check["passed"]]

    def test_a_converged_run_passes_every_criterion(self) -> None:
        result = self.score()
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]), 5)

    def test_an_insufficient_reduction_fails_even_with_a_low_final_loss(self) -> None:
        result = self.score(
            initial={"mse_normalized": 0.005}, mean_phase={"mse_normalized": 0.004}
        )
        self.assertFalse(result["passed"])
        self.assertEqual(self.failed_criteria(result), ["normalized_mse_reduction_factor"])

    def test_one_badly_fit_window_fails_the_gate(self) -> None:
        # The first run of this gate averaged acceptably but left single windows
        # off by 0.32 rad/s; the worst-case criterion is what catches that.
        windows = self.passing_windows()
        windows[1]["max_abs_angular_radps"] = 0.32
        result = self.score(mean_phase_windows=windows)
        self.assertFalse(result["passed"])
        self.assertEqual(
            self.failed_criteria(result), ["mean_phase_worst_window_abs_angular_radps"]
        )

    def test_the_uncertainty_phase_may_not_destroy_the_mean_fit(self) -> None:
        # The heteroscedastic phase is allowed to trade mean accuracy for
        # variance, but only up to the declared bound.
        self.assertTrue(self.score(uncertainty_phase={"mse_normalized": 0.049})["passed"])
        result = self.score(uncertainty_phase={"mse_normalized": 0.06})
        self.assertFalse(result["passed"])
        self.assertEqual(
            self.failed_criteria(result), ["uncertainty_phase_final_normalized_mse"]
        )

    def test_the_mean_phase_is_scored_rather_than_the_final_state(self) -> None:
        # A run whose mean phase converged still passes when the later phase
        # relaxes the mean within bounds; scoring only the final state would
        # conflate the two objectives.
        result = self.score(
            mean_phase={"mse_normalized": 0.0001},
            uncertainty_phase={"mse_normalized": 0.03},
        )
        self.assertTrue(result["passed"])
        observed = {check["criterion"]: check["observed"] for check in result["checks"]}
        self.assertAlmostEqual(observed["mean_phase_final_normalized_mse"], 0.0001)
        self.assertAlmostEqual(observed["uncertainty_phase_final_normalized_mse"], 0.03)

    def test_a_criterion_exactly_at_its_threshold_passes(self) -> None:
        windows = self.passing_windows()
        windows[0]["max_abs_linear_mps"] = CRITERIA["max_window_abs_linear_mps"]
        result = self.score(
            initial={"mse_normalized": 0.2},
            mean_phase={"mse_normalized": 0.01},
            mean_phase_windows=windows,
        )
        self.assertTrue(result["passed"])

    def test_the_recorded_scope_disclaims_generalization(self) -> None:
        self.assertIn("nothing about generalization", self.score()["scope"])

    def test_a_zero_mean_phase_loss_does_not_divide_by_zero(self) -> None:
        result = self.score(mean_phase={"mse_normalized": 0.0})
        self.assertTrue(result["passed"])
        self.assertTrue(math.isfinite(result["checks"][0]["observed"]))

    def test_scoring_no_windows_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one scored window"):
            self.score(mean_phase_windows=[])

    def test_a_missing_criterion_fails_loudly(self) -> None:
        incomplete = {key: value for key, value in CRITERIA.items() if "window" not in key}
        with self.assertRaises(KeyError):
            evaluate_acceptance(
                incomplete,
                initial={"mse_normalized": 0.194},
                mean_phase={"mse_normalized": 0.002},
                mean_phase_windows=self.passing_windows(),
                uncertainty_phase={"mse_normalized": 0.004},
            )


if __name__ == "__main__":
    unittest.main()
