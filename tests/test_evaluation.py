"""Unit tests for held-out evaluation and Mahalanobis OOD scoring.

NumPy-only except the final cross-check class, which pins the NumPy metric
formulas to the torch training losses where torch is available.
"""

from __future__ import annotations

import unittest

import numpy as np

from livifuser_nav.evaluation import (
    ACTION_SCALE,
    LOG_VARIANCE_CLAMP,
    auroc,
    fit_gaussian,
    mahalanobis_distances,
    per_episode_summary,
    per_horizon_normalized_mse,
    risk_coverage,
    sigma_coverage,
    window_nll,
    window_normalized_mse,
)

try:  # pragma: no cover - availability is what is being guarded
    import torch

    from livifuser_nav.model import ACTION_SCALE as MODEL_ACTION_SCALE
    from livifuser_nav.model import heteroscedastic_nll, mean_warmup_loss

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


def example_windows(seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    shape = (6, 8, 2)
    scale = np.asarray(ACTION_SCALE)
    target = generator.uniform(-1.0, 1.0, shape) * scale
    mean = target + generator.normal(0.0, 0.02, shape) * scale
    log_variance = generator.uniform(-6.0, 3.0, shape)
    return mean, log_variance, target


class WindowMetricTests(unittest.TestCase):
    def test_window_nll_matches_the_manual_formula(self) -> None:
        mean, log_variance, target = example_windows()
        error = (mean - target) / np.asarray(ACTION_SCALE)
        clamped = np.clip(log_variance, *LOG_VARIANCE_CLAMP)
        expected = (0.5 * (np.exp(-clamped) * error**2 + clamped)).mean(axis=(1, 2))
        np.testing.assert_allclose(window_nll(mean, log_variance, target), expected)

    def test_log_variance_is_clamped_before_use(self) -> None:
        mean, _, target = example_windows()
        saturated = np.full(mean.shape, -50.0)
        clamped_result = window_nll(mean, saturated, target)
        expected = window_nll(mean, np.full(mean.shape, LOG_VARIANCE_CLAMP[0]), target)
        np.testing.assert_allclose(clamped_result, expected)

    def test_perfect_prediction_has_zero_mse(self) -> None:
        _, _, target = example_windows()
        np.testing.assert_allclose(window_normalized_mse(target, target), 0.0)

    def test_per_horizon_mse_reports_one_value_per_step(self) -> None:
        mean, _, target = example_windows()
        values = per_horizon_normalized_mse(mean, target)
        self.assertEqual(len(values), 8)
        self.assertAlmostEqual(
            float(np.mean(values)), float(window_normalized_mse(mean, target).mean())
        )

    def test_shape_mismatch_is_refused(self) -> None:
        mean, log_variance, target = example_windows()
        with self.assertRaisesRegex(ValueError, "shapes"):
            window_nll(mean[:2], log_variance, target)


class EpisodeSummaryTests(unittest.TestCase):
    def test_episodes_are_aggregated_separately(self) -> None:
        values = np.asarray([1.0, 3.0, 10.0])
        summary = per_episode_summary(["a", "a", "b"], values)
        self.assertEqual(summary["a"]["window_count"], 2)
        self.assertEqual(summary["a"]["mean"], 2.0)
        self.assertEqual(summary["b"]["mean"], 10.0)

    def test_id_count_mismatch_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "episode id"):
            per_episode_summary(["a"], np.asarray([1.0, 2.0]))


class SigmaCoverageTests(unittest.TestCase):
    def test_huge_sigma_covers_everything(self) -> None:
        mean, _, target = example_windows()
        wide = np.full(mean.shape, 2.0)
        coverage = sigma_coverage(mean, wide, target)
        self.assertEqual(coverage["linear"]["1_sigma"], 1.0)
        self.assertEqual(coverage["angular"]["3_sigma"], 1.0)

    def test_clamp_floor_fraction_detects_saturation(self) -> None:
        mean, _, target = example_windows()
        saturated = np.full(mean.shape, LOG_VARIANCE_CLAMP[0] - 1.0)
        coverage = sigma_coverage(mean, saturated, target)
        self.assertEqual(coverage["clamp_floor_fraction"], 1.0)


class MahalanobisTests(unittest.TestCase):
    def test_identity_covariance_gives_euclidean_distance(self) -> None:
        generator = np.random.default_rng(1)
        features = generator.normal(0.0, 1.0, (5000, 4))
        mean, precision = fit_gaussian(features, shrinkage=0.0)
        query = mean + np.asarray([[2.0, 0.0, 0.0, 0.0]])
        distance = mahalanobis_distances(query, mean, precision)[0]
        self.assertAlmostEqual(distance, 2.0, delta=0.15)

    def test_full_shrinkage_is_the_scaled_identity(self) -> None:
        generator = np.random.default_rng(2)
        features = generator.normal(0.0, 1.0, (100, 3)) @ np.diag([1.0, 5.0, 0.2])
        _, precision = fit_gaussian(features, shrinkage=1.0)
        off_diagonal = precision - np.diag(np.diag(precision))
        np.testing.assert_allclose(off_diagonal, 0.0, atol=1e-12)
        diagonal = np.diag(precision)
        np.testing.assert_allclose(diagonal, diagonal[0])

    def test_far_points_score_higher(self) -> None:
        generator = np.random.default_rng(3)
        features = generator.normal(0.0, 1.0, (500, 8))
        mean, precision = fit_gaussian(features, shrinkage=0.1)
        near = mahalanobis_distances(mean[None], mean, precision)[0]
        far = mahalanobis_distances(mean[None] + 10.0, mean, precision)[0]
        self.assertLess(near, far)

    def test_invalid_shrinkage_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "shrinkage"):
            fit_gaussian(np.zeros((10, 2)), shrinkage=1.5)


class AurocTests(unittest.TestCase):
    def test_perfect_separation_is_one(self) -> None:
        self.assertEqual(auroc(np.asarray([5.0, 6.0]), np.asarray([1.0, 2.0])), 1.0)

    def test_reversed_separation_is_zero(self) -> None:
        self.assertEqual(auroc(np.asarray([1.0, 2.0]), np.asarray([5.0, 6.0])), 0.0)

    def test_identical_scores_are_chance(self) -> None:
        self.assertAlmostEqual(
            auroc(np.asarray([1.0, 1.0]), np.asarray([1.0, 1.0])), 0.5
        )

    def test_empty_scores_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            auroc(np.asarray([]), np.asarray([1.0]))


class RiskCoverageTests(unittest.TestCase):
    def test_informative_signal_reduces_risk_at_low_coverage(self) -> None:
        errors = np.asarray([0.1, 0.2, 0.9, 1.0])
        distances = np.asarray([1.0, 2.0, 9.0, 10.0])
        curve = risk_coverage(errors, distances)
        self.assertEqual(curve[0]["coverage"], 0.25)
        self.assertAlmostEqual(curve[0]["risk"], 0.1)
        self.assertAlmostEqual(curve[-1]["coverage"], 1.0)
        self.assertAlmostEqual(curve[-1]["risk"], float(errors.mean()))
        self.assertLess(curve[0]["risk"], curve[-1]["risk"])

    def test_thresholds_are_ascending_distances(self) -> None:
        errors = np.asarray([0.5, 0.5, 0.5])
        distances = np.asarray([3.0, 1.0, 2.0])
        curve = risk_coverage(errors, distances)
        self.assertEqual(
            [point["distance_threshold"] for point in curve], [1.0, 2.0, 3.0]
        )

    def test_mismatched_arrays_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching"):
            risk_coverage(np.asarray([1.0]), np.asarray([1.0, 2.0]))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable on this host")
class TorchConsistencyTests(unittest.TestCase):
    """The NumPy metrics must agree with the torch training losses exactly."""

    def test_action_scale_constants_match(self) -> None:
        self.assertEqual(tuple(ACTION_SCALE), tuple(MODEL_ACTION_SCALE))

    def test_window_nll_matches_the_training_loss(self) -> None:
        mean, log_variance, target = example_windows()
        outputs = {
            "mean": torch.from_numpy(mean),
            "log_variance": torch.from_numpy(log_variance),
        }
        loss = float(heteroscedastic_nll(outputs, torch.from_numpy(target)))
        self.assertAlmostEqual(
            float(window_nll(mean, log_variance, target).mean()), loss, places=10
        )

    def test_window_mse_matches_the_warmup_loss(self) -> None:
        mean, _, target = example_windows()
        outputs = {"mean": torch.from_numpy(mean)}
        loss = float(mean_warmup_loss(outputs, torch.from_numpy(target)))
        self.assertAlmostEqual(
            float(window_normalized_mse(mean, target).mean()), loss, places=10
        )


if __name__ == "__main__":
    unittest.main()
