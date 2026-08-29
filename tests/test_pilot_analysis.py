"""Tests for episode-unit pilot aggregation helpers."""

from __future__ import annotations

import unittest

from livifuser_nav.pilot_analysis import (
    exact_bootstrap_mean_ci,
    metric_summary,
    paired_metric_summary,
    risk_at_coverage,
    spearman_correlation,
)


class ExactBootstrapTests(unittest.TestCase):
    def test_constant_values_have_point_interval(self) -> None:
        result = exact_bootstrap_mean_ci([2.0, 2.0, 2.0])
        self.assertEqual(result["resample_count"], 27)
        self.assertEqual(result["lower"], 2.0)
        self.assertEqual(result["upper"], 2.0)

    def test_rejects_too_many_exact_units(self) -> None:
        with self.assertRaisesRegex(ValueError, "limited to eight"):
            exact_bootstrap_mean_ci(list(range(9)))


class CorrelationTests(unittest.TestCase):
    def test_spearman_direction_and_ties(self) -> None:
        self.assertAlmostEqual(spearman_correlation([1, 2, 2, 4], [1, 2, 2, 4]), 1.0)
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [3, 2, 1]), -1.0)

    def test_constant_input_has_no_correlation(self) -> None:
        self.assertIsNone(spearman_correlation([1, 1, 1], [1, 2, 3]))


class MetricSummaryTests(unittest.TestCase):
    @staticmethod
    def records(offset: float = 0.0) -> list[dict[str, float | int | str]]:
        return [
            {"episode": episode, "seed": seed, "score": value + offset}
            for episode, first in (("a", 1.0), ("b", 3.0))
            for seed, value in ((1, first), (2, first + 2.0))
        ]

    def test_episodes_are_averaged_before_bootstrap(self) -> None:
        result = metric_summary(self.records(), "score")
        self.assertEqual(result["episode_means"], {"a": 2.0, "b": 4.0})
        self.assertEqual(result["seed_means"], {"1": 2.0, "2": 4.0})
        self.assertEqual(result["mean"], 3.0)

    def test_paired_direction_and_consistency(self) -> None:
        full = self.records()
        worse = self.records(offset=0.5)
        result = paired_metric_summary(full, worse, "score")
        self.assertEqual(result["mean_delta"], 0.5)
        self.assertEqual(result["full_wins"], 4)
        self.assertEqual(result["consistency"], "full_better_every_fold_and_seed")

    def test_incomplete_grid_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete episode-by-seed"):
            metric_summary(self.records()[:-1], "score")


class RiskCoverageTests(unittest.TestCase):
    def test_selects_first_point_reaching_target(self) -> None:
        curve = [
            {"coverage": 0.4, "risk": 1.0, "distance_threshold": 2.0},
            {"coverage": 0.6, "risk": 2.0, "distance_threshold": 3.0},
            {"coverage": 1.0, "risk": 4.0, "distance_threshold": 5.0},
        ]
        self.assertEqual(risk_at_coverage(curve, 0.5), curve[1])


if __name__ == "__main__":
    unittest.main()
