from __future__ import annotations

import unittest

import numpy as np

from livifuser_nav.heldout_evaluation import (
    discrimination_metrics,
    episode_reduce,
    episode_risk_coverage,
    exact_two_sided_sign_test,
    grouped_average_precision,
    hierarchical_paired_bootstrap,
    macro_sigma_calibration,
    midrank_auroc,
    right_continuous_cdf,
)
from scripts.run_sim_heldout_evaluation_kaggle import (
    GROUPS,
    group_key,
    uncertainty_metrics,
)


class FrozenHeldoutStatisticsTests(unittest.TestCase):
    def test_right_continuous_cdf_counts_ties_on_the_right(self) -> None:
        reference = np.asarray([1.0, 2.0, 2.0, 4.0])
        observed = right_continuous_cdf(reference, np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_array_equal(observed, np.asarray([0.0, 0.25, 0.75, 0.75, 1.0]))

    def test_episode_reduction_is_sorted_and_explicit(self) -> None:
        identities, values = episode_reduce(
            ["b", "a", "b", "a"],
            np.asarray([1.0, 3.0, 5.0, 7.0]),
            "mean",
        )
        np.testing.assert_array_equal(identities, np.asarray(["a", "b"]))
        np.testing.assert_array_equal(values, np.asarray([5.0, 3.0]))
        _, maxima = episode_reduce(
            ["b", "a", "b", "a"],
            np.asarray([1.0, 3.0, 5.0, 7.0]),
            "max",
        )
        np.testing.assert_array_equal(maxima, np.asarray([7.0, 5.0]))

    def test_rank_metrics_handle_ties_and_grouped_thresholds(self) -> None:
        positive = np.asarray([1.0, 2.0])
        negative = np.asarray([1.0, 0.0])
        self.assertEqual(midrank_auroc(positive, negative), 0.875)
        self.assertAlmostEqual(grouped_average_precision(positive, negative), 5.0 / 6.0)
        metrics = discrimination_metrics(positive, negative)
        self.assertEqual(metrics["positive_episodes"], 2)
        self.assertEqual(metrics["negative_episodes"], 2)
        self.assertEqual(metrics["fpr_at_95_recall"], 0.5)

    def test_risk_coverage_uses_episode_id_to_break_score_ties(self) -> None:
        curve = episode_risk_coverage(
            np.asarray(["b", "a", "c"]),
            np.asarray([20.0, 10.0, 30.0]),
            np.asarray([0.1, 0.1, 0.9]),
        )
        self.assertEqual(len(curve), 19)
        self.assertEqual(curve[0]["retained_episode_count"], 3)
        self.assertEqual(curve[0]["risk"], 20.0)
        self.assertEqual(curve[-1]["retained_episode_ids"], ["a"])
        self.assertEqual(curve[-1]["risk"], 10.0)

    def test_calibration_is_reduced_within_episode_then_macro_averaged(self) -> None:
        mean = np.zeros((3, 1, 2), dtype=np.float64)
        target = np.zeros_like(mean)
        target[2] = 10.0
        result = macro_sigma_calibration(
            ["episode_a", "episode_a", "episode_b"],
            mean,
            np.zeros_like(mean),
            target,
        )
        self.assertEqual(result["episode_count"], 2)
        self.assertEqual(result["linear"], [0.5, 0.5, 0.5])
        self.assertEqual(result["angular"], [0.5, 0.5, 0.5])
        expected = np.mean(
            np.abs(
                np.asarray([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
                - np.asarray([0.6827, 0.9545, 0.9973])
            )
        )
        self.assertAlmostEqual(result["six_point_calibration_mae"], expected)

    def test_hierarchical_bootstrap_and_sign_test_are_frozen(self) -> None:
        worlds = ["world_a", "world_a", "world_b", "world_b"]
        identities = ["a0", "a1", "b0", "b1"]
        differences = np.asarray([1.0, 3.0, -2.0, -4.0])
        first = hierarchical_paired_bootstrap(worlds, identities, differences)
        second = hierarchical_paired_bootstrap(worlds, identities, differences)
        self.assertEqual(first, second)
        self.assertEqual(first["replicates"], 10_000)
        self.assertEqual(first["rng_seed"], 20_260_824)
        self.assertEqual(first["point_estimate"], -0.5)
        self.assertEqual(first["world_mean_differences"], [2.0, -3.0])
        self.assertEqual(first["sign_test"]["denominator"], 2)
        self.assertEqual(first["sign_test"]["two_sided_p_value"], 1.0)
        all_positive = exact_two_sided_sign_test(np.asarray([1.0, 2.0]))
        self.assertEqual(all_positive["two_sided_p_value"], 0.5)

    def test_uncertainty_summary_uses_matched_ood_c0_episodes(self) -> None:
        records = []
        for world in ("world_a", "world_b"):
            for index in range(15):
                records.append(
                    (
                        f"test_id_{world}_{index:02d}",
                        "test_id",
                        world,
                        "C0",
                        index,
                        1000 + index,
                    )
                )
        for condition in ("C0", "C1", "C3b", "C4"):
            for world_index, world in enumerate(("world_a", "world_b")):
                for index in range(10):
                    records.append(
                        (
                            f"test_ood_{world}_{condition}_{index:02d}",
                            "test_ood",
                            world,
                            condition,
                            index,
                            2000 + world_index * 100 + index,
                        )
                    )
        common = {
            "episode_ids": np.asarray([record[0] for record in records], dtype=np.str_),
            "splits": np.asarray([record[1] for record in records], dtype=np.str_),
            "worlds": np.asarray([record[2] for record in records], dtype=np.str_),
            "conditions": np.asarray([record[3] for record in records], dtype=np.str_),
            "episode_indices": np.asarray([record[4] for record in records], dtype=np.int64),
            "observation_seeds": np.asarray([record[5] for record in records], dtype=np.int64),
        }
        z_a = np.asarray(
            [0.9 if record[3] in {"C1", "C3b"} else 0.1 for record in records],
            dtype=np.float64,
        )
        z_m = np.asarray(
            [0.9 if record[3] == "C1" else 0.1 for record in records],
            dtype=np.float64,
        )
        arrays = {
            "z_a": z_a,
            "z_m": z_m,
            "combined": np.maximum(z_a, z_m),
        }
        prediction = {}
        for split, condition in GROUPS:
            selected = (common["splits"] == split) & (common["conditions"] == condition)
            prediction[group_key(split, condition)] = {
                "per_episode_normalized_mse": {
                    identity: float(index + 1)
                    for index, identity in enumerate(
                        sorted(common["episode_ids"][selected].tolist())
                    )
                }
            }
        result = uncertainty_metrics(
            common,
            arrays,
            prediction,
            {"aleatoric": 0.95, "mahalanobis": 0.95, "combined": 1.0},
        )
        c1 = result["condition_discrimination"]["C1:aleatoric"]
        self.assertEqual(c1["positive_episodes"], 20)
        self.assertEqual(c1["negative_episodes"], 20)
        self.assertEqual(c1["auroc"], 1.0)
        c3_mahalanobis = result["condition_discrimination"]["C3b:mahalanobis"]
        self.assertEqual(c3_mahalanobis["auroc"], 0.5)
        self.assertTrue(
            all(
                record["intervention_count"] == 0
                for key, record in result["interventions"].items()
                if key.endswith(":combined")
            )
        )


if __name__ == "__main__":
    unittest.main()
