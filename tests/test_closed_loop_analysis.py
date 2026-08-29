"""The closed-loop analysis, validated on synthetic outcomes only.

Written before the confirmatory batch completed and before any real outcome was
seen. Every fixture here is fabricated, so the estimator and the resampling
scheme are pinned against known answers rather than against whatever the study
happens to produce.

The property that matters most: the bootstrap resamples worlds first and
episodes within them. An interval built by resampling episodes directly is far
too narrow, because episodes inside one world are not independent, and a narrow
interval is exactly the error that would manufacture a significant result.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from livifuser_nav.closed_loop_analysis import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    Episode,
    cluster_bootstrap,
    count_attempts,
    load_episodes,
    paired_contrast,
    rate_table,
    sign_test_two_worlds,
)
from livifuser_nav.confirmatory_plan import CONSTANT_ARM


def _episode(**overrides) -> Episode:
    values = {
        "arm": "full",
        "seed": 20260805,
        "ordinal": 180,
        "world": "w0",
        "condition": "C0",
        "terminal_reason": "success",
        "success": True,
        "collision": False,
        "uncertainty_intervention": False,
        "goal_distance_m": 0.2,
        "clearance_m": 0.4,
        "stretched_intervals": 0,
    }
    values.update(overrides)
    return Episode(**values)


class RateTableTests(unittest.TestCase):
    def test_rates_carry_exact_denominators(self) -> None:
        episodes = [
            _episode(ordinal=180, success=True),
            _episode(ordinal=181, success=False),
            _episode(ordinal=182, success=False),
        ]
        table = rate_table(episodes, "success")
        entry = table[("full", "C0", 20260805)]
        self.assertEqual(entry["numerator"], 1)
        self.assertEqual(entry["denominator"], 3)
        self.assertAlmostEqual(entry["rate"], 1 / 3)

    def test_seeds_are_reported_separately_never_pooled(self) -> None:
        episodes = [
            _episode(seed=20260805, success=True),
            _episode(seed=20260806, success=False, ordinal=181),
        ]
        table = rate_table(episodes, "success")
        self.assertIn(("full", "C0", 20260805), table)
        self.assertIn(("full", "C0", 20260806), table)
        self.assertEqual(table[("full", "C0", 20260805)]["denominator"], 1)

    def test_the_constant_arm_carries_no_seed_dimension(self) -> None:
        # It has no checkpoint, so a seed key would be fabricated.
        episodes = [_episode(arm=CONSTANT_ARM, seed=0, success=False)]
        table = rate_table(episodes, "success")
        self.assertIn((CONSTANT_ARM, "C0", None), table)


class BootstrapTests(unittest.TestCase):
    def test_the_interval_is_deterministic_under_the_frozen_seed(self) -> None:
        paired = {"w0": [0.2, 0.3, 0.1], "w1": [0.25, 0.15, 0.2]}
        first = cluster_bootstrap(paired, replicates=500, seed=BOOTSTRAP_SEED)
        second = cluster_bootstrap(paired, replicates=500, seed=BOOTSTRAP_SEED)
        self.assertEqual(first, second)

    def test_world_clustering_widens_the_interval_versus_pooling(self) -> None:
        # Two worlds that disagree sharply. Pooling every episode as if
        # independent hides the disagreement; clustering by world exposes it.
        clustered = {"w0": [1.0] * 20, "w1": [-1.0] * 20}
        pooled = {"pooled": [1.0] * 20 + [-1.0] * 20}
        wide = cluster_bootstrap(clustered, replicates=2000, seed=BOOTSTRAP_SEED)
        narrow = cluster_bootstrap(pooled, replicates=2000, seed=BOOTSTRAP_SEED)
        self.assertGreater(
            wide["ci_high"] - wide["ci_low"],
            narrow["ci_high"] - narrow["ci_low"],
        )

    def test_a_clear_effect_excludes_zero_and_a_null_does_not(self) -> None:
        effect = cluster_bootstrap(
            {"w0": [0.5] * 12, "w1": [0.45] * 12}, replicates=2000, seed=BOOTSTRAP_SEED
        )
        self.assertTrue(effect["excludes_zero"])
        null = cluster_bootstrap(
            {"w0": [0.5] * 12, "w1": [-0.5] * 12}, replicates=2000, seed=BOOTSTRAP_SEED
        )
        self.assertFalse(null["excludes_zero"])

    def test_the_frozen_constants_are_what_the_amendment_specifies(self) -> None:
        self.assertEqual(BOOTSTRAP_REPLICATES, 10_000)
        self.assertEqual(BOOTSTRAP_SEED, 20260824)


class SignTestTests(unittest.TestCase):
    def test_agreement_across_two_worlds_gives_the_exact_two_sided_value(self) -> None:
        result = sign_test_two_worlds({"w0": [0.3, 0.2], "w1": [0.4, 0.1]})
        self.assertTrue(result["agree"])
        self.assertAlmostEqual(result["p_value"], 0.5)

    def test_disagreement_is_not_significant(self) -> None:
        result = sign_test_two_worlds({"w0": [0.3], "w1": [-0.3]})
        self.assertFalse(result["agree"])
        self.assertEqual(result["p_value"], 1.0)

    def test_ties_are_excluded_rather_than_counted(self) -> None:
        result = sign_test_two_worlds({"w0": [0.0], "w1": [0.3]})
        self.assertEqual(result["ties_excluded"], 1)


class PairedContrastTests(unittest.TestCase):
    def test_pairing_is_by_world_ordinal_and_seed(self) -> None:
        episodes = [
            _episode(arm="full", ordinal=180, world="w0", goal_distance_m=1.0),
            _episode(arm="lidar_only", ordinal=180, world="w0", goal_distance_m=0.4),
            _episode(arm="full", ordinal=181, world="w1", goal_distance_m=2.0),
            _episode(arm="lidar_only", ordinal=181, world="w1", goal_distance_m=0.5),
        ]
        paired = paired_contrast(episodes, "goal_distance_m", "full", "lidar_only", "C0")
        self.assertEqual(sorted(paired), ["w0", "w1"])
        self.assertAlmostEqual(paired["w0"][0], 0.6)
        self.assertAlmostEqual(paired["w1"][0], 1.5)

    def test_an_unmatched_episode_is_dropped_not_paired_arbitrarily(self) -> None:
        episodes = [
            _episode(arm="full", ordinal=180, world="w0"),
            _episode(arm="lidar_only", ordinal=999, world="w0"),
        ]
        paired = paired_contrast(episodes, "goal_distance_m", "full", "lidar_only", "C0")
        self.assertEqual(paired, {})

    def test_a_learned_arm_pairs_against_the_seedless_constant_arm(self) -> None:
        episodes = [
            _episode(arm="full", ordinal=180, world="w0", goal_distance_m=1.0),
            _episode(arm=CONSTANT_ARM, seed=0, ordinal=180, world="w0", goal_distance_m=0.25),
        ]
        paired = paired_contrast(episodes, "goal_distance_m", "full", CONSTANT_ARM, "C0")
        self.assertAlmostEqual(paired["w0"][0], 0.75)


class LoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write(self, arm: str, seed: int, ordinal: int, reason: str, attempt: int = 1) -> None:
        directory = self.tmp / arm / str(seed) / f"{ordinal:04d}" / f"attempt_{attempt:03d}"
        directory.mkdir(parents=True)
        (directory / "terminal.json").write_text(
            json.dumps(
                {
                    "terminal": True,
                    "terminal_reason": reason,
                    "success": reason == "success",
                    "collision": reason == "collision",
                    "uncertainty_intervention": reason == "uncertainty_intervention",
                    "ground_truth_goal_distance_m": 1.0,
                    "ground_truth_clearance_m": 0.4,
                    "stretched_interval_count": 0,
                }
            )
        )

    def test_operational_outcomes_are_not_loaded_as_science(self) -> None:
        class Stub:
            key = "full/20260805/180"
            episode_id = "test_ood_straight_corridor_000_c0_e000_s74000000"
            condition = "C0"

        self._write("full", 20260805, 180, "operational_failure_control_interval")
        self.assertEqual(load_episodes(self.tmp, [Stub()]), [])

    def test_operational_counts_are_reported(self) -> None:
        directory = self.tmp / "full" / "20260805" / "0180" / "attempt_001"
        directory.mkdir(parents=True)
        (directory / "operational_failure.json").write_text("{}")
        counts = count_attempts(self.tmp)
        self.assertEqual(counts["attempts_total"], 1)
        self.assertEqual(counts["operational_failures"], 1)


if __name__ == "__main__":
    unittest.main()


class ZeroDecisionLoadingTests(unittest.TestCase):
    """The loader must refuse the same episodes the classifier refuses.

    The two filters were written separately and drifted: `classify_attempt`
    rejected zero-decision episodes while `load_episodes` accepted them, so an
    episode whose policy never ran would have been counted as a real failure to
    reach the goal. Both now apply the same rule.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        class Stub:
            key = "full/20260805/180"
            episode_id = "test_ood_straight_corridor_000_c0_e000_s74000000"
            condition = "C0"

        self.plan = [Stub()]

    def _write(self, context_sequence, reason="scientific_timeout") -> None:
        d = self.tmp / "full" / "20260805" / "0180" / "attempt_001"
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "terminal": True,
            "terminal_reason": reason,
            "success": False,
            "collision": False,
            "uncertainty_intervention": False,
            "ground_truth_goal_distance_m": 1.0,
            "ground_truth_clearance_m": 0.4,
            "stretched_interval_count": 0,
        }
        if context_sequence is not None:
            payload["context_sequence"] = context_sequence
        (d / "terminal.json").write_text(json.dumps(payload))

    def test_a_zero_decision_episode_is_not_loaded(self) -> None:
        self._write(0)
        self.assertEqual(load_episodes(self.tmp, self.plan), [])

    def test_a_missing_decision_count_is_not_loaded(self) -> None:
        self._write(None)
        self.assertEqual(load_episodes(self.tmp, self.plan), [])

    def test_a_real_episode_is_loaded(self) -> None:
        self._write(908)
        self.assertEqual(len(load_episodes(self.tmp, self.plan)), 1)

    def test_the_rule_matches_the_classifier(self) -> None:
        # Whatever classify_attempt calls operational must not reach analysis.
        from livifuser_nav.confirmatory_plan import classify_attempt

        self._write(0)
        attempt = self.tmp / "full" / "20260805" / "0180" / "attempt_001"
        self.assertEqual(classify_attempt(attempt), "operational")
        self.assertEqual(load_episodes(self.tmp, self.plan), [])
