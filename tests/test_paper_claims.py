"""The paper's headline numbers, checked against the sealed evidence.

A paper is the one artifact in this repository that nothing else validates. Its
numbers are typed into LaTeX by hand, so a corrected analysis, a re-seal, or a
late episode can leave the prose stating something the evidence no longer
supports --- silently, because LaTeX compiles either way.

These tests read the sealed batch and the frozen analysis output, then assert
that the exact strings the paper prints are the ones those artifacts contain. A
number that changes in the evidence and not in the paper fails here.

The evidence lives under `artifacts/`, which is gitignored, so every test skips
rather than fails when it is absent. That is the same guarded-skip pattern
`test_model.py` uses for PyTorch on the ROS host.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / "artifacts/closed_loop_batch_seal_v1.json"
RESULTS = ROOT / "artifacts/closed_loop_results_v1.json"
PAPER = ROOT / "paper/closed_loop_negative.tex"

#: What the seal must record. These are quoted in the abstract, in Section IV,
#: and in Section V, so a re-seal that changes them changes the paper.
SEALED_ATTEMPTS = 888
SEALED_SCIENTIFIC = 505
SEALED_OPERATIONAL = 378
SEALED_ABSENT = 5
SEALED_SCHEDULE_SHA256 = "85F48F81D52F82A5CD64D14D704E032617582D7F5462A7B6FD46C5E64899047D"

#: (contrast key, point, low, high). The three C0 contrasts adjudicate 14.1 and
#: the C3b one carries the claim that the result survives LiDAR corruption.
HEADLINE_CONTRASTS = (
    ("C0|full-lidar_only", -0.333, -0.500, -0.150),
    ("C0|full-rgb_only", +0.150, +0.000, +0.383),
    ("C0|full-concat", +0.121, -0.067, +0.362),
)


def _skip_unless(path: Path) -> None:
    if not path.is_file():
        raise unittest.SkipTest(f"evidence not present in this checkout: {path}")


class SealTests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless(SEAL)
        self.seal = json.loads(SEAL.read_text(encoding="utf-8"))

    def test_the_attempt_accounting_is_what_the_paper_reports(self) -> None:
        counts = self.seal["classification_counts"]
        self.assertEqual(self.seal["attempts_total"], SEALED_ATTEMPTS)
        self.assertEqual(counts["scientific"], SEALED_SCIENTIFIC)
        self.assertEqual(counts["operational"], SEALED_OPERATIONAL)
        self.assertEqual(counts["absent"], SEALED_ABSENT)

    def test_no_attempt_lies_outside_the_frozen_plan(self) -> None:
        # The paper states this explicitly; it is the claim that makes the
        # denominators meaningful.
        self.assertEqual(self.seal["attempts_outside_frozen_plan"], [])

    def test_the_schedule_identity_is_the_frozen_one(self) -> None:
        self.assertEqual(self.seal["schedule_sha256"], SEALED_SCHEDULE_SHA256)

    def test_the_seal_computes_no_outcome(self) -> None:
        # Sealing before analysing is the ordering the paper claims. A seal that
        # carried a rate would have been computed with outcomes in view.
        for forbidden in ("success_rates", "collision_rates", "contrasts"):
            self.assertNotIn(forbidden, self.seal)


class ResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless(RESULTS)
        self.results = json.loads(RESULTS.read_text(encoding="utf-8"))

    def test_the_analysed_episode_count_matches_the_seal(self) -> None:
        self.assertEqual(self.results["episodes"], SEALED_SCIENTIFIC)

    def test_the_c0_success_counts_are_what_table_one_prints(self) -> None:
        expected = {
            "full": (9, 60),
            "lidar_only": (29, 60),
            "concat": (1, 59),
            "rgb_only": (0, 60),
        }
        for arm, (numerator, denominator) in expected.items():
            cells = [
                value
                for key, value in self.results["success_rates"].items()
                if key.startswith(arm + "|C0|")
            ]
            self.assertEqual(sum(c["numerator"] for c in cells), numerator, arm)
            self.assertEqual(sum(c["denominator"] for c in cells), denominator, arm)

    def test_the_headline_contrasts_round_to_the_printed_values(self) -> None:
        for key, point, low, high in HEADLINE_CONTRASTS:
            contrast = self.results["contrasts"][key]
            self.assertAlmostEqual(contrast["point"], point, places=3, msg=key)
            self.assertAlmostEqual(contrast["ci_low"], low, places=3, msg=key)
            self.assertAlmostEqual(contrast["ci_high"], high, places=3, msg=key)

    def test_only_the_lidar_contrast_excludes_zero_on_c0(self) -> None:
        # This is the whole of criterion 14.1: fusion must exceed all three
        # baselines. It exceeds none, and the one interval that resolves does so
        # against fusion.
        self.assertTrue(self.results["contrasts"]["C0|full-lidar_only"]["excludes_zero"])
        self.assertLess(self.results["contrasts"]["C0|full-lidar_only"]["point"], 0.0)
        for key in ("C0|full-rgb_only", "C0|full-concat"):
            self.assertFalse(self.results["contrasts"][key]["excludes_zero"], key)

    def test_the_operational_record_matches_the_seal(self) -> None:
        operational = self.results["operational"]
        self.assertEqual(operational["attempts_total"], SEALED_ATTEMPTS)
        self.assertEqual(operational["teardown_leaks"], 0)


class UncertaintyGateTests(unittest.TestCase):
    """Zero interventions in 505 episodes, read from the episodes themselves.

    The reported analysis file carries success and collision rates only, so this
    claim cannot be checked against it. It is recomputed from the episode
    metadata, which is the artifact the claim actually rests on.
    """

    def setUp(self) -> None:
        evidence = ROOT / "artifacts/final_meta_x/evidence/confirmatory_closed_loop_v1"
        if not evidence.is_dir():
            raise unittest.SkipTest(f"episode metadata not extracted: {evidence}")
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from livifuser_nav.closed_loop_analysis import load_episodes, rate_table
        from livifuser_nav.confirmatory_plan import build_plan, locate_schedule

        self.episodes = load_episodes(evidence, build_plan(locate_schedule(ROOT)))
        self.rate_table = rate_table

    def test_the_gate_fired_on_no_episode(self) -> None:
        self.assertEqual(len(self.episodes), SEALED_SCIENTIFIC)
        self.assertEqual(sum(1 for e in self.episodes if e.uncertainty_intervention), 0)

    def test_no_episode_terminated_on_the_gate(self) -> None:
        # The terminal reason is the independent record: an intervention that
        # fired would have ended the episode with this reason.
        reasons = {e.terminal_reason for e in self.episodes}
        self.assertNotIn("uncertainty_intervention", reasons)


class PaperTextTests(unittest.TestCase):
    """The prose must quote the numbers the artifacts hold."""

    def setUp(self) -> None:
        _skip_unless(PAPER)
        # LaTeX wraps prose at arbitrary columns, so a phrase can straddle a
        # newline without the claim changing. Collapsing runs of whitespace
        # keeps this suite sensitive to edits and blind to reflow.
        raw = PAPER.read_text(encoding="utf-8")
        self.text = " ".join(raw.split())

    def test_the_sealed_counts_appear_in_the_paper(self) -> None:
        for value in ("888 attempts", "505 scientific", "378 operational"):
            self.assertIn(value, self.text)

    def test_the_headline_contrast_appears_verbatim(self) -> None:
        self.assertIn("-0.333", self.text)
        self.assertIn("[-0.500, -0.150]", self.text)

    def test_the_c0_success_fractions_appear(self) -> None:
        self.assertIn("$9/60$", self.text)
        self.assertIn("$29/60$", self.text)

    def test_the_paper_states_the_criterion_failed(self) -> None:
        self.assertIn("Criterion 14.1 fails", self.text)

    def test_the_paper_does_not_claim_fusion_helped(self) -> None:
        # A phrasing guard, not a proof. These are the sentences a later edit
        # would most plausibly introduce by accident when softening a negative.
        for forbidden in (
            "fusion is supported",
            "fusion outperforms",
            "improves on the LiDAR-only baseline",
        ):
            self.assertNotIn(forbidden, self.text)

    def test_the_c1_coverage_defect_is_disclosed(self) -> None:
        # The defect is ours and limits what C1 can be used to claim. It must
        # not disappear from the paper in an edit that shortens the limitations.
        self.assertIn("within-world evidence", self.text)
        self.assertIn("single cluster", self.text)


if __name__ == "__main__":
    unittest.main()
