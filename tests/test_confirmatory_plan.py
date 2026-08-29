"""The confirmatory execution plan and its no-rerun rules.

These are the rules that decide whether the evidence is admissible, so they are
tested against the amendment's wording rather than against the implementation's
convenience. The central one: an identity with an accepted scientific outcome is
finished forever, because rerunning it is optional stopping.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from livifuser_nav.confirmatory_plan import (
    CONSTANT_ARM,
    LEARNED_ROLLOUTS,
    MAX_ATTEMPTS_PER_IDENTITY,
    NAV2_ARM,
    SEEDS,
    TOTAL_ROLLOUTS,
    VARIANTS,
    build_plan,
    classify_attempt,
    identity_state,
    next_identity,
    progress,
    select_subset,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "artifacts/simulation/confirmatory_v3/schedule.json"


def _write_attempt(root: Path, identity, number: int, payload: dict | None) -> Path:
    directory = root / identity.arm / str(identity.seed) / f"{identity.ordinal:04d}"
    attempt = directory / f"attempt_{number:03d}"
    attempt.mkdir(parents=True, exist_ok=True)
    if payload is None:
        (attempt / "operational_failure.json").write_text('{"status":"operational_failure"}')
    else:
        (attempt / "terminal.json").write_text(json.dumps(payload))
    return attempt


def _terminal(reason: str, context_sequence: int = 640) -> dict:
    # A real episode always carries a positive control-decision count; an
    # episode with zero is one where the policy never ran and is operational.
    return {
        "terminal": True,
        "terminal_reason": reason,
        "context_sequence": context_sequence,
    }


@unittest.skipUnless(SCHEDULE.is_file(), "confirmatory schedule is not present")
class PlanShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_plan(SCHEDULE)

    def test_the_plan_is_exactly_the_frozen_scope(self) -> None:
        self.assertEqual(len(self.plan), TOTAL_ROLLOUTS)
        self.assertEqual(len({identity.key for identity in self.plan}), TOTAL_ROLLOUTS)

    def test_learned_arms_come_first_in_the_frozen_order(self) -> None:
        learned = self.plan[:LEARNED_ROLLOUTS]
        self.assertEqual(learned[0].arm, VARIANTS[0])
        self.assertEqual(learned[0].seed, SEEDS[0])
        # Variant is the outermost loop, then seed, then ordinal.
        self.assertEqual([identity.arm for identity in learned[:240]], ["full"] * 240)
        self.assertEqual([identity.seed for identity in learned[:80]], [SEEDS[0]] * 80)
        self.assertEqual(
            [identity.ordinal for identity in learned[:80]], list(range(180, 260))
        )

    def test_reference_arms_follow_the_learned_set(self) -> None:
        self.assertEqual(self.plan[LEARNED_ROLLOUTS].arm, CONSTANT_ARM)
        self.assertEqual(self.plan[LEARNED_ROLLOUTS].seed, 0)
        self.assertEqual(self.plan[-1].arm, NAV2_ARM)

    def test_nav2_covers_only_c0_and_c4(self) -> None:
        nav2 = [identity for identity in self.plan if identity.arm == NAV2_ARM]
        self.assertEqual(len(nav2), 40)
        self.assertEqual({identity.condition for identity in nav2}, {"C0", "C4"})

    def test_every_learned_identity_carries_a_real_training_seed(self) -> None:
        for identity in self.plan[:LEARNED_ROLLOUTS]:
            self.assertIn(identity.seed, SEEDS)

    def test_the_plan_is_stable_across_builds(self) -> None:
        # The order must not depend on dictionary iteration or filesystem state.
        again = build_plan(SCHEDULE)
        self.assertEqual([i.key for i in self.plan], [i.key for i in again])


class AttemptClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _attempt(self, name: str, payload: dict | None) -> Path:
        attempt = self.tmp / name
        attempt.mkdir()
        if payload is None:
            (attempt / "operational_failure.json").write_text("{}")
        else:
            payload = {"context_sequence": 640, **payload}
            (attempt / "terminal.json").write_text(json.dumps(payload))
        return attempt

    def test_scientific_terminals_are_recognised(self) -> None:
        for reason in ("success", "collision", "uncertainty_intervention", "scientific_timeout"):
            attempt = self._attempt(f"a_{reason}", _terminal(reason))
            self.assertEqual(classify_attempt(attempt), "scientific", reason)

    def test_an_operational_terminal_is_not_scientific(self) -> None:
        # These reasons appear in terminal.json but are infrastructure failures;
        # treating them as scientific would retire an identity that never ran.
        for reason in (
            "operational_failure_control_interval",
            "operational_failure_proposal_stamp_regression",
            "watchdog_timeout",
            "operational_failure_policy_identity",
        ):
            attempt = self._attempt(f"b_{reason}", _terminal(reason))
            self.assertEqual(classify_attempt(attempt), "operational", reason)

    def test_a_corrupt_terminal_record_is_operational_not_scientific(self) -> None:
        attempt = self.tmp / "corrupt"
        attempt.mkdir()
        (attempt / "terminal.json").write_text("{not json")
        self.assertEqual(classify_attempt(attempt), "operational")

    def test_an_empty_attempt_is_absent(self) -> None:
        attempt = self.tmp / "empty"
        attempt.mkdir()
        self.assertEqual(classify_attempt(attempt), "absent")


@unittest.skipUnless(SCHEDULE.is_file(), "confirmatory schedule is not present")
class ExecutionOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.plan = build_plan(SCHEDULE)

    def test_an_empty_root_starts_at_the_first_identity(self) -> None:
        identity, reason = next_identity(self.tmp, self.plan)
        self.assertEqual(identity.key, self.plan[0].key)
        self.assertEqual(reason, "first attempt")

    def test_a_scientific_outcome_retires_the_identity_forever(self) -> None:
        _write_attempt(self.tmp, self.plan[0], 1, _terminal("collision"))
        identity, _ = next_identity(self.tmp, self.plan)
        self.assertEqual(identity.key, self.plan[1].key)
        # And it stays retired even with further attempts present.
        self.assertTrue(identity_state(self.tmp, self.plan[0])["complete"])

    def test_an_operational_failure_retries_the_same_identity(self) -> None:
        _write_attempt(self.tmp, self.plan[0], 1, None)
        identity, reason = next_identity(self.tmp, self.plan)
        self.assertEqual(identity.key, self.plan[0].key)
        self.assertEqual(reason, "retry")

    def test_repeated_failure_pauses_the_batch_rather_than_skipping(self) -> None:
        # Skipping would silently drop a scientific identity from the design.
        for number in range(1, MAX_ATTEMPTS_PER_IDENTITY + 1):
            _write_attempt(self.tmp, self.plan[0], number, None)
        identity, reason = next_identity(self.tmp, self.plan)
        self.assertIsNone(identity)
        self.assertIn("paused", reason)
        self.assertIn(self.plan[0].key, reason)

    def test_a_late_scientific_outcome_after_retries_still_retires_it(self) -> None:
        _write_attempt(self.tmp, self.plan[0], 1, None)
        _write_attempt(self.tmp, self.plan[0], 2, _terminal("success"))
        identity, _ = next_identity(self.tmp, self.plan)
        self.assertEqual(identity.key, self.plan[1].key)

    def test_the_batch_reports_complete_only_when_every_identity_is_done(self) -> None:
        for identity in self.plan:
            _write_attempt(self.tmp, identity, 1, _terminal("scientific_timeout"))
        state = progress(self.tmp, self.plan)
        self.assertTrue(state["batch_complete"])
        self.assertEqual(state["identities_complete"], TOTAL_ROLLOUTS)
        identity, reason = next_identity(self.tmp, self.plan)
        self.assertIsNone(identity)
        self.assertIn("accepted scientific outcome", reason)

    def test_progress_reveals_no_scientific_aggregate(self) -> None:
        # Section 9 forbids showing one before the set is complete and sealed; a
        # visible success rate mid-batch invites stopping on a favourable trend.
        _write_attempt(self.tmp, self.plan[0], 1, _terminal("success"))
        _write_attempt(self.tmp, self.plan[1], 1, _terminal("collision"))
        state = progress(self.tmp, self.plan)
        text = json.dumps(state).lower()
        for forbidden in ("success_rate", "collision_rate", "successes", "collisions"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(SCHEDULE.is_file(), "confirmatory schedule is not present")
class AuditClearanceTests(unittest.TestCase):
    """Releasing a paused identity is a deliberate, recorded act.

    Section 9 requires repeated failure to pause the batch for a documented
    audit, and requires the cause to be resolved "without changing scientific
    inputs". So the failed attempts stay, and only attempts made after the
    clearance count against the budget again.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.plan = build_plan(SCHEDULE)
        # The first identity must already be finished, or the walk stops there
        # and never reaches the paused one under test.
        _write_attempt(self.tmp, self.plan[0], 1, _terminal("collision"))
        self.identity = self.plan[1]
        for number in range(1, MAX_ATTEMPTS_PER_IDENTITY + 1):
            _write_attempt(self.tmp, self.identity, number, None)

    def _clear(self, attempts_at_clearance: int) -> None:
        directory = (
            self.tmp
            / self.identity.arm
            / str(self.identity.seed)
            / f"{self.identity.ordinal:04d}"
        )
        (directory / "audit_cleared.json").write_text(
            json.dumps(
                {
                    "attempts_at_clearance": attempts_at_clearance,
                    "reason": "interpreter resolved from PATH; wait_sim_terminal ran "
                    "under python 3.12 which has no rclpy",
                    "scientific_inputs_changed": False,
                }
            )
        )

    def test_without_clearance_the_identity_stays_paused(self) -> None:
        identity, reason = next_identity(self.tmp, self.plan)
        self.assertIsNone(identity)
        self.assertIn("paused", reason)

    def test_a_clearance_releases_the_identity(self) -> None:
        self._clear(MAX_ATTEMPTS_PER_IDENTITY)
        identity, reason = next_identity(self.tmp, self.plan)
        self.assertEqual(identity.key, self.identity.key)
        self.assertEqual(reason, "retry")

    def test_the_failed_attempts_are_preserved(self) -> None:
        self._clear(MAX_ATTEMPTS_PER_IDENTITY)
        state = identity_state(self.tmp, self.identity)
        self.assertEqual(state["attempts"], MAX_ATTEMPTS_PER_IDENTITY)
        self.assertEqual(state["outcomes"], ["operational"] * MAX_ATTEMPTS_PER_IDENTITY)

    def test_the_budget_resets_only_for_later_attempts(self) -> None:
        self._clear(MAX_ATTEMPTS_PER_IDENTITY)
        # Fail the full budget again after the clearance.
        for number in range(
            MAX_ATTEMPTS_PER_IDENTITY + 1, 2 * MAX_ATTEMPTS_PER_IDENTITY + 1
        ):
            _write_attempt(self.tmp, self.identity, number, None)
        identity, reason = next_identity(self.tmp, self.plan)
        self.assertIsNone(identity)
        self.assertIn("paused", reason)

    def test_a_clearance_does_not_fabricate_a_scientific_outcome(self) -> None:
        self._clear(MAX_ATTEMPTS_PER_IDENTITY)
        self.assertFalse(identity_state(self.tmp, self.identity)["complete"])
        state = progress(self.tmp, self.plan)
        # Only the deliberately completed first identity counts; a clearance
        # must never manufacture an outcome for the identity it releases.
        self.assertEqual(state["identities_complete"], 1)

    def test_a_corrupt_clearance_is_ignored_rather_than_trusted(self) -> None:
        directory = (
            self.tmp
            / self.identity.arm
            / str(self.identity.seed)
            / f"{self.identity.ordinal:04d}"
        )
        (directory / "audit_cleared.json").write_text("{not json")
        identity, reason = next_identity(self.tmp, self.plan)
        self.assertIsNone(identity)
        self.assertIn("paused", reason)


@unittest.skipUnless(SCHEDULE.is_file(), "confirmatory schedule is not present")
class LidarConditionBindingTests(unittest.TestCase):
    """The LiDAR condition is read from the schedule, never derived.

    C1 and C4 are camera and obstacle-visibility manipulations that leave the
    LiDAR nominal; only C3 binds to C3b. Deriving the LiDAR condition from the
    episode condition passed "C1" and "C4" to the analytic LiDAR node, which
    accepts only C0/C3a/C3b. The node died at startup, /scan never existed, and
    every affected episode recorded a robot that never moved.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_plan(SCHEDULE)

    def test_every_identity_carries_a_supported_lidar_condition(self) -> None:
        supported = {"C0", "C3a", "C3b"}
        for identity in self.plan:
            self.assertIn(
                identity.lidar_condition,
                supported,
                f"{identity.key} ({identity.condition}) -> {identity.lidar_condition}",
            )

    def test_the_frozen_mapping_is_exactly_this(self) -> None:
        seen = {}
        for identity in self.plan:
            seen.setdefault(identity.condition, set()).add(identity.lidar_condition)
        self.assertEqual(seen.get("C0"), {"C0"})
        self.assertEqual(seen.get("C1"), {"C0"}, "C1 is a camera condition; LiDAR stays nominal")
        self.assertEqual(seen.get("C3"), {"C3b"})
        self.assertEqual(seen.get("C4"), {"C0"}, "C4 hides an obstacle from LiDAR, not the sensor")

    def test_the_episode_condition_is_not_a_usable_substitute(self) -> None:
        # The exact bug: the old code passed identity.condition with one special
        # case for C3. This asserts that substitution is wrong for C1 and C4.
        wrong = [
            i for i in self.plan
            if (i.condition if i.condition != "C3" else "C3b") != i.lidar_condition
        ]
        self.assertTrue(wrong, "if this ever passes empty, the mapping changed")
        self.assertEqual({i.condition for i in wrong}, {"C1", "C4"})


class SilentNoOpEpisodeTests(unittest.TestCase):
    """An episode where the policy never ticked is operational, not scientific."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _attempt(self, name: str, payload: dict) -> Path:
        attempt = self.tmp / name
        attempt.mkdir()
        (attempt / "terminal.json").write_text(json.dumps(payload))
        return attempt

    def test_zero_control_decisions_is_never_a_scientific_outcome(self) -> None:
        for reason in ("scientific_timeout", "success", "collision"):
            attempt = self._attempt(
                f"zero_{reason}",
                {"terminal": True, "terminal_reason": reason, "context_sequence": 0},
            )
            self.assertEqual(classify_attempt(attempt), "operational", reason)

    def test_a_real_episode_with_decisions_stays_scientific(self) -> None:
        attempt = self._attempt(
            "real",
            {"terminal": True, "terminal_reason": "scientific_timeout", "context_sequence": 1203},
        )
        self.assertEqual(classify_attempt(attempt), "scientific")

    def test_a_missing_or_null_sequence_is_treated_as_zero(self) -> None:
        for payload in (
            {"terminal": True, "terminal_reason": "success"},
            {"terminal": True, "terminal_reason": "success", "context_sequence": None},
        ):
            attempt = self._attempt(f"missing_{len(payload)}", payload)
            self.assertEqual(classify_attempt(attempt), "operational")

    def test_one_decision_is_enough_to_count(self) -> None:
        attempt = self._attempt(
            "one", {"terminal": True, "terminal_reason": "collision", "context_sequence": 1}
        )
        self.assertEqual(classify_attempt(attempt), "scientific")


@unittest.skipUnless(SCHEDULE.is_file(), "confirmatory schedule is not present")
class SubsetSelectionTests(unittest.TestCase):
    """A reduced scope must be a rule, not a choice.

    The compute allowance ran out before the plan could finish, so only part of
    it can be executed. Which part is decided by a written rule applied to the
    frozen schedule -- the first N of each cell in frozen order -- so the
    selection is reproducible from the schedule alone and cannot encode anything
    about how the episodes turned out.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_plan(SCHEDULE)

    def test_filtering_by_arm_and_condition(self) -> None:
        subset = select_subset(self.plan, arms=("rgb_only",), conditions=("C0",))
        self.assertEqual({i.arm for i in subset}, {"rgb_only"})
        self.assertEqual({i.condition for i in subset}, {"C0"})
        self.assertEqual(len(subset), 60, "3 seeds x 20 episodes")

    def test_max_per_cell_caps_each_arm_seed_condition(self) -> None:
        subset = select_subset(
            self.plan, arms=("full", "lidar_only", "concat", "rgb_only"),
            conditions=("C1",), max_per_cell=10,
        )
        self.assertEqual(len(subset), 120, "4 arms x 3 seeds x 10")
        counts = {}
        for i in subset:
            counts[(i.arm, i.seed)] = counts.get((i.arm, i.seed), 0) + 1
        self.assertEqual(set(counts.values()), {10})

    def test_selection_takes_the_first_of_each_cell_in_frozen_order(self) -> None:
        subset = select_subset(
            self.plan, arms=("full",), conditions=("C1",), max_per_cell=3
        )
        first_seed = [i for i in subset if i.seed == SEEDS[0]]
        self.assertEqual([i.ordinal for i in first_seed], [190, 191, 192])

    def test_the_subset_preserves_plan_order(self) -> None:
        subset = select_subset(self.plan, conditions=("C0",), max_per_cell=5)
        indices = [i.index for i in subset]
        self.assertEqual(indices, sorted(indices))

    def test_selection_is_deterministic(self) -> None:
        a = select_subset(self.plan, conditions=("C1",), max_per_cell=7)
        b = select_subset(build_plan(SCHEDULE), conditions=("C1",), max_per_cell=7)
        self.assertEqual([i.key for i in a], [i.key for i in b])

    def test_no_filter_returns_the_whole_plan(self) -> None:
        self.assertEqual(len(select_subset(self.plan)), len(self.plan))


class AttemptCapTests(unittest.TestCase):
    """The frozen default is three; raising it is deliberate and documented."""

    def test_the_default_is_the_frozen_three(self) -> None:
        env = dict(os.environ)
        env.pop("LIVIFUSER_MAX_ATTEMPTS", None)
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src');"
             " from livifuser_nav.confirmatory_plan import MAX_ATTEMPTS_PER_IDENTITY as m;"
             " print(m)"],
            capture_output=True, text=True, env=env, cwd=str(ROOT), check=True,
        )
        self.assertEqual(result.stdout.strip(), "3")

    def test_the_environment_can_raise_it(self) -> None:
        env = dict(os.environ, LIVIFUSER_MAX_ATTEMPTS="8")
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src');"
             " from livifuser_nav.confirmatory_plan import MAX_ATTEMPTS_PER_IDENTITY as m;"
             " print(m)"],
            capture_output=True, text=True, env=env, cwd=str(ROOT), check=True,
        )
        self.assertEqual(result.stdout.strip(), "8")
