from __future__ import annotations

import unittest

from livifuser_nav.simulation_supervision import (
    PrivilegedState,
    ProposalInput,
    SimulationSupervisor,
)


def proposal(stamp_ns: int, **updates) -> ProposalInput:
    values = {
        "stamp_ns": stamp_ns,
        "linear_x": 0.10,
        "angular_z": 0.50,
        "valid": True,
        "inference_ready": True,
        "status": "inference",
        "combined_intervention": False,
    }
    values.update(updates)
    return ProposalInput(**values)


SAFE = PrivilegedState(True, False, 1.0, 0.5)


class SimulationSupervisorTest(unittest.TestCase):
    def test_clipping_and_slew(self) -> None:
        supervisor = SimulationSupervisor()
        first = supervisor.step(
            proposal(1_000_000_000, linear_x=1.0, angular_z=2.0), SAFE
        )
        self.assertEqual(first.clipped, (0.1, 0.5))
        self.assertAlmostEqual(first.executed[0], 0.05)
        self.assertAlmostEqual(first.executed[1], 0.1)
        second = supervisor.step(proposal(1_100_000_000), SAFE)
        self.assertAlmostEqual(second.executed[0], 0.1)
        self.assertAlmostEqual(second.executed[1], 0.2)

    def test_stop_bypasses_slew(self) -> None:
        supervisor = SimulationSupervisor()
        supervisor.step(proposal(1_000_000_000), SAFE)
        stopped = supervisor.step(
            proposal(1_100_000_000, combined_intervention=True), SAFE
        )
        self.assertEqual(stopped.executed, (0.0, 0.0))
        self.assertEqual(stopped.terminal_reason, "uncertainty_intervention")

    def test_collision_precedes_uncertainty(self) -> None:
        supervisor = SimulationSupervisor()
        state = PrivilegedState(True, True, 1.0, 0.05)
        decision = supervisor.step(
            proposal(1_000_000_000, combined_intervention=True), state
        )
        self.assertEqual(decision.terminal_reason, "collision")

    def test_success_requires_three_consecutive_samples(self) -> None:
        supervisor = SimulationSupervisor()
        at_goal = PrivilegedState(True, False, 0.2, 1.0)
        self.assertFalse(supervisor.step(proposal(1_000_000_000), at_goal).terminal_reason)
        self.assertFalse(supervisor.step(proposal(1_100_000_000), at_goal).terminal_reason)
        result = supervisor.step(proposal(1_200_000_000), at_goal)
        self.assertEqual(result.terminal_reason, "success")

    def test_invalid_input_and_missing_ground_truth_command_zero(self) -> None:
        supervisor = SimulationSupervisor()
        invalid = supervisor.step(
            proposal(1_000_000_000, valid=False, status="rgb_missing"), SAFE
        )
        self.assertEqual(invalid.executed, (0.0, 0.0))
        self.assertEqual(invalid.reason, "rgb_missing")
        missing = supervisor.step(
            proposal(1_100_000_000), PrivilegedState(False, False, float("inf"), float("nan"))
        )
        self.assertEqual(missing.executed, (0.0, 0.0))
        self.assertEqual(missing.reason, "ground_truth_missing")

    def test_scientific_timeout_is_terminal(self) -> None:
        supervisor = SimulationSupervisor(scientific_deadline_sec=0.2)
        supervisor.step(proposal(1_000_000_000), SAFE)
        decision = supervisor.step(proposal(1_200_000_000), SAFE)
        self.assertEqual(decision.terminal_reason, "scientific_timeout")

    def test_scientific_timeout_ends_warmup_that_never_becomes_ready(self) -> None:
        supervisor = SimulationSupervisor(scientific_deadline_sec=0.2)
        supervisor.step(
            proposal(1_000_000_000, valid=True, inference_ready=False, status="warmup"),
            SAFE,
        )
        decision = supervisor.step(
            proposal(1_200_000_000, valid=True, inference_ready=False, status="warmup"),
            SAFE,
        )
        self.assertEqual(decision.terminal_reason, "scientific_timeout")
        self.assertEqual(decision.executed, (0.0, 0.0))


class ControlIntervalRecoveryTest(unittest.TestCase):
    """A stretched control interval recovers; it does not end the episode.

    Amendment section 3: a missing or stale input "clears the complete K-context
    history and makes the supervisor command zero for that tick". Section 6
    reserves termination for non-finite commands, exceptions, integrity loss,
    clock failure, and watchdog expiry. Terminating on a stretched interval was
    stricter than the contract and caused
    operational_failure_control_interval in both gate 6 sweeps.
    """

    def test_a_stretched_interval_commands_zero_without_terminating(self) -> None:
        supervisor = SimulationSupervisor()
        supervisor.step(proposal(1_000_000_000), SAFE)
        # 400 ms of simulated time: two ticks were missed.
        decision = supervisor.step(proposal(1_400_000_000), SAFE)
        self.assertEqual(decision.executed, (0.0, 0.0))
        self.assertEqual(decision.reason, "control_interval_recovered")
        self.assertEqual(decision.terminal_reason, "")
        self.assertEqual(supervisor.stretched_interval_count, 1)

    def test_the_episode_continues_after_recovery(self) -> None:
        supervisor = SimulationSupervisor()
        supervisor.step(proposal(1_000_000_000), SAFE)
        supervisor.step(proposal(1_400_000_000), SAFE)
        resumed = supervisor.step(proposal(1_500_000_000), SAFE)
        self.assertEqual(resumed.terminal_reason, "")
        self.assertNotEqual(resumed.executed, (0.0, 0.0))

    def test_a_long_gap_does_not_authorise_a_large_velocity_step(self) -> None:
        # The measured gap must not reach the slew limiter: at 0.5 m/s^2 a
        # naive 2 s interval would permit a 1.0 m/s step.
        supervisor = SimulationSupervisor()
        supervisor.step(proposal(1_000_000_000), SAFE)
        supervisor.step(proposal(3_000_000_000), SAFE)
        following = supervisor.step(proposal(3_100_000_000), SAFE)
        self.assertLessEqual(abs(following.executed[0]), 0.05 + 1e-9)

    def test_stretched_intervals_are_counted(self) -> None:
        supervisor = SimulationSupervisor()
        supervisor.step(proposal(1_000_000_000), SAFE)
        for stamp in (1_400_000_000, 1_900_000_000, 2_500_000_000):
            supervisor.step(proposal(stamp), SAFE)
        self.assertEqual(supervisor.stretched_interval_count, 3)
        self.assertEqual(supervisor.terminal_reason, "")

    def test_a_stamp_that_does_not_advance_is_still_terminal(self) -> None:
        # Not a missing input: proposals are ordered by construction, so this
        # means two publishers on one topic or a corrupted clock.
        supervisor = SimulationSupervisor()
        supervisor.step(proposal(1_000_000_000), SAFE)
        decision = supervisor.step(proposal(1_000_000_000), SAFE)
        self.assertEqual(
            decision.terminal_reason, "operational_failure_proposal_stamp_regression"
        )

    def test_a_nominal_interval_is_untouched(self) -> None:
        supervisor = SimulationSupervisor()
        supervisor.step(proposal(1_000_000_000), SAFE)
        decision = supervisor.step(proposal(1_100_000_000), SAFE)
        self.assertEqual(supervisor.stretched_interval_count, 0)
        self.assertEqual(decision.reason, "learned_command")


if __name__ == "__main__":
    unittest.main()
