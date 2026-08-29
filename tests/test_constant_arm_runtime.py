"""Pin the open-loop constant-training-mean reference arm.

Closed-loop execution amendment section 1.1 and readiness gate 5. The arm's
whole evidential value is that it is sensor-blind and exactly the preregistered
offline constant; every property that could silently erode either is pinned
here rather than left to review.
"""

from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from livifuser_nav.live_runtime import (
    CONSTANT_ARM_ANGULAR_Z_RADPS,
    CONSTANT_ARM_LINEAR_X_MPS,
    CONSTANT_ARM_NAME,
    CONSTANT_ARM_UNCERTAINTY,
    CONTEXT_K,
    ConstantActionRuntime,
    ConstantArmDecision,
    RuntimeDecision,
)

ROOT = Path(__file__).resolve().parents[1]
COMPANION = ROOT / "config/simulation_closed_loop_execution_v1.proposed.json"

# The exact float64 constant computed before held-out access from all 56,128
# verified training rows across 120 episodes. Held-out evaluation amendment
# section 3 records both the decimal and the hexadecimal form.
EXPECTED_LINEAR_X = 0.047447296062892504
EXPECTED_ANGULAR_Z = -0.005205255475722954


class ConstantArmActionTests(unittest.TestCase):
    def test_action_matches_the_preregistered_constant_exactly(self) -> None:
        self.assertEqual(CONSTANT_ARM_LINEAR_X_MPS, EXPECTED_LINEAR_X)
        self.assertEqual(CONSTANT_ARM_ANGULAR_Z_RADPS, EXPECTED_ANGULAR_Z)
        # Bit-exact, not merely close: a rounded literal is a different arm.
        self.assertEqual(CONSTANT_ARM_LINEAR_X_MPS.hex(), EXPECTED_LINEAR_X.hex())
        self.assertEqual(CONSTANT_ARM_ANGULAR_Z_RADPS.hex(), EXPECTED_ANGULAR_Z.hex())

    def test_action_is_inside_the_frozen_supervisor_limits(self) -> None:
        # If it were not, the supervisor would clip it and the arm would no
        # longer be the preregistered constant.
        self.assertLessEqual(abs(CONSTANT_ARM_LINEAR_X_MPS), 0.10)
        self.assertLessEqual(abs(CONSTANT_ARM_ANGULAR_Z_RADPS), 0.50)

    def test_companion_config_agrees_with_the_implementation(self) -> None:
        arm = json.loads(COMPANION.read_text("utf-8"))["reference_arms"][CONSTANT_ARM_NAME]
        self.assertEqual(arm["action_linear_x_mps"], CONSTANT_ARM_LINEAR_X_MPS)
        self.assertEqual(arm["action_angular_z_radps"], CONSTANT_ARM_ANGULAR_Z_RADPS)
        self.assertEqual(
            float.fromhex(arm["action_linear_x_float64_hex"]), CONSTANT_ARM_LINEAR_X_MPS
        )
        self.assertEqual(
            float.fromhex(arm["action_angular_z_float64_hex"]), CONSTANT_ARM_ANGULAR_Z_RADPS
        )
        self.assertEqual(arm["zero_command_contexts_before_first_proposal"], CONTEXT_K - 1)
        self.assertEqual(arm["first_proposal_on_accepted_context"], CONTEXT_K)


class ConstantArmWarmupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = ConstantActionRuntime()

    def test_first_seven_contexts_command_zero_and_the_eighth_proposes(self) -> None:
        for index in range(1, CONTEXT_K):
            decision = self.runtime.accept()
            self.assertFalse(decision.ready, index)
            self.assertEqual(decision.status, f"warmup_{index}_of_{CONTEXT_K}")
            np.testing.assert_array_equal(decision.proposed_action, np.zeros(2))
        decision = self.runtime.accept()
        self.assertTrue(decision.ready)
        self.assertEqual(decision.status, CONSTANT_ARM_NAME)
        np.testing.assert_array_equal(
            decision.proposed_action,
            np.asarray([EXPECTED_LINEAR_X, EXPECTED_ANGULAR_Z], dtype=np.float64),
        )

    def test_action_never_changes_after_warmup(self) -> None:
        for _ in range(CONTEXT_K):
            self.runtime.accept()
        expected = np.asarray([EXPECTED_LINEAR_X, EXPECTED_ANGULAR_Z], dtype=np.float64)
        for _ in range(1200):  # two full 120 s episodes at 10 Hz
            decision = self.runtime.accept()
            self.assertTrue(decision.ready)
            np.testing.assert_array_equal(decision.proposed_action, expected)

    def test_reset_re_arms_the_full_warmup(self) -> None:
        for _ in range(CONTEXT_K + 40):
            self.runtime.accept()
        self.runtime.clear_history()
        self.assertEqual(self.runtime.accepted_contexts, 0)
        decision = self.runtime.accept()
        self.assertFalse(decision.ready)
        np.testing.assert_array_equal(decision.proposed_action, np.zeros(2))

    def test_proposed_action_is_not_a_shared_mutable_buffer(self) -> None:
        for _ in range(CONTEXT_K):
            self.runtime.accept()
        first = self.runtime.accept().proposed_action
        first[0] = 99.0
        second = self.runtime.accept().proposed_action
        self.assertEqual(second[0], EXPECTED_LINEAR_X)


class ConstantArmSensorBlindnessTests(unittest.TestCase):
    def test_accept_takes_no_observation_argument(self) -> None:
        # Structural, not conventional: there is no parameter through which a
        # camera, scan, odometry, or goal could reach this arm.
        signature = inspect.signature(ConstantActionRuntime.accept)
        self.assertEqual(list(signature.parameters), ["self"])

    def test_runtime_has_no_model_or_sensor_state(self) -> None:
        runtime = ConstantActionRuntime()
        forbidden = ("backbone", "policy", "material", "sensor_contract", "history", "device")
        for name in forbidden:
            self.assertFalse(hasattr(runtime, name), name)

    def test_uncertainty_is_not_applicable_rather_than_a_gate_that_did_not_fire(self) -> None:
        runtime = ConstantActionRuntime()
        for _ in range(CONTEXT_K):
            decision = runtime.accept()
        self.assertEqual(decision.uncertainty, "NOT_APPLICABLE")
        self.assertEqual(CONSTANT_ARM_UNCERTAINTY, "NOT_APPLICABLE")
        self.assertFalse(decision.combined_intervention)
        # A zero-valued score would be indistinguishable from a real score that
        # did not exceed its threshold, so these fields must not exist at all.
        for name in ("aleatoric", "mahalanobis", "combined", "z_aleatoric", "z_mahalanobis"):
            self.assertFalse(hasattr(decision, name), name)

    def test_decision_type_is_distinct_from_the_learned_runtime_decision(self) -> None:
        self.assertIsNot(ConstantArmDecision, RuntimeDecision)
        self.assertFalse(issubclass(ConstantArmDecision, RuntimeDecision))


if __name__ == "__main__":
    unittest.main()
