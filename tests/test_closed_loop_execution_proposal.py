from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/simulation_closed_loop_execution_v1.proposed.json"
AMENDMENT = (
    ROOT
    / "docs/experiments/PREREGISTRATION_SIM_CLOSED_LOOP_EXECUTION_AMENDMENT_2026-08-24.md"
)


class ClosedLoopExecutionProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text("utf-8"))
        cls.amendment = AMENDMENT.read_text("utf-8")

    def test_scope_is_exact_and_not_authorized(self) -> None:
        config = self.config
        self.assertEqual(
            config["learned_scope"]["variants"],
            ["full", "lidar_only", "concat", "rgb_only"],
        )
        self.assertEqual(
            config["learned_scope"]["seeds"], [20260805, 20260806, 20260807]
        )
        self.assertEqual(config["learned_scope"]["rollouts"], 960)
        self.assertEqual(config["nav2_structural_probe"]["rollouts"], 40)
        self.assertEqual(config["execution_order"]["total_identity_count"], 1080)
        self.assertEqual(
            config["execution_order"]["variant_order"],
            ["full", "lidar_only", "concat", "rgb_only"],
        )
        self.assertFalse(config["authorization"]["confirmatory_launch_authorized"])
        self.assertFalse(config["authorization"]["physical_robot_allowed"])
        self.assertFalse(config["authorization"]["cmd_vel_allowed"])

    def test_constant_reference_arm_is_exact_and_sensor_blind(self) -> None:
        arm = self.config["reference_arms"]["constant_training_mean"]
        # Byte-identical to the preregistered offline trivial baseline. These are
        # the exact float64 values; a rounded literal would silently change the arm.
        self.assertEqual(arm["action_linear_x_mps"], float.fromhex("0x1.84b0311bf5c89p-5"))
        self.assertEqual(arm["action_angular_z_radps"], float.fromhex("-0x1.5521b2091a221p-8"))
        self.assertEqual(arm["action_linear_x_mps"], 0.047447296062892504)
        self.assertEqual(arm["action_angular_z_radps"], -0.005205255475722954)
        self.assertFalse(arm["is_learned_variant"])
        self.assertFalse(arm["seed_replication"])
        self.assertEqual(arm["rollouts"], 80)
        self.assertEqual(arm["zero_command_contexts_before_first_proposal"], 7)
        self.assertEqual(arm["first_proposal_on_accepted_context"], 8)
        self.assertEqual(arm["reset_rule"], "IDENTICAL_TO_SECTION_3_LEARNED_ARMS")
        self.assertEqual(arm["reserved_identity_seed"], 0)
        self.assertNotIn(arm["reserved_identity_seed"], self.config["learned_scope"]["seeds"])
        self.assertTrue(arm["separate_node_from_learned_runner"])
        for source in ("camera", "lidar", "odometry", "goal"):
            self.assertFalse(arm[f"consumes_{source}"], source)
        self.assertFalse(arm["loads_backbone"])
        self.assertFalse(arm["loads_checkpoint"])
        self.assertEqual(arm["uncertainty_scores"], "NOT_APPLICABLE")
        self.assertTrue(arm["excluded_from_intervention_denominators"])
        self.assertTrue(arm["pooling_into_learned_variant_means_forbidden"])
        self.assertTrue(arm["recompute_or_retune_forbidden"])

    def test_repeat_last_arm_is_recorded_as_rejected(self) -> None:
        # Kept in the record so the rejection cannot later be mistaken for an
        # oversight: it is inert by construction, not merely untried.
        rejected = self.config["reference_arms"]["repeat_last_action"]
        self.assertEqual(rejected["status"], "CONSIDERED_AND_REJECTED_2026-08-25")
        self.assertIn("fixed point", rejected["reason"])
        self.assertIn("privileged expert", rejected["offline_analogue_differs"])
        self.assertIn("rejected as", self.amendment)

    def test_execution_order_places_constant_arm_between_learned_and_nav2(self) -> None:
        order = self.config["execution_order"]
        self.assertTrue(order["constant_arm_after_learned"])
        self.assertTrue(order["constant_arm_before_nav2"])
        self.assertEqual(order["nav2_after_learned"], "AFTER_LEARNED_AND_CONSTANT_ARM")
        self.assertEqual(
            order["total_identity_count"],
            self.config["learned_scope"]["rollouts"]
            + self.config["reference_arms"]["constant_training_mean"]["rollouts"]
            + self.config["nav2_structural_probe"]["rollouts"],
        )

    def test_frozen_backbone_and_gate_are_exact(self) -> None:
        config = self.config
        self.assertEqual(
            config["backbone"]["weights_sha256"],
            "208146E499DACE99E4C9376DDB8A26F77D64C31C46C4DC4B86FF8BC63B0235E2",
        )
        self.assertEqual(config["backbone"]["patch_tokens_after_pool"], [49, 384])
        self.assertTrue(config["backbone"]["live_cached_features_forbidden"])
        self.assertEqual(config["uncertainty"]["active_gate"], "combined")
        self.assertEqual(config["uncertainty"]["active_threshold_all_12"], 1.0)
        self.assertEqual(
            config["uncertainty"]["aleatoric_and_mahalanobis_flags"],
            "COUNTERFACTUAL_LOG_ONLY",
        )

    def test_amendment_is_unambiguously_proposed(self) -> None:
        self.assertIn("PROPOSED — NOT APPROVED — DO NOT EXECUTE", self.amendment)
        self.assertIn("None. This document is preparatory and proposed only.", self.amendment)
        self.assertIn("do not launch any of the", self.amendment)
        self.assertIn("80 constant-training-mean", self.amendment)
        self.assertFalse(self.config["readiness"]["constant_arm_unit_test_passed"])


if __name__ == "__main__":
    unittest.main()
