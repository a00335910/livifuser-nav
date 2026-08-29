"""Frozen C1 visual-scene contract tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_sim.visual_conditions import (  # noqa: E402
    C1_VISUAL_CONTRACT_NAME,
    C1_VISUAL_CONTRACT_SHA256,
    c1_condition_descriptor,
    evaluate_c1_development_gate,
    load_c1_visual_contract,
)


class TestFrozenC1VisualContract(unittest.TestCase):
    def test_contract_is_checksum_pinned_and_numerically_explicit(self):
        contract = load_c1_visual_contract()
        self.assertEqual(contract["name"], C1_VISUAL_CONTRACT_NAME)
        self.assertEqual(
            c1_condition_descriptor(),
            {"name": C1_VISUAL_CONTRACT_NAME, "sha256": C1_VISUAL_CONTRACT_SHA256},
        )
        self.assertEqual(contract["scene"]["ambient_rgba"], [0.25, 0.18, 0.12, 1.0])
        self.assertEqual(
            contract["directional_light"]["diffuse_rgba"],
            [0.55, 0.38, 0.22, 1.0],
        )
        self.assertEqual(
            contract["material_transform"]["rgb_permutation"], [2, 0, 1]
        )

    def test_checksum_rejects_silent_contract_drift(self):
        source = PACKAGE_ROOT / "config" / "c1_visual_condition_v1.json"
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / source.name
            changed.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_c1_visual_contract(changed)

    def test_development_gate_accepts_visible_nondegenerate_shift(self):
        c0 = {
            "channel_mean_rgb_normalized": [0.50, 0.45, 0.40],
            "luminance_mean_normalized": 0.45,
        }
        c1 = {
            "channel_mean_rgb_normalized": [0.30, 0.25, 0.18],
            "luminance_mean_normalized": 0.25,
            "luminance_std_normalized": 0.10,
            "near_black_fraction": 0.05,
            "near_white_fraction": 0.0,
        }
        result = evaluate_c1_development_gate(c0, c1)
        self.assertTrue(result["valid"])
        self.assertEqual(result["issues"], [])

    def test_development_gate_rejects_vacuous_shift(self):
        unchanged = {
            "channel_mean_rgb_normalized": [0.40, 0.40, 0.40],
            "luminance_mean_normalized": 0.40,
            "luminance_std_normalized": 0.10,
            "near_black_fraction": 0.0,
            "near_white_fraction": 0.0,
        }
        result = evaluate_c1_development_gate(unchanged, unchanged)
        self.assertFalse(result["valid"])
        self.assertIn("appearance_shift_too_small", result["issues"])
        self.assertIn("illumination_shift_too_small", result["issues"])


if __name__ == "__main__":
    unittest.main()
