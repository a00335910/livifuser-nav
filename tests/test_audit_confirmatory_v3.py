from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "audit_confirmatory_v3.py"
SPEC = importlib.util.spec_from_file_location("audit_confirmatory_v3", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def temporal_stats() -> dict:
    return {
        "valid_rgb_frames": 100,
        "unique_frame_hashes": 80,
        "modal_frame_fraction": 0.1,
        "moving_pair_count": 20,
        "changed_moving_pair_fraction": 1.0,
        "motion_pair_mean_absolute_rgb_difference_median": 0.02,
        "maximum_identical_motion_run_sec": 0.0,
        "gates": {
            "minimum_valid_rgb_frames": 60,
            "minimum_unique_frame_hashes": 20,
            "maximum_modal_frame_fraction": 0.5,
            "minimum_moving_pair_count": 10,
            "minimum_changed_moving_pair_fraction": 0.5,
            "minimum_motion_pair_median_rgb_difference": 0.001,
            "maximum_identical_motion_run_sec": 1.0,
        },
    }


class TestConfirmatoryV3Audit(unittest.TestCase):
    def test_temporal_gate_accepts_motion_linked_rgb(self) -> None:
        values = MODULE.validate_temporal_rgb(temporal_stats())
        self.assertEqual(values["changed_moving_pair_fraction"], 1.0)

    def test_temporal_gate_rejects_degenerate_motion_frames(self) -> None:
        stats = temporal_stats()
        stats["changed_moving_pair_fraction"] = 0.0
        with self.assertRaisesRegex(ValueError, "changed_moving_pair_fraction"):
            MODULE.validate_temporal_rgb(stats)

    def test_c4_comparison_allows_only_lidar_membership_change(self) -> None:
        c0 = {
            "name": "world",
            "seed": 4,
            "start": [0, 0],
            "goal": [1, 0],
            "obstacles": [
                {
                    "name": "low",
                    "layers": {
                        "collision": True,
                        "expert": True,
                        "camera": True,
                        "lidar": True,
                    },
                    "profile_switchable": True,
                }
            ],
        }
        c4 = {
            "name": "world",
            "seed": 4,
            "start": [0, 0],
            "goal": [1, 0],
            "condition": "C4",
            "c4_hidden_from_lidar": ["low"],
            "obstacles": [
                {
                    "name": "low",
                    "layers": {
                        "collision": True,
                        "expert": True,
                        "camera": True,
                        "lidar": False,
                    },
                }
            ],
        }
        result = MODULE.compare_c4_worlds(c0, c4)
        self.assertTrue(result["only_lidar_membership_changed"])

    def test_c4_comparison_rejects_collision_change(self) -> None:
        c0 = {
            "obstacles": [
                {
                    "name": "low",
                    "layers": {
                        "collision": True,
                        "expert": True,
                        "camera": True,
                        "lidar": True,
                    },
                    "profile_switchable": True,
                }
            ]
        }
        c4 = {
            "condition": "C4",
            "c4_hidden_from_lidar": ["low"],
            "obstacles": [
                {
                    "name": "low",
                    "layers": {
                        "collision": False,
                        "expert": True,
                        "camera": True,
                        "lidar": False,
                    },
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "changed more than LiDAR"):
            MODULE.compare_c4_worlds(c0, c4)


if __name__ == "__main__":
    unittest.main()
