from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SIM_PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "livifuser_sim"
sys.path.insert(0, str(SIM_PACKAGE_ROOT))

from livifuser_sim.lidar_policy_features import lidar_only_features  # noqa: E402

from livifuser_nav.learning_data import tokenize_lidar  # noqa: E402


class LidarOnlyFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {
            "calibration": {
                "lidar_geometry": {
                    "angular_frame": {
                        "angle_min_rad": 0.0,
                        "range_min_m": 0.12,
                        "range_max_m": 8.0,
                    }
                },
                "static_transforms": {
                    "base_scan->camera": {
                        "translation": [0.0723955522, 0.0048472604, -0.0838973150],
                        "quaternion_xyzw": [
                            -0.4806489642,
                            0.5212435451,
                            -0.4930249275,
                            0.5041905996,
                        ],
                    }
                },
                "recorded_camera_info": {
                    "k": [316.21156, 0.0, 223.13834, 0.0, 315.6497, 107.39364, 0.0, 0.0, 1.0],
                    "d": [0.012344, 0.038138, -0.016819, 0.004823, 0.0],
                },
            }
        }

    def test_matches_training_tokenizer_feature_subset_exactly(self) -> None:
        beam_count = 399
        increment = math.tau / (beam_count + 1)
        ranges = np.linspace(0.15, 7.5, beam_count, dtype=np.float32)
        ranges[5] = 0.0
        ranges[101] = np.nan
        ranges[302] = 9.0
        training = tokenize_lidar(
            ranges,
            beam_count,
            increment,
            self.manifest,
            sectors=80,
            range_clip_m=10.0,
        )
        deployed = lidar_only_features(
            ranges,
            angle_min_rad=0.0,
            angle_increment_rad=increment,
            range_min_m=0.12,
            range_max_m=8.0,
            sectors=80,
            range_clip_m=10.0,
        )
        np.testing.assert_array_equal(deployed, training.features)

    def test_all_invalid_sector_uses_far_range_and_zero_validity(self) -> None:
        features = lidar_only_features(
            [math.nan] * 80,
            angle_min_rad=0.0,
            angle_increment_rad=math.tau / 81,
            range_min_m=0.12,
            range_max_m=8.0,
        )
        self.assertTrue(np.all(features[:, 0] == 1.0))
        self.assertTrue(np.all(features[:, 3] == 0.0))

    def test_invalid_contract_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            lidar_only_features(
                [1.0] * 20,
                angle_min_rad=0.0,
                angle_increment_rad=0.1,
                range_min_m=0.12,
                range_max_m=8.0,
                sectors=80,
            )

    def test_fixture_is_json_serializable_for_debugging(self) -> None:
        json.dumps(self.manifest)
