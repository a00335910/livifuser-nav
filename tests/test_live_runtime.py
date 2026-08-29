from __future__ import annotations

import contextlib
import os
import unittest

import numpy as np

from livifuser_nav.live_runtime import (
    CUBLAS_WORKSPACE_CONFIG,
    VALIDATION_REFERENCE_COUNT,
    LiveObservation,
    _validate_observation,
    configure_deterministic_torch,
    right_continuous_cdf,
)


class LiveRuntimeContractTest(unittest.TestCase):
    def test_right_continuous_cdf_counts_equal_values(self) -> None:
        reference = np.arange(VALIDATION_REFERENCE_COUNT, dtype=np.float64)
        self.assertEqual(right_continuous_cdf(reference, 0.0), 1 / reference.size)
        self.assertEqual(right_continuous_cdf(reference, -1.0), 0.0)
        self.assertEqual(right_continuous_cdf(reference, float(reference[-1])), 1.0)

    def test_observation_contract(self) -> None:
        observation = LiveObservation(
            rgb=np.zeros((240, 320, 3), dtype=np.uint8),
            scan_ranges=np.ones(400, dtype=np.float32),
            scan_beam_count=400,
            scan_angle_increment_rad=0.01,
            goal=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
            robot_state=np.zeros(2, dtype=np.float32),
        )
        _validate_observation(observation)

    def test_invalid_rgb_is_rejected(self) -> None:
        observation = LiveObservation(
            rgb=np.zeros((224, 224, 3), dtype=np.uint8),
            scan_ranges=np.ones(400, dtype=np.float32),
            scan_beam_count=400,
            scan_angle_increment_rad=0.01,
            goal=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
            robot_state=np.zeros(2, dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "RGB"):
            _validate_observation(observation)


class DeterministicCudaConfigTests(unittest.TestCase):
    """cuBLAS determinism is part of the frozen execution contract.

    Without CUBLAS_WORKSPACE_CONFIG, torch.use_deterministic_algorithms(True)
    raises on the first CUDA GEMM. The CUDA parity gate failed exactly this way.
    """

    def setUp(self) -> None:
        self._saved = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    def tearDown(self) -> None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        if self._saved is not None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = self._saved

    class _FakeTorch:
        class backends:
            pass

        def set_grad_enabled(self, _value):
            pass

        def use_deterministic_algorithms(self, _value):
            pass

    def test_cuda_route_sets_a_deterministic_cublas_workspace(self) -> None:
        with contextlib.suppress(Exception):
            configure_deterministic_torch(self._FakeTorch(), "cuda:0")
        self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")
        self.assertEqual(CUBLAS_WORKSPACE_CONFIG, ":4096:8")

    def test_cpu_route_does_not_set_it(self) -> None:
        with contextlib.suppress(Exception):
            configure_deterministic_torch(self._FakeTorch(), "cpu")
        self.assertIsNone(os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

    def test_a_non_deterministic_inherited_value_is_rejected(self) -> None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":2:2"
        with self.assertRaises(ValueError):
            configure_deterministic_torch(self._FakeTorch(), "cuda:0")

    def test_a_valid_inherited_value_is_preserved(self) -> None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        with contextlib.suppress(Exception):
            configure_deterministic_torch(self._FakeTorch(), "cuda:0")
        self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":16:8")


if __name__ == "__main__":
    unittest.main()