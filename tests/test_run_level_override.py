"""The calibration override must be narrow.

`--allow-calibration-mismatch` exists so an operator can proceed when recorded
CameraInfo disagrees with the accepted intrinsics. It must not become a general
"ignore all run-level faults" switch: a timestamp regression, a missing
transform, or a mid-run LiDAR geometry change are different faults and would
silently corrupt a dataset if excused.
"""

import unittest

from livifuser_nav.export_schema import (
    OVERRIDABLE_RUN_LEVEL_CODES,
    RejectionCode,
    apply_run_level_override,
)
from livifuser_nav.sampling import assemble_samples
from tests.test_sampling import series

NON_OVERRIDABLE = (
    RejectionCode.TIMESTAMP_REGRESSION,
    RejectionCode.TF_UNAVAILABLE,
    RejectionCode.LIDAR_PAYLOAD_INVALID,
    RejectionCode.CAMERA_PAYLOAD_INVALID,
)


class OverrideScopeTests(unittest.TestCase):
    def test_only_calibration_mismatch_is_overridable(self) -> None:
        self.assertEqual(
            OVERRIDABLE_RUN_LEVEL_CODES, frozenset({RejectionCode.CALIBRATION_MISMATCH})
        )

    def test_override_disabled_retains_everything(self) -> None:
        codes = [RejectionCode.CALIBRATION_MISMATCH, RejectionCode.TF_UNAVAILABLE]
        retained, downgraded = apply_run_level_override(
            codes, allow_calibration_mismatch=False
        )
        self.assertEqual(retained, tuple(codes))
        self.assertEqual(downgraded, ())

    def test_multiple_simultaneous_codes_lose_only_the_calibration_one(self) -> None:
        codes = [
            RejectionCode.TIMESTAMP_REGRESSION,
            RejectionCode.CALIBRATION_MISMATCH,
            RejectionCode.TF_UNAVAILABLE,
            RejectionCode.LIDAR_PAYLOAD_INVALID,
        ]
        retained, downgraded = apply_run_level_override(
            codes, allow_calibration_mismatch=True
        )
        self.assertEqual(downgraded, (RejectionCode.CALIBRATION_MISMATCH,))
        self.assertEqual(
            retained,
            (
                RejectionCode.TIMESTAMP_REGRESSION,
                RejectionCode.TF_UNAVAILABLE,
                RejectionCode.LIDAR_PAYLOAD_INVALID,
            ),
        )
        self.assertNotIn(RejectionCode.CALIBRATION_MISMATCH, retained)

    def test_each_non_overridable_code_survives_alone(self) -> None:
        for code in NON_OVERRIDABLE:
            with self.subTest(code=code.value):
                retained, downgraded = apply_run_level_override(
                    [code], allow_calibration_mismatch=True
                )
                self.assertEqual(retained, (code,))
                self.assertEqual(downgraded, ())

    def test_calibration_only_clears_completely(self) -> None:
        retained, downgraded = apply_run_level_override(
            [RejectionCode.CALIBRATION_MISMATCH], allow_calibration_mismatch=True
        )
        self.assertEqual(retained, ())
        self.assertEqual(downgraded, (RejectionCode.CALIBRATION_MISMATCH,))

    def test_duplicates_are_collapsed(self) -> None:
        retained, downgraded = apply_run_level_override(
            [
                RejectionCode.TF_UNAVAILABLE,
                RejectionCode.TF_UNAVAILABLE,
                RejectionCode.CALIBRATION_MISMATCH,
                RejectionCode.CALIBRATION_MISMATCH,
            ],
            allow_calibration_mismatch=True,
        )
        self.assertEqual(retained, (RejectionCode.TF_UNAVAILABLE,))
        self.assertEqual(downgraded, (RejectionCode.CALIBRATION_MISMATCH,))

    def test_empty_input_is_stable(self) -> None:
        self.assertEqual(apply_run_level_override([], allow_calibration_mismatch=True), ((), ()))


class OverrideEndToEndTests(unittest.TestCase):
    """The retained codes must still reject every sample."""

    def _assemble(self, codes: tuple[RejectionCode, ...]):
        return assemble_samples(
            camera=series(0, 100),
            lidar=series(0, 100),
            odometry=series(0, 100),
            goal=series(0, 100),
            action=series(0, 100),
            run_level_codes=codes,
        )

    def test_downgrading_calibration_alone_restores_all_samples(self) -> None:
        retained, _ = apply_run_level_override(
            [RejectionCode.CALIBRATION_MISMATCH], allow_calibration_mismatch=True
        )
        self.assertEqual(len(self._assemble(retained).accepted), 2)

    def test_a_surviving_fault_still_rejects_every_sample(self) -> None:
        retained, downgraded = apply_run_level_override(
            [RejectionCode.CALIBRATION_MISMATCH, RejectionCode.TIMESTAMP_REGRESSION],
            allow_calibration_mismatch=True,
        )
        result = self._assemble(retained)
        self.assertEqual(len(result.accepted), 0)
        self.assertEqual(
            result.rejection_counts(), {RejectionCode.TIMESTAMP_REGRESSION.value: 2}
        )
        # The downgraded code must not reappear anywhere in the rejections.
        self.assertNotIn(
            RejectionCode.CALIBRATION_MISMATCH.value, result.all_rejection_counts()
        )
        self.assertEqual(downgraded, (RejectionCode.CALIBRATION_MISMATCH,))

    def test_geometry_fault_is_not_excused_by_the_calibration_override(self) -> None:
        retained, _ = apply_run_level_override(
            [RejectionCode.CALIBRATION_MISMATCH, RejectionCode.LIDAR_PAYLOAD_INVALID],
            allow_calibration_mismatch=True,
        )
        result = self._assemble(retained)
        self.assertEqual(len(result.accepted), 0)
        self.assertEqual(
            result.rejection_counts(), {RejectionCode.LIDAR_PAYLOAD_INVALID.value: 2}
        )


if __name__ == "__main__":
    unittest.main()
