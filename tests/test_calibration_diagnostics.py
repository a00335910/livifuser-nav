from __future__ import annotations

import unittest

from livifuser_nav.calibration_diagnostics import camera_fov_degrees

ACCEPTED_K = [
    316.21156,
    0.0,
    223.13834,
    0.0,
    315.6497,
    107.39364,
    0.0,
    0.0,
    1.0,
]


class CameraDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.diagnostic = camera_fov_degrees(ACCEPTED_K, 320, 240)

    def test_manifest_declares_camera_optical_sign_convention(self) -> None:
        convention = self.diagnostic["bearing_convention"]
        self.assertIn("camera optical frame", convention)
        self.assertIn("positive toward +x/image-right", convention)
        self.assertIn("positive toward +y/image-down", convention)

    def test_asymmetric_boundary_values_are_camera_optical(self) -> None:
        image_left, image_right = self.diagnostic["horizontal_bearing_range_deg"]
        self.assertAlmostEqual(image_left, -35.21, delta=0.01)
        self.assertAlmostEqual(image_right, 17.03, delta=0.01)

        # REP-103 base_scan has positive bearing toward robot-left, opposite to
        # camera optical +x. The physical right-edge return was -17.06 degrees.
        image_right_body_bearing = -image_right
        self.assertAlmostEqual(image_right_body_bearing, -17.03, delta=0.01)
        self.assertAlmostEqual(image_right_body_bearing, -17.06, delta=0.04)

    def test_diagnostic_disclaims_physical_fov_measurement(self) -> None:
        interpretation = self.diagnostic["interpretation"]
        self.assertIn("not a physical FOV measurement", interpretation)
        self.assertIn("0 <= u < width", interpretation)
        self.assertIn("opposite left/right sign", interpretation)

    def test_continuous_boundary_definition_is_explicit(self) -> None:
        definition = self.diagnostic["image_boundary_definition"]
        self.assertIn("u=[0,width]", definition)
        self.assertIn("0 <= u < width", definition)


if __name__ == "__main__":
    unittest.main()
