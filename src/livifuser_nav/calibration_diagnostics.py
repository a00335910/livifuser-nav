"""Frame-explicit calibration diagnostics that are safe to test without ROS."""

from __future__ import annotations

import math
from typing import Any

CAMERA_OPTICAL_BEARING_CONVENTION = (
    "camera optical frame (x right, y down, z forward); horizontal bearing is "
    "positive toward +x/image-right and vertical bearing is positive toward "
    "+y/image-down"
)


def camera_fov_degrees(
    camera_matrix: list[float],
    width: int,
    height: int,
) -> dict[str, Any]:
    """Summarize intrinsics as a frame-explicit diagnostic image envelope.

    This diagnostic does not measure the physical field of view and must never
    replace the full distorted camera/LiDAR projection used by the fusion mask.
    """

    fx, _, cx, _, fy, cy, *_ = camera_matrix
    # These are continuous image-plane boundaries at u=0/width and v=0/height,
    # not the centres of the final pixels at width-1/height-1.
    left_deg = math.degrees(math.atan((0.0 - cx) / fx))
    right_deg = math.degrees(math.atan((width - cx) / fx))
    top_deg = math.degrees(math.atan((0.0 - cy) / fy))
    bottom_deg = math.degrees(math.atan((height - cy) / fy))
    return {
        "bearing_convention": CAMERA_OPTICAL_BEARING_CONVENTION,
        "image_boundary_definition": (
            "continuous boundaries u=[0,width], v=[0,height]; inclusion remains "
            "0 <= u < width and 0 <= v < height"
        ),
        "horizontal_fov_deg": right_deg - left_deg,
        "vertical_fov_deg": bottom_deg - top_deg,
        "horizontal_bearing_range_deg": [left_deg, right_deg],
        "vertical_bearing_range_deg": [top_deg, bottom_deg],
        "horizontal_symmetric_fov_deg": math.degrees(2.0 * math.atan(width / (2.0 * fx))),
        "principal_point_offset_px": [cx - width / 2.0, cy - height / 2.0],
        "principal_point_offset_fraction": [
            (cx - width / 2.0) / width,
            (cy - height / 2.0) / height,
        ],
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "interpretation": (
            "DIAGNOSTIC ONLY. These camera-optical bearings are a first-order "
            "continuous image envelope and ignore lens distortion entirely. They "
            "are not a physical FOV measurement. The section 3.2 mask must project "
            "each LiDAR point through camera-from-LiDAR extrinsics and full "
            "intrinsics with distortion, then test 0 <= u < width and 0 <= v < "
            "height. Do not substitute this range for that projection. REP-103 "
            "base_scan horizontal bearing has the opposite left/right sign."
        ),
        "asymmetry_hypothesis": (
            "The physical edge test corroborates the asymmetric projection but "
            "does not measure the FOV extent or prove that an off-centre sensor "
            "crop caused the principal-point offset."
        ),
    }
