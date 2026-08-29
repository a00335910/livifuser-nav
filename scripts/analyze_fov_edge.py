#!/usr/bin/env python3
"""Project captured LDS-03 scans into camera frames for the physical FOV edge gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--extrinsics", type=Path, required=True)
    return parser.parse_args()


def project_scan(
    ranges: list[float | None],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[dict[str, float | int]]:
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    k1, k2, p1, p2, k3 = (float(value) for value in distortion)
    projected = []

    for beam_index, raw_range in enumerate(ranges):
        if raw_range is None:
            continue
        distance = float(raw_range)
        if not math.isfinite(distance) or not range_min <= distance <= range_max:
            continue
        angle = angle_min + beam_index * angle_increment
        point_lidar = np.array([distance * math.cos(angle), distance * math.sin(angle), 0.0])
        point_camera = rotation @ point_lidar + translation
        x_cam, y_cam, z_cam = (float(value) for value in point_camera)
        if z_cam <= 0.0:
            continue

        x_norm = x_cam / z_cam
        y_norm = y_cam / z_cam
        radius_2 = x_norm * x_norm + y_norm * y_norm
        radial = 1.0 + k1 * radius_2 + k2 * radius_2**2 + k3 * radius_2**3
        x_distorted = (
            x_norm * radial + 2.0 * p1 * x_norm * y_norm + p2 * (radius_2 + 2.0 * x_norm**2)
        )
        y_distorted = (
            y_norm * radial + p1 * (radius_2 + 2.0 * y_norm**2) + 2.0 * p2 * x_norm * y_norm
        )
        projected.append(
            {
                "beam_index": beam_index,
                "lidar_bearing_deg": math.degrees(angle),
                "range_m": distance,
                "camera_depth_m": z_cam,
                "u_px": fx * x_distorted + cx,
                "v_px": fy * y_distorted + cy,
            }
        )
    return projected


def color_for_range(distance: float) -> tuple[int, int, int]:
    scaled = int(np.clip((distance - 0.1) / 3.0 * 255.0, 0.0, 255.0))
    return 255 - scaled, scaled, 40


def analyze_capture(
    capture_dir: Path,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> None:
    pair_path = capture_dir / "pair.json"
    image_path = capture_dir / "image_raw.png"
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {image_path}")

    scan = pair["scan"]
    projected = project_scan(
        scan["ranges"],
        float(scan["angle_min"]),
        float(scan["angle_increment"]),
        float(scan["range_min"]),
        float(scan["range_max"]),
        rotation,
        translation,
        camera_matrix,
        distortion,
    )
    height, width = image.shape[:2]
    in_frame = [
        point
        for point in projected
        if 0.0 <= float(point["u_px"]) < width and 0.0 <= float(point["v_px"]) < height
    ]

    overlay = image.copy()
    for point in in_frame:
        u_px = int(round(float(point["u_px"])))
        v_px = int(round(float(point["v_px"])))
        color = color_for_range(float(point["range_m"]))
        cv2.circle(overlay, (u_px, v_px), 3, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (u_px, v_px), 4, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(
        overlay,
        f"projected LDS-03 returns: {len(in_frame)}",
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    output_image = capture_dir / "lidar_projection_overlay.png"
    output_json = capture_dir / "lidar_projection.json"
    if not cv2.imwrite(str(output_image), overlay):
        raise OSError(f"Could not write {output_image}")
    output_json.write_text(
        json.dumps(
            {
                "projection_model": "full K + plumb_bob distortion + accepted camera_from_lidar",
                "lidar_bearing_convention": (
                    "REP-103 base_scan; zero=robot-forward, positive=robot-left "
                    "(counter-clockwise about +z)"
                ),
                "capture_pair_delta_ms": pair["absolute_delta_ms"],
                "valid_forward_projected_returns": len(projected),
                "in_frame_returns": len(in_frame),
                "in_frame": in_frame,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{capture_dir}: {len(in_frame)} in-frame returns -> {output_image}")


def main() -> None:
    args = parse_args()
    intrinsics = yaml.safe_load(args.intrinsics.read_text(encoding="utf-8"))
    extrinsics = yaml.safe_load(args.extrinsics.read_text(encoding="utf-8"))
    camera_matrix = np.asarray(intrinsics["camera_matrix"]["data"], dtype=float).reshape(3, 3)
    distortion = np.asarray(intrinsics["distortion_coefficients"]["data"], dtype=float)
    camera_from_lidar = extrinsics["camera_from_lidar"]
    rotation = np.asarray(camera_from_lidar["rotation_matrix"], dtype=float)
    translation = np.asarray(camera_from_lidar["translation_m"], dtype=float)

    for capture_dir in args.captures:
        analyze_capture(
            capture_dir,
            rotation,
            translation,
            camera_matrix,
            distortion,
        )


if __name__ == "__main__":
    main()
