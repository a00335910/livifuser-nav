"""Solve the LDS-03 to camera optical-frame extrinsic from planar board poses."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# WSL has a newer /usr/local NumPy/OpenCV stack that is ABI-incompatible with
# Ubuntu 22.04's SciPy. Select the mutually compatible Ubuntu packages.
sys.path = [path for path in sys.path if path != "/usr/local/lib/python3.10/dist-packages"]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

LIDAR_TO_OPTICAL_NOMINAL = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ]
)


@dataclass
class PoseObservation:
    name: str
    split: str
    directory: Path
    normal_camera: np.ndarray
    plane_offset_camera: float
    lidar_points: np.ndarray
    pnp_reprojection_rmse_px: float


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def checkerboard_object_points(columns: int, rows: int, square_size_m: float) -> np.ndarray:
    points = np.zeros((columns * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size_m
    return points


def detect_corners(gray: np.ndarray, pattern: tuple[int, int]) -> tuple[str, np.ndarray]:
    detected, corners = cv2.findChessboardCorners(gray, pattern)
    if detected:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        # A 5x5 half-window crosses neighboring squares in the far 320x240 views.
        # Three pixels remains sub-pixel stable across the captured target scales.
        refined = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
        return "classic", refined

    detected, corners = cv2.findChessboardCornersSB(gray, pattern)
    if detected:
        return "sb", corners
    raise ValueError("checkerboard was not detected")


def normalized_angle_deg(angle_rad: float) -> float:
    return math.degrees((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


def angle_selected(angle_deg: float, intervals: list[list[float]]) -> bool:
    return any(lower <= angle_deg <= upper for lower, upper in intervals)


def load_observation(
    root: Path,
    pose_config: dict[str, Any],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    object_points: np.ndarray,
    pattern: tuple[int, int],
) -> PoseObservation:
    directory = root / pose_config["path"]
    image = cv2.imread(str(directory / "image_raw.png"))
    if image is None:
        raise FileNotFoundError(directory / "image_raw.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, corners = detect_corners(gray, pattern)
    solved, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        corners,
        camera_matrix,
        distortion,
    )
    if not solved:
        raise RuntimeError(f"PnP failed for {pose_config['name']}")

    board_rotation = cv2.Rodrigues(rotation_vector)[0]
    normal = board_rotation[:, 2]
    plane_offset = -float(normal @ translation_vector[:, 0])
    if normal[2] < 0.0:
        normal = -normal
        plane_offset = -plane_offset

    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion,
    )
    reprojection_errors = np.linalg.norm(
        projected.reshape(-1, 2) - corners.reshape(-1, 2),
        axis=1,
    )

    pair = json.loads((directory / "pair.json").read_text(encoding="utf-8"))
    scan = pair["scan"]
    intervals = pose_config["lidar_angle_intervals_deg"]
    lidar_points = []
    for index, distance in enumerate(scan["ranges"]):
        if distance is None or not math.isfinite(distance):
            continue
        if not scan["range_min"] <= distance <= scan["range_max"]:
            continue
        angle = scan["angle_min"] + index * scan["angle_increment"]
        angle_deg = normalized_angle_deg(angle)
        if not angle_selected(angle_deg, intervals):
            continue
        lidar_points.append(
            [
                distance * math.cos(angle),
                distance * math.sin(angle),
                0.0,
            ]
        )

    if len(lidar_points) < 5:
        raise ValueError(f"too few selected LiDAR points for {pose_config['name']}")

    return PoseObservation(
        name=pose_config["name"],
        split=pose_config["split"],
        directory=directory,
        normal_camera=normal,
        plane_offset_camera=plane_offset,
        lidar_points=np.asarray(lidar_points),
        pnp_reprojection_rmse_px=float(np.sqrt(np.mean(reprojection_errors**2))),
    )


def transform_from_parameters(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta_rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
    rotation_camera_from_lidar = delta_rotation @ LIDAR_TO_OPTICAL_NOMINAL
    translation_camera_from_lidar = parameters[3:]
    return rotation_camera_from_lidar, translation_camera_from_lidar


def physical_residuals(
    parameters: np.ndarray,
    observations: list[PoseObservation],
) -> list[np.ndarray]:
    rotation, translation = transform_from_parameters(parameters)
    residuals = []
    for observation in observations:
        points_camera = (rotation @ observation.lidar_points.T).T + translation
        residuals.append(
            points_camera @ observation.normal_camera + observation.plane_offset_camera
        )
    return residuals


def weighted_residual_vector(
    parameters: np.ndarray,
    observations: list[PoseObservation],
) -> np.ndarray:
    residuals = physical_residuals(parameters, observations)
    mean_count = np.mean([len(values) for values in residuals])
    balanced = [values * math.sqrt(mean_count / len(values)) for values in residuals]
    return np.concatenate(balanced)


def solve(observations: list[PoseObservation]) -> tuple[np.ndarray, Any]:
    lower = np.array([-0.8, -0.8, -0.8, -0.5, -0.5, -0.5])
    upper = np.array([0.8, 0.8, 0.8, 0.5, 0.5, 0.5])
    starts = [np.zeros(6)]
    generator = np.random.default_rng(20260729)
    for _ in range(15):
        start = np.zeros(6)
        start[:3] = generator.normal(0.0, 0.12, 3)
        start[3:] = generator.normal(0.0, 0.06, 3)
        starts.append(np.clip(start, lower + 1e-6, upper - 1e-6))

    solutions = [
        least_squares(
            weighted_residual_vector,
            start,
            args=(observations,),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.01,
            max_nfev=5000,
            x_scale="jac",
        )
        for start in starts
    ]
    best = min(solutions, key=lambda solution: np.sum(solution.fun**2))
    return best.x, best


def error_statistics(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "rmse_m": float(np.sqrt(np.mean(values**2))),
        "median_abs_m": float(np.median(absolute)),
        "p95_abs_m": float(np.percentile(absolute, 95)),
        "max_abs_m": float(np.max(absolute)),
    }


def evaluate(
    parameters: np.ndarray,
    observations: list[PoseObservation],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    pose_residuals = physical_residuals(parameters, observations)
    per_pose = {
        observation.name: {
            **error_statistics(residual),
            "lidar_points": len(residual),
            "pnp_reprojection_rmse_px": observation.pnp_reprojection_rmse_px,
        }
        for observation, residual in zip(observations, pose_residuals, strict=True)
    }
    overall = error_statistics(np.concatenate(pose_residuals))
    return per_pose, overall


def transform_record(parameters: np.ndarray) -> dict[str, Any]:
    rotation_camera_from_lidar, translation_camera_from_lidar = transform_from_parameters(
        parameters
    )
    rotation_lidar_from_camera = rotation_camera_from_lidar.T
    translation_lidar_from_camera = -rotation_lidar_from_camera @ translation_camera_from_lidar
    quaternion_camera_in_lidar = Rotation.from_matrix(rotation_lidar_from_camera).as_quat()
    return {
        "coordinate_mapping": {
            "equation": (
                "p_camera_optical = R_camera_from_lidar * p_base_scan "
                "+ t_camera_from_lidar"
            ),
            "rotation_matrix": rotation_camera_from_lidar.tolist(),
            "translation_m": translation_camera_from_lidar.tolist(),
        },
        "ros_static_tf_parent_to_child": {
            "parent_frame": "base_scan",
            "child_frame": "camera_optical_frame",
            "translation_m": translation_lidar_from_camera.tolist(),
            "rotation_quaternion_xyzw": quaternion_camera_in_lidar.tolist(),
        },
    }


def render_overlays(
    output: Path,
    parameters: np.ndarray,
    observations: list[PoseObservation],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rotation, translation = transform_from_parameters(parameters)
    rotation_vector = cv2.Rodrigues(rotation)[0]
    for observation in observations:
        image = cv2.imread(str(observation.directory / "image_raw.png"))
        projected, _ = cv2.projectPoints(
            observation.lidar_points,
            rotation_vector,
            translation.reshape(3, 1),
            camera_matrix,
            distortion,
        )
        for pixel in projected.reshape(-1, 2):
            x, y = (int(round(value)) for value in pixel)
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        cv2.imwrite(str(output / f"{observation.name}_projection.png"), image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    selection = load_yaml(root / args.selection)
    intrinsics = load_yaml(root / selection["camera_intrinsics"])
    camera_matrix = np.asarray(intrinsics["camera_matrix"]["data"], dtype=float).reshape(3, 3)
    distortion = np.asarray(intrinsics["distortion_coefficients"]["data"], dtype=float)
    board = selection["checkerboard"]
    pattern = (board["columns"], board["rows"])
    object_points = checkerboard_object_points(
        board["columns"],
        board["rows"],
        board["square_size_m"],
    )
    excluded = set(args.exclude)
    observations = [
        load_observation(
            root,
            pose,
            camera_matrix,
            distortion,
            object_points,
            pattern,
        )
        for pose in selection["poses"]
        if pose["name"] not in excluded
    ]
    training = [observation for observation in observations if observation.split == "train"]
    validation = [
        observation for observation in observations if observation.split == "validation"
    ]

    training_parameters, training_solution = solve(training)
    training_per_pose, training_overall = evaluate(training_parameters, training)
    validation_per_pose, validation_overall = evaluate(training_parameters, validation)

    final_parameters, final_solution = solve(observations)
    final_per_pose, final_overall = evaluate(final_parameters, observations)
    singular_values = np.linalg.svd(final_solution.jac, compute_uv=False)

    final_rotation, final_translation = transform_from_parameters(final_parameters)
    leave_one_out = {}
    leave_one_out_residuals = []
    for held_out in observations:
        remaining = [observation for observation in observations if observation is not held_out]
        held_out_parameters, _ = solve(remaining)
        held_out_residual = physical_residuals(held_out_parameters, [held_out])[0]
        held_out_rotation, held_out_translation = transform_from_parameters(held_out_parameters)
        rotation_difference = Rotation.from_matrix(
            held_out_rotation @ final_rotation.T
        ).magnitude()
        leave_one_out[held_out.name] = {
            **error_statistics(held_out_residual),
            "translation_sensitivity_m": float(
                np.linalg.norm(held_out_translation - final_translation)
            ),
            "rotation_sensitivity_deg": float(math.degrees(rotation_difference)),
        }
        leave_one_out_residuals.append(held_out_residual)

    record = {
        "method": (
            "checkerboard plane PnP plus manually selected 2D LiDAR "
            "point-to-plane least squares"
        ),
        "selection": str(args.selection),
        "excluded_pose_names": sorted(excluded),
        "training_pose_names": [observation.name for observation in training],
        "validation_pose_names": [observation.name for observation in validation],
        "training_fit": {
            "optimizer_success": bool(training_solution.success),
            "training_overall": training_overall,
            "training_per_pose": training_per_pose,
            "held_out_validation_overall": validation_overall,
            "held_out_validation_per_pose": validation_per_pose,
            **transform_record(training_parameters),
        },
        "final_all_pose_fit": {
            "optimizer_success": bool(final_solution.success),
            "overall": final_overall,
            "per_pose": final_per_pose,
            "jacobian_singular_values": singular_values.tolist(),
            "jacobian_condition_number": float(singular_values[0] / singular_values[-1]),
            **transform_record(final_parameters),
        },
        "leave_one_pose_out": {
            "overall": error_statistics(np.concatenate(leave_one_out_residuals)),
            "per_pose": leave_one_out,
        },
    }

    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    render_overlays(
        output / "overlays",
        final_parameters,
        observations,
        camera_matrix,
        distortion,
    )

    print(json.dumps(record["training_fit"]["held_out_validation_overall"], indent=2))
    print(json.dumps(record["final_all_pose_fit"]["overall"], indent=2))
    print(json.dumps(record["final_all_pose_fit"]["ros_static_tf_parent_to_child"], indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
