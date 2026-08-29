"""Deterministic Stage 2 loading, windowing, and sensor preprocessing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

#: Pillow moved the resampling constants into `Image.Resampling` in 9.1. The ROS
#: host still ships 9.0.1, and its copy of the tests must resample identically.
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC

RGB_SIZE = 224
RGB_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
RGB_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
VISUAL_GRID = 7
LIDAR_SECTORS = 80


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


@dataclass(frozen=True, slots=True)
class WindowRef:
    run_index: int
    context_rows: tuple[int, ...]
    action_rows: tuple[int, ...]

    @property
    def origin_row(self) -> int:
        return self.context_rows[-1]


class ExportRun:
    """Memory-mapped accepted export with strict shape checks.

    Training requires the default policy view. `expected_view="sensor"` exists
    for feature-only uses such as OOD probe recordings, whose action stream is
    intentionally absent; nothing that reads actions may accept a sensor view.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        expected_view: str = "policy",
        load_rgb: bool = True,
    ) -> None:
        if expected_view not in ("policy", "sensor"):
            raise ValueError(f"unknown export view {expected_view!r}")
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text("utf-8"))
        if self.manifest.get("view") != expected_view:
            raise ValueError(f"{self.root} is not a {expected_view} export")
        self.scan_ranges = np.load(self.root / "scan_ranges.npy", mmap_mode="r")
        self.vectors = np.load(self.root / "vectors.npz")
        required = {
            "segment_id",
            "goal",
            "robot_state",
            "action",
            "scan_angle_increment_rad",
            "scan_beam_count",
        }
        missing = required.difference(self.vectors.files)
        if missing:
            raise ValueError(f"{self.root} vectors are missing {sorted(missing)}")
        count = int(self.vectors["action"].shape[0])
        self.rgb = None
        if load_rgb:
            self.rgb = np.load(self.root / "rgb_320x240_rgb8.npy", mmap_mode="r")
            if self.rgb.shape != (count, 240, 320, 3):
                raise ValueError(f"unexpected RGB shape {self.rgb.shape} in {self.root}")
        elif "rgb_320x240_rgb8.npy" not in self.manifest.get("outputs", {}):
            raise ValueError("RGB-light export loading requires a manifested RGB identity")
        if self.scan_ranges.shape[0] != count:
            raise ValueError("scan and vector row counts differ")
        if self.vectors["goal"].shape != (count, 3):
            raise ValueError("goal must have shape [rows, 3]")
        if self.vectors["robot_state"].shape != (count, 2):
            raise ValueError("robot_state must have shape [rows, 2]")
        self.count = count

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    def verify_output_hashes(self) -> dict[str, str]:
        observed: dict[str, str] = {}
        for name, record in self.manifest["outputs"].items():
            digest = sha256_file(self.root / name)
            if digest != record["sha256"]:
                raise ValueError(f"hash mismatch for {self.root / name}")
            observed[name] = digest
        return observed


class WindowDataset:
    """Construct K/H windows without ever crossing an exporter segment."""

    def __init__(
        self,
        roots: list[str | Path],
        context_k: int = 8,
        horizon_h: int = 8,
        *,
        load_rgb: bool = True,
    ):
        if context_k <= 0 or horizon_h <= 0:
            raise ValueError("context_k and horizon_h must be positive")
        self.runs = tuple(ExportRun(root, load_rgb=load_rgb) for root in roots)
        self.context_k = context_k
        self.horizon_h = horizon_h
        windows: list[WindowRef] = []
        for run_index, run in enumerate(self.runs):
            segment_ids = np.asarray(run.vectors["segment_id"])
            for segment_id in np.unique(segment_ids):
                rows = np.flatnonzero(segment_ids == segment_id)
                if rows.size and not np.all(np.diff(rows) == 1):
                    raise ValueError(f"segment {segment_id} is not contiguous in {run.root}")
                for origin in range(context_k - 1, len(rows) - horizon_h + 1):
                    context = rows[origin - context_k + 1 : origin + 1]
                    actions = rows[origin : origin + horizon_h]
                    windows.append(
                        WindowRef(
                            run_index,
                            tuple(int(row) for row in context),
                            tuple(int(row) for row in actions),
                        )
                    )
        self.windows = tuple(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def targets(self, ref: WindowRef) -> np.ndarray:
        return np.asarray(self.runs[ref.run_index].vectors["action"][list(ref.action_rows)])


def preprocess_rgb(image: np.ndarray) -> np.ndarray:
    """Full-FOV 4:3 letterbox followed by ImageNet normalization.

    The 320x240 image becomes 224x168 and receives 28 normalized-zero rows on
    both sides. This avoids cropping calibrated camera coverage and has an exact
    source-pixel to model-pixel mapping for the fusion mask.
    """

    if image.shape != (240, 320, 3) or image.dtype != np.uint8:
        raise ValueError("expected uint8 RGB image with shape [240, 320, 3]")
    resized = np.asarray(Image.fromarray(image).resize((224, 168), BICUBIC))
    normalized = (resized.astype(np.float32) / 255.0 - RGB_MEAN) / RGB_STD
    canvas = np.zeros((RGB_SIZE, RGB_SIZE, 3), dtype=np.float32)
    canvas[28:196] = normalized
    return np.ascontiguousarray(canvas.transpose(2, 0, 1))


def source_to_model_pixels(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return u * (224.0 / 320.0), v * (168.0 / 240.0) + 28.0


def quaternion_matrix_xyzw(quaternion: list[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_from_lidar(manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    transform = manifest["calibration"]["static_transforms"]["base_scan->camera"]
    parent_from_child = quaternion_matrix_xyzw(transform["quaternion_xyzw"])
    translation = np.asarray(transform["translation"], dtype=np.float64)
    child_from_parent = parent_from_child.T
    return child_from_parent, -child_from_parent @ translation


@dataclass(frozen=True, slots=True)
class LidarTokens:
    features: np.ndarray
    visual_mask: np.ndarray
    in_fov: np.ndarray
    sector_bearing_rad: np.ndarray


def _project_points(
    points_lidar: np.ndarray, manifest: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation, translation = camera_from_lidar(manifest)
    camera = points_lidar @ rotation.T + translation
    z = camera[:, 2]
    x = np.divide(camera[:, 0], z, out=np.zeros_like(z), where=z > 0)
    y = np.divide(camera[:, 1], z, out=np.zeros_like(z), where=z > 0)
    calibration = manifest["calibration"]
    k = calibration["recorded_camera_info"]["k"]
    d = calibration["recorded_camera_info"]["d"]
    k1, k2, p1, p2, k3 = (float(value) for value in d)
    radius2 = x * x + y * y
    radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x)
    yd = y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y
    u = float(k[0]) * xd + float(k[2])
    v = float(k[4]) * yd + float(k[5])
    return u, v, z


def tokenize_lidar(
    ranges: np.ndarray,
    beam_count: int,
    angle_increment_rad: float,
    manifest: dict[str, Any],
    *,
    sectors: int = LIDAR_SECTORS,
    range_clip_m: float = 10.0,
    visual_radius: int = 1,
) -> LidarTokens:
    """Aggregate a variable-length scan and build its calibrated patch mask."""

    if not 1 <= beam_count <= len(ranges):
        raise ValueError("beam_count is outside the stored scan payload")
    if sectors < 1 or sectors > beam_count:
        raise ValueError("sectors must be between one and beam_count")
    geometry = manifest["calibration"]["lidar_geometry"]["angular_frame"]
    angle_min = float(geometry["angle_min_rad"])
    range_min = float(geometry["range_min_m"])
    range_max = float(geometry["range_max_m"])
    raw = np.asarray(ranges[:beam_count], dtype=np.float64)
    theta = angle_min + np.arange(beam_count, dtype=np.float64) * angle_increment_rad
    valid = np.isfinite(raw) & (raw >= range_min) & (raw <= range_max)
    features = np.empty((sectors, 4), dtype=np.float32)
    bearings = np.empty(sectors, dtype=np.float64)
    representative_range = np.empty(sectors, dtype=np.float64)
    for sector in range(sectors):
        start = sector * beam_count // sectors
        end = (sector + 1) * beam_count // sectors
        sector_valid = valid[start:end]
        fraction = float(np.mean(sector_valid))
        bearing = float(np.mean(theta[start:end]))
        if np.any(sector_valid):
            measured = raw[start:end][sector_valid]
            normalized_range = float(np.mean(np.minimum(measured, range_clip_m)) / range_clip_m)
            representative_range[sector] = float(np.mean(measured))
        else:
            normalized_range = 1.0
            representative_range[sector] = range_clip_m
        features[sector] = (normalized_range, math.sin(bearing), math.cos(bearing), fraction)
        bearings[sector] = bearing

    points = np.column_stack(
        (
            representative_range * np.cos(bearings),
            representative_range * np.sin(bearings),
            np.zeros(sectors),
        )
    )
    u, v, depth = _project_points(points, manifest)
    in_fov = (depth > 0) & (u >= 0) & (u < 320) & (v >= 0) & (v < 240)
    model_u, model_v = source_to_model_pixels(u, v)
    grid_x = np.zeros(sectors, dtype=int)
    grid_y = np.zeros(sectors, dtype=int)
    grid_x[in_fov] = np.clip(
        (model_u[in_fov] / (RGB_SIZE / VISUAL_GRID)).astype(int), 0, VISUAL_GRID - 1
    )
    grid_y[in_fov] = np.clip(
        (model_v[in_fov] / (RGB_SIZE / VISUAL_GRID)).astype(int), 0, VISUAL_GRID - 1
    )
    grid = np.zeros((sectors, VISUAL_GRID, VISUAL_GRID), dtype=bool)
    for sector in np.flatnonzero(in_fov):
        top = max(0, grid_y[sector] - visual_radius)
        bottom = min(VISUAL_GRID, grid_y[sector] + visual_radius + 1)
        left = max(0, grid_x[sector] - visual_radius)
        right = min(VISUAL_GRID, grid_x[sector] + visual_radius + 1)
        grid[sector, top:bottom, left:right] = True
    mask = grid.reshape(sectors, VISUAL_GRID * VISUAL_GRID)
    return LidarTokens(features, mask, in_fov.astype(bool), bearings.astype(np.float32))
