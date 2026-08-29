#!/usr/bin/env python3
"""Summarize and extract frames from an excluded visual-probe MCAP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

IMAGE_TOPIC = "/camera/image_raw"
TRUTH_TOPIC = "/livifuser/sim/ground_truth/odom"
SAMPLE_INDICES = {0, 100, 200, 300, 400}


def _rgb_array(message: Image) -> np.ndarray:
    if message.encoding != "rgb8":
        raise ValueError(f"expected rgb8, received {message.encoding!r}")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    rows = raw.reshape(message.height, message.step)
    return rows[:, : message.width * 3].reshape(message.height, message.width, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing summary directory: {args.output}")
    args.output.mkdir(parents=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )

    hashes: list[str] = []
    frame_differences: list[float] = []
    selected_frames: dict[int, np.ndarray] = {}
    previous: np.ndarray | None = None
    last_frame: np.ndarray | None = None
    first_truth: tuple[float, float] | None = None
    last_truth: tuple[float, float] | None = None

    while reader.has_next():
        topic, serialized, _timestamp = reader.read_next()
        if topic == IMAGE_TOPIC:
            message = deserialize_message(serialized, Image)
            frame = _rgb_array(message)
            index = len(hashes)
            hashes.append(hashlib.sha256(frame.tobytes()).hexdigest().upper())
            if previous is not None:
                difference = np.abs(
                    frame.astype(np.int16) - previous.astype(np.int16)
                ).mean() / 255.0
                frame_differences.append(float(difference))
            if index in SAMPLE_INDICES:
                selected_frames[index] = frame.copy()
            previous = frame.copy()
            last_frame = frame.copy()
        elif topic == TRUTH_TOPIC:
            message = deserialize_message(serialized, Odometry)
            pose = message.pose.pose.position
            xy = (float(pose.x), float(pose.y))
            if first_truth is None:
                first_truth = xy
            last_truth = xy

    if not hashes or first_truth is None or last_truth is None or last_frame is None:
        raise SystemExit("probe bag lacks RGB frames or ground-truth poses")
    selected_frames[len(hashes) - 1] = last_frame
    for index, frame in sorted(selected_frames.items()):
        target = args.output / f"frame_{index:04d}.png"
        if not cv2.imwrite(str(target), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to write {target}")

    counts = Counter(hashes)
    displacement = math.hypot(
        last_truth[0] - first_truth[0], last_truth[1] - first_truth[1]
    )
    summary = {
        "status": "EXCLUDED_VISUAL_COMPATIBILITY_PROBE",
        "confirmatory_use_permitted": False,
        "image_count": len(hashes),
        "unique_image_count": len(counts),
        "modal_image_fraction": max(counts.values()) / len(hashes),
        "consecutive_rgb_mad_mean_normalized": float(np.mean(frame_differences)),
        "consecutive_rgb_mad_median_normalized": float(np.median(frame_differences)),
        "ground_truth_start_xy_m": list(first_truth),
        "ground_truth_end_xy_m": list(last_truth),
        "ground_truth_displacement_m": displacement,
        "extracted_frame_indices": sorted(selected_frames),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
