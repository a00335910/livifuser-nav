"""Measure continuous ROS camera/LiDAR timestamp and arrival-time behavior."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan

SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=20,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def stamp_ns(message: Image | LaserScan) -> int:
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return math.nan
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def statistics_ms(values_ns: list[int]) -> dict[str, float | int]:
    values_ms = sorted(value / 1_000_000 for value in values_ns)
    if not values_ms:
        return {"count": 0}
    return {
        "count": len(values_ms),
        "min_ms": values_ms[0],
        "median_ms": percentile(values_ms, 0.5),
        "mean_ms": sum(values_ms) / len(values_ms),
        "p95_ms": percentile(values_ms, 0.95),
        "max_ms": values_ms[-1],
    }


def interval_statistics(stamps: list[int]) -> dict[str, float | int]:
    ordered = sorted(stamps)
    intervals = [later - earlier for earlier, later in zip(ordered, ordered[1:], strict=False)]
    result = statistics_ms(intervals)
    positive = [interval for interval in intervals if interval > 0]
    if positive:
        result["median_hz"] = 1_000_000_000 / percentile(sorted(positive), 0.5)
    result["nonpositive_intervals"] = sum(interval <= 0 for interval in intervals)
    return result


def nearest_offsets(source_stamps: list[int], target_stamps: list[int]) -> list[int]:
    ordered_targets = sorted(target_stamps)
    if not ordered_targets:
        return []
    offsets = []
    for source in source_stamps:
        index = bisect.bisect_left(ordered_targets, source)
        candidates = []
        if index < len(ordered_targets):
            candidates.append(abs(ordered_targets[index] - source))
        if index > 0:
            candidates.append(abs(ordered_targets[index - 1] - source))
        offsets.append(min(candidates))
    return offsets


def previous_offsets(source_stamps: list[int], target_stamps: list[int]) -> list[int]:
    ordered_targets = sorted(target_stamps)
    offsets = []
    for source in source_stamps:
        index = bisect.bisect_right(ordered_targets, source) - 1
        if index >= 0:
            offsets.append(source - ordered_targets[index])
    return offsets


class TimingSampler(Node):
    def __init__(self) -> None:
        super().__init__("livifuser_sensor_timing_sampler")
        self.images: list[tuple[int, int]] = []
        self.scans: list[tuple[int, int]] = []
        self.scan_metadata: dict[str, float | int | str] = {}
        self.create_subscription(Image, "/camera/image_raw", self._on_image, SENSOR_QOS)
        self.create_subscription(LaserScan, "/scan", self._on_scan, SENSOR_QOS)

    def _on_image(self, message: Image) -> None:
        self.images.append((stamp_ns(message), self.get_clock().now().nanoseconds))

    def _on_scan(self, message: LaserScan) -> None:
        self.scans.append((stamp_ns(message), self.get_clock().now().nanoseconds))
        if not self.scan_metadata:
            self.scan_metadata = {
                "frame_id": message.header.frame_id,
                "beam_count": len(message.ranges),
                "time_increment_sec": message.time_increment,
                "scan_time_sec": message.scan_time,
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-sec", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rclpy.init()
    node = TimingSampler()
    deadline = time.monotonic() + args.duration_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    images = node.images
    scans = node.scans
    scan_metadata = node.scan_metadata
    node.destroy_node()
    rclpy.shutdown()

    image_stamps = [header for header, _ in images]
    scan_stamps = [header for header, _ in scans]
    if not image_stamps or not scan_stamps:
        raise RuntimeError(
            f"insufficient samples: images={len(image_stamps)} scans={len(scan_stamps)}"
        )
    overlap_start = max(min(image_stamps), min(scan_stamps))
    overlap_end = min(max(image_stamps), max(scan_stamps))
    overlap_images = [stamp for stamp in image_stamps if overlap_start <= stamp <= overlap_end]
    # Keep all scan samples when associating boundary images. Restricting scans
    # to the camera overlap can discard the true nearest bracketing scan.
    nearest = nearest_offsets(overlap_images, scan_stamps)
    previous = previous_offsets(overlap_images, scan_stamps)

    record = {
        "duration_sec": args.duration_sec,
        "image_samples": len(images),
        "scan_samples": len(scans),
        "image_header_intervals": interval_statistics(image_stamps),
        "scan_header_intervals": interval_statistics(scan_stamps),
        "image_arrival_minus_header": statistics_ms(
            [arrival - header for header, arrival in images]
        ),
        "scan_arrival_minus_header": statistics_ms(
            [arrival - header for header, arrival in scans]
        ),
        "nearest_scan_for_each_image": {
            **statistics_ms(nearest),
            "within_75_ms_percent": 100.0
            * sum(offset <= 75_000_000 for offset in nearest)
            / len(nearest),
        },
        "latest_previous_scan_for_each_image": statistics_ms(previous),
        "scan_metadata": scan_metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
