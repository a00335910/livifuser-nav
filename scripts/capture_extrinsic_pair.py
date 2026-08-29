"""Capture one synchronized raw-camera/LaserScan pair for extrinsic calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan

LATEST_SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def stamp_ns(message: Image | LaserScan) -> int:
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


def decode_image(message: Image) -> np.ndarray:
    channels_by_encoding = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
    }
    if message.encoding not in channels_by_encoding:
        raise ValueError(f"unsupported image encoding: {message.encoding}")

    channels = channels_by_encoding[message.encoding]
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    pixels = rows[:, : message.width * channels].reshape(message.height, message.width, channels)

    if message.encoding == "rgb8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if message.encoding == "rgba8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    if message.encoding == "bgra8":
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    if message.encoding == "mono8":
        return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    return pixels.copy()


class PairCapture(Node):
    def __init__(self, max_delta_ns: int) -> None:
        super().__init__("livifuser_extrinsic_pair_capture")
        self.max_delta_ns = max_delta_ns
        self.image: Image | None = None
        self.scan: LaserScan | None = None
        self.pair: tuple[Image, LaserScan] | None = None
        self.create_subscription(
            Image,
            "/camera/image_raw",
            self._on_image,
            LATEST_SENSOR_QOS,
        )
        self.create_subscription(
            LaserScan,
            "/scan",
            self._on_scan,
            LATEST_SENSOR_QOS,
        )

    def _on_image(self, message: Image) -> None:
        self.image = message
        self._try_pair()

    def _on_scan(self, message: LaserScan) -> None:
        self.scan = message
        self._try_pair()

    def _try_pair(self) -> None:
        if self.image is None or self.scan is None:
            return
        if abs(stamp_ns(self.image) - stamp_ns(self.scan)) <= self.max_delta_ns:
            self.pair = (self.image, self.scan)


def render_scan(scan: LaserScan, path: Path) -> None:
    size = 900
    origin = np.array([size // 2, size - 100])
    pixels_per_metre = 150.0
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)

    points: list[tuple[int, int]] = []
    for index, distance in enumerate(scan.ranges):
        valid_range = scan.range_min <= distance <= min(scan.range_max, 5.0)
        if not math.isfinite(distance) or not valid_range:
            continue
        angle = scan.angle_min + index * scan.angle_increment
        x_forward = distance * math.cos(angle)
        y_left = distance * math.sin(angle)
        pixel = origin + np.array([-y_left, -x_forward]) * pixels_per_metre
        px, py = int(round(pixel[0])), int(round(pixel[1]))
        if 0 <= px < size and 0 <= py < size:
            points.append((px, py))

    for point in points:
        cv2.circle(canvas, point, 2, (40, 40, 40), -1)
    cv2.circle(canvas, tuple(origin), 8, (0, 0, 255), -1)
    cv2.arrowedLine(
        canvas,
        tuple(origin),
        (origin[0], origin[1] - 100),
        (0, 140, 0),
        3,
        tipLength=0.15,
    )
    cv2.putText(canvas, "robot forward", (origin[0] + 12, origin[1] - 90), 0, 0.7, (0, 100, 0), 2)
    cv2.imwrite(str(path), canvas)


def write_pair(image_message: Image, scan: LaserScan, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    image = decode_image(image_message)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = "classic"
    detected, corners = cv2.findChessboardCorners(gray, (8, 6))
    if not detected:
        detected, corners = cv2.findChessboardCornersSB(gray, (8, 6))
        detector = "sb" if detected else "none"
    annotated = image.copy()
    if detected:
        cv2.drawChessboardCorners(annotated, (8, 6), corners, detected)

    cv2.imwrite(str(output / "image_raw.png"), image)
    cv2.imwrite(str(output / "image_checkerboard.png"), annotated)
    render_scan(scan, output / "scan_topdown.png")

    ranges = [float(value) if math.isfinite(value) else None for value in scan.ranges]
    record = {
        "image": {
            "timestamp_ns": stamp_ns(image_message),
            "frame_id": image_message.header.frame_id,
            "width": image_message.width,
            "height": image_message.height,
            "encoding": image_message.encoding,
            "checkerboard_detected": bool(detected),
            "checkerboard_corners": 0 if corners is None else len(corners),
            "checkerboard_detector": detector,
        },
        "scan": {
            "timestamp_ns": stamp_ns(scan),
            "frame_id": scan.header.frame_id,
            "angle_min": scan.angle_min,
            "angle_max": scan.angle_max,
            "angle_increment": scan.angle_increment,
            "range_min": scan.range_min,
            "range_max": scan.range_max,
            "ranges": ranges,
        },
        "absolute_delta_ms": abs(stamp_ns(image_message) - stamp_ns(scan)) / 1_000_000,
    }
    (output / "pair.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items() if key != "scan"}, indent=2))
    print(f"scan_frame={scan.header.frame_id} scan_beams={len(scan.ranges)}")
    print(f"saved={output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-delta-ms", type=float, default=75.0)
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    args = parser.parse_args()

    rclpy.init()
    node = PairCapture(round(args.max_delta_ms * 1_000_000))
    deadline = node.get_clock().now().nanoseconds + round(args.timeout_sec * 1_000_000_000)
    while node.pair is None and node.get_clock().now().nanoseconds < deadline:
        rclpy.spin_once(node, timeout_sec=0.25)

    pair = node.pair
    node.destroy_node()
    rclpy.shutdown()
    if pair is None:
        raise TimeoutError("no image/scan pair met the timestamp limit")
    write_pair(*pair, args.output)


if __name__ == "__main__":
    main()
