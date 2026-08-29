#!/usr/bin/env python3
"""Export the locked 10 Hz goal-conditioned training view from a pilot MCAP bag.

Runs on the ROS host (WSL `Ubuntu-TB3`, ROS 2 Humble) because it needs
`rosbag2_py`. All association and rejection logic lives in the platform-neutral
`livifuser_nav` package so it stays unit tested on Windows.

Two passes over the bag:

1. Read headers and payload metadata only, never retaining image bytes, then
   assemble the 10 Hz grid and decide which samples are accepted.
2. Re-read and copy only the accepted frames straight into preallocated memmaps,
   so peak memory stays independent of bag length.

Nothing here resizes, normalizes, or tokenizes. The export stays at capture
fidelity (320x240 RGB, raw ranges) and records the intended train resolution as
a preprocessing parameter, so a preprocessing change never requires re-export.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import rosbag2_py  # noqa: E402
import yaml  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
from rosidl_runtime_py.utilities import get_message  # noqa: E402

from livifuser_nav.association import TimeSeries  # noqa: E402
from livifuser_nav.calibration_diagnostics import camera_fov_degrees  # noqa: E402
from livifuser_nav.contracts import StampedValue  # noqa: E402
from livifuser_nav.decode import (  # noqa: E402
    goal_payload,
    image_payload,
    odometry_payload,
    scan_payload,
    twist_payload,
)
from livifuser_nav.export_schema import (  # noqa: E402
    EXPORT_SCHEMA_VERSION,
    ExportPolicy,
    RejectionCode,
    StreamRule,
    TimestampSource,
    apply_run_level_override,
)
from livifuser_nav.manifest_schema import assert_manifest_valid  # noqa: E402
from livifuser_nav.provenance import (  # noqa: E402
    code_identity,
    environment_identity,
    flush_and_close_memmaps,
    hash_paths,
    sha256_bytes,
)
from livifuser_nav.run_checks import (  # noqa: E402
    compare_transform,
    count_timestamp_regressions,
    scan_geometry_report,
)
from livifuser_nav.sampling import Payload, assemble_samples  # noqa: E402

STAMPED_ACTION_TOPIC = "/livifuser/cmd_vel_stamped"
INTENT_TOPIC = "/livifuser/teleop_intent_stamped"
UNSTAMPED_ACTION_TOPIC = "/cmd_vel"

EXPECTED_STATIC_TF = (("base_scan", "camera"), ("camera", "camera_optical_frame"))

SOURCE_FILES = [
    REPO_ROOT / "scripts" / "export_pilot_dataset.py",
    REPO_ROOT / "src" / "livifuser_nav" / "association.py",
    REPO_ROOT / "src" / "livifuser_nav" / "calibration_diagnostics.py",
    REPO_ROOT / "src" / "livifuser_nav" / "contracts.py",
    REPO_ROOT / "src" / "livifuser_nav" / "decode.py",
    REPO_ROOT / "src" / "livifuser_nav" / "export_schema.py",
    REPO_ROOT / "src" / "livifuser_nav" / "manifest_schema.py",
    REPO_ROOT / "src" / "livifuser_nav" / "provenance.py",
    REPO_ROOT / "src" / "livifuser_nav" / "replay_safety.py",
    REPO_ROOT / "src" / "livifuser_nav" / "run_checks.py",
    REPO_ROOT / "src" / "livifuser_nav" / "sampling.py",
]


def stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return stamp.sec * 1_000_000_000 + stamp.nanosec


@dataclass
class StreamBuffer:
    """Header-time metadata for one topic, in bag read order."""

    rule: StreamRule
    stamps: list[int]
    payloads: list[Payload]
    read_indices: list[int]

    def sorted_series(self) -> tuple[TimeSeries[Payload], list[int], int]:
        """Sort by header time, reporting genuine regression events.

        The count is of adjacent backward steps in arrival order, not of
        displaced positions after sorting: one early element out of place
        displaces every later position and would read as many faults.
        """

        regressions = count_timestamp_regressions(self.stamps)
        order = sorted(range(len(self.stamps)), key=lambda index: self.stamps[index])
        series = TimeSeries(
            StampedValue(self.stamps[index], self.payloads[index]) for index in order
        )
        return series, [self.read_indices[index] for index in order], regressions


def load_calibration(intrinsics: Path, extrinsics: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    if intrinsics.is_file():
        report["intrinsics"] = yaml.safe_load(intrinsics.read_text(encoding="utf-8"))
    if extrinsics.is_file():
        report["extrinsics"] = yaml.safe_load(extrinsics.read_text(encoding="utf-8"))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="rosbag2 directory containing the MCAP")
    parser.add_argument("--output", type=Path, required=True, help="export directory")
    parser.add_argument("--environment-id", required=True, help="split unit, e.g. corridor_a")
    parser.add_argument("--run-id", default=None, help="defaults to the bag directory name")
    parser.add_argument(
        "--domain",
        choices=("hardware", "simulation"),
        default="hardware",
        help=(
            "source domain; simulation selects the Gazebo rgb8 camera contract "
            "and is written into the manifest"
        ),
    )
    parser.add_argument(
        "--view",
        choices=("policy", "sensor"),
        default="policy",
        help="policy requires a usable action per sample; sensor records it as advisory",
    )
    parser.add_argument("--lidar-causal", action="store_true")
    parser.add_argument("--max-lidar-delta-ms", type=float, default=75.0)
    parser.add_argument("--max-camera-delta-ms", type=float, default=50.0)
    parser.add_argument("--max-action-staleness-ms", type=float, default=150.0)
    parser.add_argument("--max-odom-staleness-ms", type=float, default=100.0)
    parser.add_argument("--max-goal-staleness-ms", type=float, default=150.0)
    parser.add_argument("--train-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=REPO_ROOT / "config/calibration/rpi_camera_v3_320x240_2026-07-29.yaml",
    )
    parser.add_argument(
        "--extrinsics",
        type=Path,
        default=REPO_ROOT / "config/calibration/lidar_camera_extrinsics_2026-07-29.yaml",
    )
    parser.add_argument("--allow-calibration-mismatch", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace a non-empty export directory (evidence is otherwise never overwritten)",
    )
    args = parser.parse_args()

    run_id = args.run_id or args.bag.name
    source_image_encoding = "rgb8" if args.domain == "simulation" else "bgra8"
    policy = ExportPolicy(
        camera=StreamRule(
            topic="/camera/image_raw",
            policy=ExportPolicy().camera.policy,
            timestamp_source=TimestampSource.HEADER_STAMP,
            max_delta_ns=int(args.max_camera_delta_ms * 1_000_000),
            message_type="sensor_msgs/msg/Image",
        ),
        lidar=StreamRule(
            topic="/scan",
            policy=ExportPolicy().lidar.policy,
            timestamp_source=TimestampSource.HEADER_STAMP,
            max_delta_ns=int(args.max_lidar_delta_ms * 1_000_000),
            message_type="sensor_msgs/msg/LaserScan",
        ),
        odometry=StreamRule(
            topic="/odom",
            policy=ExportPolicy().odometry.policy,
            timestamp_source=TimestampSource.HEADER_STAMP,
            max_delta_ns=int(args.max_odom_staleness_ms * 1_000_000),
            message_type="nav_msgs/msg/Odometry",
        ),
        goal=StreamRule(
            topic="/livifuser/goal_relative",
            policy=ExportPolicy().goal.policy,
            timestamp_source=TimestampSource.HEADER_STAMP,
            max_delta_ns=int(args.max_goal_staleness_ms * 1_000_000),
            message_type="livifuser_interfaces/msg/RelativeGoal",
        ),
        action=ExportPolicy().action,
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    message_types = {name: get_message(kind) for name, kind in topic_types.items()}

    # A stamped final command is preferred; an unstamped Twist only has a bag
    # receive time, which is a proxy for publication time and must be declared.
    if STAMPED_ACTION_TOPIC in topic_types:
        action_topic = STAMPED_ACTION_TOPIC
        action_source = TimestampSource.HEADER_STAMP
    else:
        action_topic = UNSTAMPED_ACTION_TOPIC
        action_source = TimestampSource.BAG_RECEIVE
    policy = ExportPolicy(
        camera=policy.camera,
        lidar=policy.lidar,
        odometry=policy.odometry,
        goal=policy.goal,
        action=StreamRule(
            topic=action_topic,
            policy=ExportPolicy().action.policy,
            timestamp_source=action_source,
            max_delta_ns=int(args.max_action_staleness_ms * 1_000_000),
            message_type=topic_types.get(action_topic, "geometry_msgs/msg/Twist"),
        ),
    )

    buffers = {
        name: StreamBuffer(rule=rule, stamps=[], payloads=[], read_indices=[])
        for name, rule in policy.streams().items()
    }
    counts: Counter[str] = Counter()
    read_positions: Counter[str] = Counter()
    static_transforms: dict[tuple[str, str], dict[str, list[float]]] = {}
    camera_info_variants: dict[tuple[Any, ...], dict[str, Any]] = {}
    camera_info_count = 0
    scan_geometry_data: list[dict[str, Any]] = []
    intent_count = 0

    while reader.has_next():
        topic, serialized, receive_ns = reader.read_next()
        counts[topic] += 1
        read_index = read_positions[topic]
        read_positions[topic] += 1
        if topic not in message_types:
            continue
        message = deserialize_message(serialized, message_types[topic])

        if topic == policy.camera.topic:
            payload = image_payload(message, (320, 240, source_image_encoding))
            # Retain arrival time: a scan whose header post-dates this frame may
            # still have arrived before the frame finished its pipeline, and
            # arrival order is what decides online availability.
            buffers["camera"].stamps.append(stamp_ns(message))
            buffers["camera"].payloads.append(_with_receive(payload, receive_ns))
            buffers["camera"].read_indices.append(read_index)
        elif topic == policy.lidar.topic:
            payload = scan_payload(message)
            scan_geometry_data.append(dict(payload.data))
            buffers["lidar"].stamps.append(stamp_ns(message))
            buffers["lidar"].payloads.append(_with_receive(payload, receive_ns))
            buffers["lidar"].read_indices.append(read_index)
        elif topic == policy.odometry.topic:
            buffers["odometry"].stamps.append(stamp_ns(message))
            buffers["odometry"].payloads.append(odometry_payload(message))
            buffers["odometry"].read_indices.append(read_index)
        elif topic == policy.goal.topic:
            buffers["goal"].stamps.append(stamp_ns(message))
            buffers["goal"].payloads.append(goal_payload(message))
            buffers["goal"].read_indices.append(read_index)
        elif topic == action_topic:
            if action_source is TimestampSource.HEADER_STAMP:
                timestamp = stamp_ns(message)
                twist = message.twist
            else:
                timestamp = receive_ns
                twist = message
            buffers["action"].stamps.append(timestamp)
            buffers["action"].payloads.append(twist_payload(twist.linear.x, twist.angular.z))
            buffers["action"].read_indices.append(read_index)
        elif topic == INTENT_TOPIC:
            intent_count += 1
        elif topic == "/camera/camera_info":
            # Every CameraInfo is checked, not just the first: a driver reload
            # mid-run can change calibration and the first message would hide it.
            camera_info_count += 1
            signature = (
                message.width,
                message.height,
                message.distortion_model,
                tuple(message.k),
                tuple(message.d),
            )
            if signature not in camera_info_variants:
                camera_info_variants[signature] = {
                    "width": message.width,
                    "height": message.height,
                    "distortion_model": message.distortion_model,
                    "k": list(message.k),
                    "d": list(message.d),
                    "frame_id": message.header.frame_id,
                    "first_seen_message": camera_info_count,
                }
        elif topic == "/tf_static":
            for transform in message.transforms:
                key = (transform.header.frame_id, transform.child_frame_id)
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                static_transforms[key] = {
                    "translation": [translation.x, translation.y, translation.z],
                    "quaternion_xyzw": [
                        rotation.x,
                        rotation.y,
                        rotation.z,
                        rotation.w,
                    ],
                }

    if not buffers["camera"].stamps:
        print("No camera messages found; nothing to export.", file=sys.stderr)
        return 1

    series: dict[str, TimeSeries[Payload]] = {}
    read_maps: dict[str, list[int]] = {}
    regressions: dict[str, int] = {}
    for name, buffer in buffers.items():
        stream, read_map, regression_count = buffer.sorted_series()
        series[name] = stream
        read_maps[name] = read_map
        regressions[name] = regression_count

    calibration = load_calibration(args.intrinsics, args.extrinsics)
    run_level: list[RejectionCode] = []
    calibration_notes: list[str] = []

    accepted_k = (
        calibration.get("intrinsics", {}).get("camera_matrix", {}).get("data")
        if calibration.get("intrinsics")
        else None
    )
    camera_info = next(iter(camera_info_variants.values()), None)
    if not camera_info_variants:
        calibration_notes.append("bag contains no CameraInfo")
        run_level.append(RejectionCode.CALIBRATION_MISMATCH)
    elif len(camera_info_variants) > 1:
        calibration_notes.append(
            f"CameraInfo changed mid-run: {len(camera_info_variants)} distinct "
            f"calibrations across {camera_info_count} messages"
        )
        run_level.append(RejectionCode.CALIBRATION_MISMATCH)
    for variant in camera_info_variants.values():
        if not any(variant["k"]):
            calibration_notes.append("recorded CameraInfo K is all zeros (uncalibrated)")
            run_level.append(RejectionCode.CALIBRATION_MISMATCH)
        elif accepted_k is not None:
            if len(variant["k"]) != len(accepted_k):
                calibration_notes.append(
                    f"CameraInfo K has {len(variant['k'])} entries, "
                    f"accepted intrinsics have {len(accepted_k)}"
                )
                run_level.append(RejectionCode.CALIBRATION_MISMATCH)
            else:
                deltas = [
                    abs(recorded - accepted)
                    for recorded, accepted in zip(variant["k"], accepted_k, strict=True)
                ]
                if max(deltas) > 1e-3:
                    calibration_notes.append(
                        "recorded CameraInfo K differs from accepted intrinsics "
                        f"(max {max(deltas):.6f})"
                    )
                    run_level.append(RejectionCode.CALIBRATION_MISMATCH)

    missing_tf = [pair for pair in EXPECTED_STATIC_TF if pair not in static_transforms]
    if missing_tf:
        calibration_notes.append(f"static TF chain incomplete: missing {missing_tf}")
        run_level.append(RejectionCode.TF_UNAVAILABLE)

    # Frame names existing proves only that something publishes. Compare the
    # numbers: an identity transform under the right names would pass a name-only
    # check while destroying the geometry the fusion mask depends on.
    tf_comparison: dict[str, Any] | None = None
    extrinsics = calibration.get("extrinsics") or {}
    accepted_translation = extrinsics.get("translation_m")
    accepted_quaternion = extrinsics.get("rotation_quaternion_xyzw")
    measured = static_transforms.get(("base_scan", "camera"))
    if measured and accepted_translation and accepted_quaternion:
        comparison = compare_transform(
            measured["translation"],
            measured["quaternion_xyzw"],
            [accepted_translation["x"], accepted_translation["y"], accepted_translation["z"]],
            [
                accepted_quaternion["x"],
                accepted_quaternion["y"],
                accepted_quaternion["z"],
                accepted_quaternion["w"],
            ],
        )
        tf_comparison = {
            "pair": ["base_scan", "camera"],
            "recorded": measured,
            **comparison.as_manifest(),
        }
        if not comparison.matches:
            calibration_notes.append(
                "recorded base_scan -> camera transform differs from accepted "
                f"extrinsics by {comparison.translation_error_m * 1000:.2f} mm and "
                f"{math.degrees(comparison.rotation_error_rad):.3f} deg"
            )
            run_level.append(RejectionCode.CALIBRATION_MISMATCH)
    elif not measured:
        calibration_notes.append(
            "base_scan -> camera transform absent; cannot verify extrinsics numerically"
        )
        run_level.append(RejectionCode.TF_UNAVAILABLE)

    geometry_report = scan_geometry_report(scan_geometry_data)
    if scan_geometry_data and not geometry_report.is_constant:
        calibration_notes.append(
            "LiDAR angular frame changed mid-run: "
            f"{len(geometry_report.frame_variants)} variants; stored ranges cannot be "
            "tokenized with one geometry"
        )
        run_level.append(RejectionCode.LIDAR_PAYLOAD_INVALID)
    elif len(geometry_report.beam_counts) > 1:
        # Expected on this scanner. Not a fault, but it does mean bearings are
        # per-scan, so the note records the worst-case disagreement.
        calibration_notes.append(
            f"beam count varies across the run: {list(geometry_report.beam_counts)}; "
            f"angle_increment covaries, giving up to "
            f"{geometry_report.max_bearing_spread_deg:.3f} deg bearing spread at the "
            "far beam. Per-scan increment stored in vectors.npz; use it, not a "
            "global bearing table."
        )

    if any(regressions.values()):
        regressed = {name: count for name, count in regressions.items() if count}
        calibration_notes.append(f"timestamp regression events per stream: {regressed}")
        run_level.append(RejectionCode.TIMESTAMP_REGRESSION)

    retained_codes, downgraded_codes = apply_run_level_override(
        run_level, allow_calibration_mismatch=args.allow_calibration_mismatch
    )
    if downgraded_codes:
        calibration_notes.append(
            "--allow-calibration-mismatch downgraded to warnings: "
            + ", ".join(code.value for code in downgraded_codes)
        )
    if args.allow_calibration_mismatch and retained_codes:
        calibration_notes.append(
            "run-level faults NOT downgradable and still rejecting every sample: "
            + ", ".join(code.value for code in retained_codes)
        )
    run_level = list(retained_codes)

    for name, count in regressions.items():
        if count:
            calibration_notes.append(f"{name}: {count} out-of-order arrivals were sorted")

    result = assemble_samples(
        camera=series["camera"],
        lidar=series["lidar"] or None,
        odometry=series["odometry"] or None,
        goal=series["goal"] or None,
        action=series["action"] or None,
        policy=policy,
        require_action=args.view == "policy",
        lidar_causal=args.lidar_causal,
        run_level_codes=tuple(dict.fromkeys(run_level)),
    )

    accepted = result.accepted
    output: Path = args.output
    if output.exists() and any(output.iterdir()):
        if not args.force:
            print(
                f"refusing to overwrite existing export at {output}",
                file=sys.stderr,
            )
            print(
                "Raw and derived evidence is never overwritten; choose a new "
                "directory or pass --force.",
                file=sys.stderr,
            )
            return 1
        print(f"NOTE: --force given; replacing contents of {output}")
    output.mkdir(parents=True, exist_ok=True)

    # Second pass: copy only accepted frames into preallocated arrays so peak
    # memory does not scale with bag length.
    # Pad every row to the longest scan observed; short scans keep NaN in the
    # tail, which the validity mask treats as no return.
    beam_count = geometry_report.max_beam_count

    rgb_path = output / "rgb_320x240_rgb8.npy"
    scan_path = output / "scan_ranges.npy"
    rgb = np.lib.format.open_memmap(
        rgb_path, mode="w+", dtype=np.uint8, shape=(len(accepted), 240, 320, 3)
    )
    scan = np.lib.format.open_memmap(
        scan_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(accepted), beam_count if beam_count else 1),
    )

    wanted_camera = {
        sample.selections["camera"].source_index: position
        for position, sample in enumerate(accepted)
    }
    wanted_scan = {
        sample.selections["lidar"].source_index: position
        for position, sample in enumerate(accepted)
        if sample.selections["lidar"].source_index is not None
    }
    camera_read_wanted = {
        read_maps["camera"][sorted_index]: position
        for sorted_index, position in wanted_camera.items()
        if sorted_index is not None
    }
    scan_read_wanted = {
        read_maps["lidar"][sorted_index]: position
        for sorted_index, position in wanted_scan.items()
    }

    alpha_values: set[int] = set()
    reader2 = rosbag2_py.SequentialReader()
    reader2.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    positions: Counter[str] = Counter()
    while reader2.has_next():
        topic, serialized, _receive_ns = reader2.read_next()
        index = positions[topic]
        positions[topic] += 1
        if topic == policy.camera.topic and index in camera_read_wanted:
            message = deserialize_message(serialized, message_types[topic])
            channels = 3 if source_image_encoding == "rgb8" else 4
            frame = np.frombuffer(message.data, dtype=np.uint8).reshape(
                message.height, message.step // channels, channels
            )[:, : message.width, :]
            if source_image_encoding == "bgra8":
                alpha_values.update(np.unique(frame[:, :, 3]).tolist())
                # bgra8 -> rgb8; alpha is a padding channel on the real driver.
                frame = frame[:, :, [2, 1, 0]]
            rgb[camera_read_wanted[index]] = frame
        elif topic == policy.lidar.topic and index in scan_read_wanted and beam_count:
            message = deserialize_message(serialized, message_types[topic])
            values = np.asarray(message.ranges, dtype=np.float32)
            row = np.full(beam_count, np.nan, dtype=np.float32)
            row[: min(beam_count, values.size)] = values[:beam_count]
            scan[scan_read_wanted[index]] = row

    # DrvFS can expose a post-close file image that differs from a hash read
    # while a writable NumPy mapping is still alive. Flush and explicitly
    # unmap both arrays before any output is hashed or entered in the manifest.
    flush_and_close_memmaps(rgb, scan)
    del rgb, scan

    vectors = {
        "grid_timestamp_ns": np.array(
            [sample.grid_timestamp_ns for sample in accepted], dtype=np.int64
        ),
        "observation_timestamp_ns": np.array(
            [sample.observation_timestamp_ns or 0 for sample in accepted], dtype=np.int64
        ),
        "segment_id": np.array([sample.segment_id for sample in accepted], dtype=np.int32),
        "goal": np.array(
            [
                [
                    _value(sample, "goal", "rho_m"),
                    _value(sample, "goal", "sin_alpha"),
                    _value(sample, "goal", "cos_alpha"),
                ]
                for sample in accepted
            ],
            dtype=np.float32,
        ).reshape(len(accepted), 3),
        "robot_state": np.array(
            [
                [
                    _value(sample, "odometry", "linear_velocity_mps"),
                    _value(sample, "odometry", "angular_velocity_radps"),
                ]
                for sample in accepted
            ],
            dtype=np.float32,
        ).reshape(len(accepted), 2),
        "action": np.array(
            [
                [
                    _value(sample, "action", "linear_velocity_mps"),
                    _value(sample, "action", "angular_velocity_radps"),
                ]
                for sample in accepted
            ],
            dtype=np.float32,
        ).reshape(len(accepted), 2),
        "lidar_signed_delta_ns": np.array(
            [sample.selections["lidar"].signed_delta_ns or 0 for sample in accepted],
            dtype=np.int64,
        ),
        # Bearings must be reconstructed per scan: this driver's increment
        # covaries with beam count, so a single global table would be wrong.
        "scan_angle_increment_rad": np.array(
            [_value(sample, "lidar", "angle_increment") for sample in accepted],
            dtype=np.float64,
        ),
        "scan_beam_count": np.array(
            [_value(sample, "lidar", "beam_count") for sample in accepted],
            dtype=np.int32,
        ),
    }
    np.savez(output / "vectors.npz", **vectors)

    samples_path = output / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for position, sample in enumerate(accepted):
            record = sample.as_record()
            record["row"] = position
            handle.write(json.dumps(record) + "\n")

    rejections_path = output / "rejections.jsonl"
    with rejections_path.open("w", encoding="utf-8") as handle:
        for sample in result.samples:
            if not sample.accepted:
                handle.write(json.dumps(sample.as_record()) + "\n")

    future_statistics = _future_statistics(result.samples)
    effective_configuration = {
        "domain": args.domain,
        "source_image_encoding": source_image_encoding,
        "view": args.view,
        "lidar_causal": bool(args.lidar_causal),
        "max_lidar_delta_ms": args.max_lidar_delta_ms,
        "max_camera_delta_ms": args.max_camera_delta_ms,
        "max_action_staleness_ms": args.max_action_staleness_ms,
        "max_odom_staleness_ms": args.max_odom_staleness_ms,
        "max_goal_staleness_ms": args.max_goal_staleness_ms,
        "train_size": list(args.train_size),
        "action_topic": action_topic,
        "run_level_codes_retained": [code.value for code in retained_codes],
        "run_level_codes_downgraded": [code.value for code in downgraded_codes],
        "action_timestamp_source": action_source.value,
        "allow_calibration_mismatch": bool(args.allow_calibration_mismatch),
        "intrinsics_path": str(args.intrinsics),
        "extrinsics_path": str(args.extrinsics),
    }

    intrinsics_data = calibration.get("intrinsics") or {}
    fov = (
        camera_fov_degrees(
            intrinsics_data["camera_matrix"]["data"],
            intrinsics_data.get("image_width", 320),
            intrinsics_data.get("image_height", 240),
        )
        if intrinsics_data.get("camera_matrix")
        else None
    )

    manifest: dict[str, Any] = {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "environment_id": args.environment_id,
        "domain": args.domain,
        "view": args.view,
        "code": code_identity(REPO_ROOT, SOURCE_FILES).as_manifest(),
        "environment": environment_identity(),
        "inputs": hash_paths(
            {
                **_mcap_inputs(args.bag),
                "metadata": args.bag / "metadata.yaml",
                "intrinsics": args.intrinsics,
                "extrinsics": args.extrinsics,
            }
        ),
        "effective_configuration": effective_configuration,
        "effective_configuration_sha256": sha256_bytes(
            json.dumps(effective_configuration, sort_keys=True).encode("utf-8")
        ),
        "association_policy": policy.as_manifest(),
        "run_level_codes_retained": [code.value for code in retained_codes],
        "run_level_codes_downgraded": [code.value for code in downgraded_codes],
        "action_timestamp_source": action_source.value,
        "action_topic": action_topic,
        "action_policy_note": (
            "Zero-order hold from the latest prior command. Commands are never "
            "interpolated and a missing command is never manufactured as zero."
        ),
        "teleop_intent_messages": intent_count,
        "lidar_association_mode": "causal" if args.lidar_causal else "nearest",
        "lidar_future_selection": future_statistics,
        "lidar_future_selection_note": (
            "Nearest association is spec-locked in section 7.2 but may select a scan "
            "recorded after the camera frame, which the online policy could not have "
            "seen. This fraction quantifies that train/deploy asymmetry."
        ),
        "preprocessing": {
            "capture_size": [320, 240],
            "stored_encoding": "rgb8",
            "source_encoding": source_image_encoding,
            "alpha_values_observed": sorted(alpha_values),
            "intended_train_size": list(args.train_size),
            "resize_applied": False,
            "normalization_applied": False,
            "lidar_tokenization_applied": False,
            "note": (
                "Stored at capture fidelity. Resize, normalization, and the "
                "[r, sin, cos, validity] tokenization are preprocessing steps applied "
                "downstream so they can change without re-exporting."
            ),
        },
        "calibration": {
            "recorded_camera_info": camera_info,
            "camera_info_message_count": camera_info_count,
            "camera_info_distinct_variants": list(camera_info_variants.values()),
            "accepted_intrinsics_camera_matrix": accepted_k,
            "derived_camera_fov": fov,
            "static_transforms": {
                f"{parent}->{child}": value
                for (parent, child), value in sorted(static_transforms.items())
            },
            "transform_verification": tf_comparison,
            "lidar_geometry": geometry_report.as_manifest(),
            "expected_static_tf": [list(pair) for pair in EXPECTED_STATIC_TF],
            "notes": calibration_notes,
        },
        "counts": {
            "bag_messages": dict(sorted(counts.items())),
            "source_messages_per_stream": dict(result.streams_present),
            "grid_ticks": len(result.samples),
            "accepted_samples": len(accepted),
            "rejected_samples": len(result.samples) - len(accepted),
            "timestamp_regression_events": regressions,
            "acceptance_rate": (
                len(accepted) / len(result.samples) if result.samples else 0.0
            ),
        },
        "rejections": {
            "by_primary_reason": result.rejection_counts(),
            "by_any_reason": result.all_rejection_counts(),
            "advisory": result.advisory_counts(),
        },
        "contiguity": {
            "segment_count": len(result.segment_lengths),
            "segment_lengths": list(result.segment_lengths),
            "longest_segment": max(result.segment_lengths, default=0),
            "windowable_k8_h8": result.windowable_count(context_k=8, horizon_h=8),
        },
        "outputs": {},
    }

    for path in (rgb_path, scan_path, output / "vectors.npz", samples_path, rejections_path):
        manifest["outputs"][path.name] = {
            "sha256": _hash_if_present(path),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    manifest["manifest_sha256_excludes_self"] = sha256_bytes(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    )

    try:
        assert_manifest_valid(manifest)
    except ValueError as error:
        print("MANIFEST SCHEMA FAILURE", file=sys.stderr)
        print(error, file=sys.stderr)
        return 1
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"export_schema_version : {EXPORT_SCHEMA_VERSION}")
    print(f"view                  : {args.view}")
    print(f"grid ticks            : {len(result.samples)}")
    print(f"accepted samples      : {len(accepted)}")
    print(f"acceptance rate       : {manifest['counts']['acceptance_rate']:.4%}")
    print(f"segments              : {list(result.segment_lengths)}")
    print(f"windowable K=8 H=8    : {manifest['contiguity']['windowable_k8_h8']}")
    print(f"action timestamps     : {action_source.value}")
    all_ticks = future_statistics["all_grid_ticks"]
    print(
        "lidar future (all ticks): "
        f"{all_ticks['future_header_fraction']:.2%} by header, "
        f"{all_ticks['unavailable_at_camera_arrival_fraction']:.2%} unavailable at "
        "camera arrival"
    )
    wait = all_ticks["required_wait_ms"]
    if wait["count"]:
        print(
            f"required wait for those    : median {wait['median_ms']:.2f} ms, "
            f"p95 {wait['p95_ms']:.2f} ms, max {wait['max_ms']:.2f} ms"
        )
    if result.rejection_counts():
        print("rejections by primary reason:")
        for reason, count in result.rejection_counts().items():
            print(f"  {count:6d}  {reason}")
    if result.advisory_counts():
        print("advisory (not rejected in sensor view):")
        for reason, count in result.advisory_counts().items():
            print(f"  {count:6d}  {reason}")
    for note in calibration_notes:
        print(f"NOTE: {note}")
    print(f"\nwrote {output}")
    return 0


def _with_receive(payload: Payload, receive_ns: int) -> Payload:
    """Attach bag arrival time so online availability can be evaluated later."""

    return Payload(
        data={**payload.data, "receive_ns": receive_ns},
        valid=payload.valid,
        invalid_code=payload.invalid_code,
    )


def _receive_ns(sample: Any, stream: str) -> int | None:
    selection = sample.selections.get(stream)
    if selection is None or selection.payload is None:
        return None
    value = selection.payload.data.get("receive_ns")
    return int(value) if isinstance(value, int) else None


def _future_statistics(samples: Any) -> dict[str, Any]:
    """LiDAR future-selection rates, with the denominator stated explicitly.

    Reporting only the accepted-sample rate conflates phase with selection bias:
    action and goal rejection removes ticks non-uniformly in time, so the
    accepted subset is not representative of the run.

    ``unavailable_at_camera_arrival`` is not by itself the train/deploy gap:
    online inference could wait briefly for some of those scans. What decides
    whether exact nearest association is feasible is how long that wait would
    be, so ``required_wait_ms`` reports the median, p95, and maximum against the
    100 ms budget.
    """

    def rates(subset: list[Any]) -> dict[str, Any]:
        deltas = [
            sample.selections["lidar"].signed_delta_ns
            for sample in subset
            if sample.selections["lidar"].signed_delta_ns is not None
        ]
        future = [delta for delta in deltas if delta > 0]
        arrived_in_time = 0
        waits: list[int] = []
        for sample in subset:
            delta = sample.selections["lidar"].signed_delta_ns
            if delta is None or delta <= 0:
                continue
            scan_arrival = _receive_ns(sample, "lidar")
            camera_arrival = _receive_ns(sample, "camera")
            if scan_arrival is None or camera_arrival is None:
                continue
            if scan_arrival <= camera_arrival:
                arrived_in_time += 1
            else:
                # How long online inference would have to wait for this scan.
                waits.append(scan_arrival - camera_arrival)
        return {
            "count": len(deltas),
            "future_header_count": len(future),
            "future_header_fraction": (len(future) / len(deltas)) if deltas else 0.0,
            "future_header_but_available_by_camera_arrival": arrived_in_time,
            "unavailable_at_camera_arrival": len(future) - arrived_in_time,
            "unavailable_at_camera_arrival_fraction": (
                (len(future) - arrived_in_time) / len(deltas) if deltas else 0.0
            ),
            "required_wait_ms": _wait_distribution(waits),
        }

    all_ticks = list(samples)
    eligible = [
        sample for sample in all_ticks if sample.selections["lidar"].eligible
    ]
    accepted = [sample for sample in all_ticks if sample.accepted]
    return {
        "all_grid_ticks": rates(all_ticks),
        "lidar_eligible_ticks": rates(eligible),
        "accepted_samples": rates(accepted),
        "note": (
            "Compare runs using all_grid_ticks. The accepted_samples denominator is "
            "subject to selection bias from action and goal rejection."
        ),
    }


def _wait_distribution(waits_ns: list[int]) -> dict[str, Any]:
    """Median, p95, and maximum wait, for comparison against the 100 ms budget."""

    if not waits_ns:
        return {"count": 0}
    ordered = sorted(value / 1_000_000 for value in waits_ns)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "median_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "max_ms": ordered[-1],
    }


def _value(sample: Any, stream: str, key: str) -> float:
    selection = sample.selections.get(stream)
    if selection is None or selection.payload is None:
        return math.nan
    value = selection.payload.data.get(key)
    return float(value) if isinstance(value, (int, float)) else math.nan


def _mcap_inputs(bag: Path) -> dict[str, Path]:
    """Every MCAP shard, so a split recording is hashed completely."""

    shards = sorted(bag.glob("*.mcap"))
    if not shards:
        return {"mcap_missing": bag / "missing.mcap"}
    return {f"mcap[{index}] {path.name}": path for index, path in enumerate(shards)}


def _hash_if_present(path: Path) -> str | None:
    from livifuser_nav.provenance import sha256_file

    return sha256_file(path) if path.is_file() else None


if __name__ == "__main__":
    raise SystemExit(main())
