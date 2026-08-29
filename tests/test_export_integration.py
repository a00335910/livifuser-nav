"""End-to-end MCAP -> arrays -> manifest test over a synthesized bag.

Skipped automatically where `rosbag2_py` is unavailable (Windows), and runs on
the ROS host. It uses a bag built here rather than a recorded pilot bag so the
expected sample count, action values, and geometry are known exactly.

This covers the layer the unit tests cannot reach: bag reading, the two-pass
memmap write, and manifest assembly.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the package importable however this file is invoked, so the suite does
# not depend on the caller's PYTHONPATH.
sys.path.insert(0, str(REPO_ROOT / "src"))

try:  # pragma: no cover - availability is the thing under test
    import numpy as np
    import rosbag2_py
    from geometry_msgs.msg import Twist
    from livifuser_interfaces.msg import RelativeGoal
    from nav_msgs.msg import Odometry
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import CameraInfo, Image, LaserScan
    from tf2_msgs.msg import TFMessage

    ROS_AVAILABLE = True
except ImportError:  # pragma: no cover
    ROS_AVAILABLE = False

MS = 1_000_000
TWO_PI = 2.0 * math.pi
BEAMS = 399

ACCEPTED_K = [
    316.21156, 0.0, 223.13834,
    0.0, 315.6497, 107.39364,
    0.0, 0.0, 1.0,
]
ACCEPTED_D = [0.012344, 0.038138, -0.016819, 0.004823, 0.0]
ACCEPTED_T = (0.0723955522, 0.0048472604, -0.0838973150)
ACCEPTED_Q = (-0.4806489642, 0.5212435451, -0.4930249275, 0.5041905996)

#: One second of data: 30 frames, 10 scans, 20 odom, 10 goals, 10 commands.
DURATION_NS = 1_000_000_000
BASE_NS = 1_000_000_000


def _stamp(message, timestamp_ns: int, frame_id: str):
    message.header.stamp.sec = timestamp_ns // 1_000_000_000
    message.header.stamp.nanosec = timestamp_ns % 1_000_000_000
    message.header.frame_id = frame_id
    return message


def _image(timestamp_ns: int, value: int, encoding: str = "bgra8") -> Image:
    message = _stamp(Image(), timestamp_ns, "camera")
    message.width = 320
    message.height = 240
    message.encoding = encoding
    channels = 3 if encoding == "rgb8" else 4
    message.step = 320 * channels
    # Distinct constant per frame so row ordering is verifiable downstream.
    message.data = bytes([value % 256]) * (240 * 320 * channels)
    return message


def _camera_info(timestamp_ns: int) -> CameraInfo:
    message = _stamp(CameraInfo(), timestamp_ns, "camera")
    message.width = 320
    message.height = 240
    message.distortion_model = "plumb_bob"
    message.k = ACCEPTED_K
    message.d = ACCEPTED_D
    return message


def _scan(timestamp_ns: int) -> LaserScan:
    message = _stamp(LaserScan(), timestamp_ns, "base_scan")
    message.angle_min = 0.0
    message.angle_max = TWO_PI
    message.angle_increment = TWO_PI / (BEAMS + 1)
    message.range_min = 0.1
    message.range_max = 100.0
    message.scan_time = 0.0996
    message.ranges = [1.25] * BEAMS
    return message


def _odometry(timestamp_ns: int, linear: float) -> Odometry:
    message = _stamp(Odometry(), timestamp_ns, "odom")
    message.twist.twist.linear.x = linear
    message.twist.twist.angular.z = 0.0
    message.pose.pose.orientation.w = 1.0
    return message


def _goal(timestamp_ns: int) -> RelativeGoal:
    message = _stamp(RelativeGoal(), timestamp_ns, "base_link")
    message.rho_m = 1.0
    message.sin_alpha = 0.0
    message.cos_alpha = 1.0
    return message


def _command(linear: float) -> Twist:
    message = Twist()
    message.linear.x = linear
    message.angular.z = 0.0
    return message


def _static_tf(timestamp_ns: int) -> TFMessage:
    from geometry_msgs.msg import TransformStamped

    measured = _stamp(TransformStamped(), timestamp_ns, "base_scan")
    measured.child_frame_id = "camera"
    measured.transform.translation.x = ACCEPTED_T[0]
    measured.transform.translation.y = ACCEPTED_T[1]
    measured.transform.translation.z = ACCEPTED_T[2]
    measured.transform.rotation.x = ACCEPTED_Q[0]
    measured.transform.rotation.y = ACCEPTED_Q[1]
    measured.transform.rotation.z = ACCEPTED_Q[2]
    measured.transform.rotation.w = ACCEPTED_Q[3]

    alias = _stamp(TransformStamped(), timestamp_ns, "camera")
    alias.child_frame_id = "camera_optical_frame"
    alias.transform.rotation.w = 1.0

    return TFMessage(transforms=[measured, alias])


def _write_bag(directory: Path, *, image_encoding: str = "bgra8") -> Path:
    bag = directory / "synthetic_bag"
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    topics = {
        "/camera/image_raw": "sensor_msgs/msg/Image",
        "/camera/camera_info": "sensor_msgs/msg/CameraInfo",
        "/scan": "sensor_msgs/msg/LaserScan",
        "/odom": "nav_msgs/msg/Odometry",
        "/livifuser/goal_relative": "livifuser_interfaces/msg/RelativeGoal",
        "/cmd_vel": "geometry_msgs/msg/Twist",
        "/tf_static": "tf2_msgs/msg/TFMessage",
    }
    for name, kind in topics.items():
        try:
            metadata = rosbag2_py.TopicMetadata(
                name=name, type=kind, serialization_format="cdr"
            )
        except TypeError:  # newer rosbag2 requires an explicit id
            metadata = rosbag2_py.TopicMetadata(
                id=0, name=name, type=kind, serialization_format="cdr"
            )
        writer.create_topic(metadata)

    records: list[tuple[int, str, object]] = [
        (BASE_NS, "/tf_static", _static_tf(BASE_NS))
    ]
    for index in range(30):  # camera at 30 Hz
        timestamp = BASE_NS + index * (DURATION_NS // 30)
        records.append(
            (
                timestamp,
                "/camera/image_raw",
                _image(timestamp, index, image_encoding),
            )
        )
        records.append((timestamp, "/camera/camera_info", _camera_info(timestamp)))
    for index in range(10):  # scan at 10 Hz, 5 ms after each tick
        timestamp = BASE_NS + index * 100 * MS + 5 * MS
        records.append((timestamp, "/scan", _scan(timestamp)))
    for index in range(20):  # odometry at 20 Hz
        timestamp = BASE_NS + index * 50 * MS
        records.append((timestamp, "/odom", _odometry(timestamp, 0.05)))
    for index in range(10):
        timestamp = BASE_NS + index * 100 * MS
        records.append((timestamp, "/livifuser/goal_relative", _goal(timestamp)))
        records.append((timestamp, "/cmd_vel", _command(0.05)))

    for timestamp, topic, message in sorted(records, key=lambda row: row[0]):
        writer.write(topic, serialize_message(message), timestamp)
    del writer
    return bag


@unittest.skipUnless(ROS_AVAILABLE, "requires rosbag2_py and ROS message packages")
class ExportRoundTripTests(unittest.TestCase):
    """One export of a known bag, asserted from several angles."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        root = Path(cls._temporary.name)
        cls.bag = _write_bag(root)
        cls.output = root / "export"
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "export_pilot_dataset.py"),
                str(cls.bag),
                "--output",
                str(cls.output),
                "--environment-id",
                "synthetic",
                "--view",
                "policy",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        cls.completed = completed
        if completed.returncode != 0:
            raise AssertionError(
                f"exporter failed ({completed.returncode})\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        cls.manifest = json.loads((cls.output / "manifest.json").read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_exporter_succeeded(self) -> None:
        self.assertEqual(self.completed.returncode, 0)

    def test_manifest_passes_schema_validation(self) -> None:
        from livifuser_nav.manifest_schema import validate_manifest

        self.assertEqual(validate_manifest(self.manifest), [])

    def test_expected_number_of_samples_accepted(self) -> None:
        # Ten 10 Hz ticks span the one-second bag and every stream is present.
        self.assertEqual(self.manifest["counts"]["grid_ticks"], 10)
        self.assertEqual(self.manifest["counts"]["accepted_samples"], 10)
        self.assertEqual(self.manifest["rejections"]["by_primary_reason"], {})

    def test_all_samples_form_one_contiguous_segment(self) -> None:
        self.assertEqual(self.manifest["contiguity"]["segment_lengths"], [10])

    def test_rgb_array_shape_and_content(self) -> None:
        rgb = np.load(self.output / "rgb_320x240_rgb8.npy", mmap_mode="r")
        self.assertEqual(rgb.shape, (10, 240, 320, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        # Each synthetic frame is a distinct constant, and every tick should have
        # selected a different frame.
        firsts = [int(rgb[row, 0, 0, 0]) for row in range(10)]
        self.assertEqual(len(set(firsts)), 10)

    def test_scan_array_shape_and_values(self) -> None:
        scan = np.load(self.output / "scan_ranges.npy", mmap_mode="r")
        self.assertEqual(scan.shape, (10, BEAMS))
        self.assertTrue(np.allclose(scan, 1.25))

    def test_vectors_carry_the_locked_contract(self) -> None:
        vectors = np.load(self.output / "vectors.npz")
        self.assertEqual(vectors["goal"].shape, (10, 3))
        self.assertEqual(vectors["robot_state"].shape, (10, 2))
        self.assertEqual(vectors["action"].shape, (10, 2))
        self.assertTrue(np.allclose(vectors["goal"], [1.0, 0.0, 1.0]))
        self.assertTrue(np.allclose(vectors["action"][:, 0], 0.05))
        self.assertTrue(np.allclose(vectors["robot_state"][:, 0], 0.05))

    def test_per_scan_bearing_parameters_are_stored(self) -> None:
        vectors = np.load(self.output / "vectors.npz")
        self.assertTrue(
            np.allclose(vectors["scan_angle_increment_rad"], TWO_PI / (BEAMS + 1))
        )
        self.assertTrue(np.all(vectors["scan_beam_count"] == BEAMS))

    def test_samples_and_rejections_files_are_consistent(self) -> None:
        samples = [
            json.loads(line)
            for line in (self.output / "samples.jsonl").read_text().splitlines()
        ]
        rejections = [
            line
            for line in (self.output / "rejections.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(len(samples), 10)
        self.assertEqual(len(rejections), 0)
        self.assertEqual([row["row"] for row in samples], list(range(10)))

    def test_calibration_is_verified_numerically(self) -> None:
        verification = self.manifest["calibration"]["transform_verification"]
        self.assertTrue(verification["matches"])
        self.assertLess(verification["translation_error_mm"], 0.01)
        self.assertEqual(self.manifest["run_level_codes_retained"], [])

    def test_camera_diagnostic_declares_frame_and_boundary_semantics(self) -> None:
        diagnostic = self.manifest["calibration"]["derived_camera_fov"]
        self.assertIn("camera optical frame", diagnostic["bearing_convention"])
        self.assertIn("positive toward +x/image-right", diagnostic["bearing_convention"])
        self.assertIn("0 <= u < width", diagnostic["image_boundary_definition"])
        self.assertIn("not a physical FOV measurement", diagnostic["interpretation"])

    def test_lidar_geometry_is_constant_and_recorded(self) -> None:
        geometry = self.manifest["calibration"]["lidar_geometry"]
        self.assertTrue(geometry["angular_frame_constant"])
        self.assertEqual(geometry["beam_counts_observed"], [BEAMS])
        self.assertEqual(geometry["max_beam_count"], BEAMS)

    def test_no_timestamp_regressions_in_a_clean_bag(self) -> None:
        events = self.manifest["counts"]["timestamp_regression_events"]
        self.assertEqual(set(events.values()), {0})

    def test_every_mcap_shard_is_hashed(self) -> None:
        shards = [key for key in self.manifest["inputs"] if key.startswith("mcap[")]
        self.assertTrue(shards)
        for key in shards:
            self.assertIsNotNone(self.manifest["inputs"][key]["sha256"])

    def test_output_files_are_hashed(self) -> None:
        for name, entry in self.manifest["outputs"].items():
            with self.subTest(name=name):
                self.assertIsNotNone(entry["sha256"])
                if name == "rejections.jsonl":
                    # Correctly empty: this bag has nothing to reject. The file is
                    # still created and hashed so its absence stays distinguishable
                    # from an export that never wrote it.
                    self.assertEqual(entry["size_bytes"], 0)
                else:
                    self.assertGreater(entry["size_bytes"], 0)

    def test_preprocessing_declares_nothing_was_applied(self) -> None:
        preprocessing = self.manifest["preprocessing"]
        self.assertFalse(preprocessing["resize_applied"])
        self.assertFalse(preprocessing["normalization_applied"])
        self.assertFalse(preprocessing["lidar_tokenization_applied"])
        self.assertEqual(preprocessing["capture_size"], [320, 240])


@unittest.skipUnless(ROS_AVAILABLE, "requires rosbag2_py and ROS message packages")
class DrvFSPostCloseHashRegressionTests(unittest.TestCase):
    def test_memmaps_are_unmapped_before_hash_and_manifest_matches_disk(self) -> None:
        """Exercise the frozen WSL deployment's repository-backed DrvFS path."""

        from livifuser_nav import provenance

        # The frozen WSL checkout lives on /mnt/d. Keeping the temporary export
        # under the checkout exercises DrvFS there while remaining portable to
        # native Linux clones and cleaning only this test's unique directory.
        with tempfile.TemporaryDirectory(prefix=".drvfs-export-", dir=REPO_ROOT) as name:
            root = Path(name)
            bag = _write_bag(root)
            output = root / "export"

            module_name = "_livifuser_export_drvfs_regression"
            exporter_path = REPO_ROOT / "scripts" / "export_pilot_dataset.py"
            spec = importlib.util.spec_from_file_location(module_name, exporter_path)
            assert spec is not None and spec.loader is not None
            exporter = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = exporter
            spec.loader.exec_module(exporter)

            mappings = []
            output_hashes = []
            original_open_memmap = exporter.np.lib.format.open_memmap
            original_sha256_file = provenance.sha256_file

            def tracking_open_memmap(*args, **kwargs):
                array = original_open_memmap(*args, **kwargs)
                mappings.append(array)
                return array

            def checking_sha256_file(path: Path) -> str:
                path = Path(path)
                if path.parent == output:
                    output_hashes.append(path.name)
                    self.assertEqual(len(mappings), 2)
                    self.assertTrue(
                        all(array._mmap.closed for array in mappings),
                        f"output hashing began with a writable memmap open: {path}",
                    )
                return original_sha256_file(path)

            argv = [
                str(exporter_path),
                str(bag),
                "--output",
                str(output),
                "--environment-id",
                "synthetic_drvfs",
                "--view",
                "policy",
            ]
            try:
                with (
                    mock.patch.object(
                        exporter.np.lib.format,
                        "open_memmap",
                        side_effect=tracking_open_memmap,
                    ),
                    mock.patch.object(
                        provenance,
                        "sha256_file",
                        side_effect=checking_sha256_file,
                    ),
                    mock.patch.object(sys, "argv", argv),
                ):
                    self.assertEqual(exporter.main(), 0)
            finally:
                sys.modules.pop(module_name, None)

            self.assertEqual(len(mappings), 2)
            self.assertTrue(all(array._mmap.closed for array in mappings))
            self.assertEqual(
                set(output_hashes),
                {
                    "rgb_320x240_rgb8.npy",
                    "scan_ranges.npy",
                    "vectors.npz",
                    "samples.jsonl",
                    "rejections.jsonl",
                },
            )
            manifest = json.loads((output / "manifest.json").read_text())
            for filename, declared in manifest["outputs"].items():
                path = output / filename
                with self.subTest(filename=filename):
                    self.assertEqual(declared["sha256"], original_sha256_file(path))
                    self.assertEqual(declared["size_bytes"], path.stat().st_size)


@unittest.skipUnless(ROS_AVAILABLE, "requires rosbag2_py and ROS message packages")
class ExportGuardTests(unittest.TestCase):
    def test_refuses_to_overwrite_a_populated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bag = _write_bag(root)
            output = root / "export"
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "export_pilot_dataset.py"),
                str(bag),
                "--output",
                str(output),
                "--environment-id",
                "synthetic",
            ]
            first = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(second.returncode, 1)
            self.assertIn("refusing to overwrite", second.stderr)
            forced = subprocess.run(
                [*command, "--force"], capture_output=True, text=True, check=False
            )
            self.assertEqual(forced.returncode, 0, forced.stderr)


@unittest.skipUnless(ROS_AVAILABLE, "requires rosbag2_py and ROS message packages")
class SimulationExportTests(unittest.TestCase):
    def test_rgb8_simulation_bag_is_exported_and_declared(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bag = _write_bag(root, image_encoding="rgb8")
            output = root / "export"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "export_pilot_dataset.py"),
                    str(bag),
                    "--output",
                    str(output),
                    "--environment-id",
                    "synthetic_sim",
                    "--domain",
                    "simulation",
                    "--view",
                    "policy",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["domain"], "simulation")
            self.assertEqual(manifest["preprocessing"]["source_encoding"], "rgb8")
            self.assertEqual(manifest["preprocessing"]["alpha_values_observed"], [])
            rgb = np.load(output / "rgb_320x240_rgb8.npy", mmap_mode="r")
            self.assertEqual(rgb.shape, (10, 240, 320, 3))


if __name__ == "__main__":
    unittest.main()
