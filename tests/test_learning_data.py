"""Unit tests for Stage 2 loading, windowing, and sensor preprocessing.

Everything here runs against a synthesized policy export written to a temporary
directory, so the expected window counts, beam geometry, and projection results
are known exactly and the suite never depends on a gitignored artifact. One
optional test cross-checks the real `_05`/`_06` exports when they are present.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from livifuser_nav.learning_data import (
    RGB_MEAN,
    RGB_STD,
    ExportRun,
    WindowDataset,
    camera_from_lidar,
    preprocess_rgb,
    quaternion_matrix_xyzw,
    source_to_model_pixels,
    tokenize_lidar,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

ACCEPTED_K = [
    316.21156, 0.0, 223.13834,
    0.0, 315.6497, 107.39364,
    0.0, 0.0, 1.0,
]
ACCEPTED_D = [0.012344, 0.038138, -0.016819, 0.004823, 0.0]
ACCEPTED_T = [0.0723955522, 0.0048472604, -0.0838973150]
ACCEPTED_Q = [-0.4806489642, 0.5212435451, -0.4930249275, 0.5041905996]

#: The LDS-03 frame the exporter records: bearings start at zero, not at -pi.
ANGLE_MIN_RAD = 0.0
RANGE_MIN_M = 0.10000000149011612
RANGE_MAX_M = 100.0

#: The observed beam-count band, and the driver's matching increment rule.
MAX_BEAM_COUNT = 404


def increment_for(beam_count: int) -> float:
    return 2.0 * math.pi / (beam_count + 1)


def manifest_for(run_id: str, view: str = "policy") -> dict:
    return {
        "run_id": run_id,
        "view": view,
        "export_schema_version": "test",
        "calibration": {
            "recorded_camera_info": {
                "width": 320,
                "height": 240,
                "distortion_model": "plumb_bob",
                "k": ACCEPTED_K,
                "d": ACCEPTED_D,
            },
            "static_transforms": {
                "base_scan->camera": {
                    "translation": ACCEPTED_T,
                    "quaternion_xyzw": ACCEPTED_Q,
                }
            },
            "lidar_geometry": {
                "angular_frame": {
                    "angle_min_rad": ANGLE_MIN_RAD,
                    "angle_max_rad": 6.2831854820251465,
                    "range_min_m": RANGE_MIN_M,
                    "range_max_m": RANGE_MAX_M,
                }
            },
        },
    }


def write_export(
    root: Path,
    *,
    segment_lengths: list[int],
    run_id: str = "synthetic_run",
    view: str = "policy",
    rgb_shape: tuple[int, int, int] | None = None,
    goal_columns: int = 3,
) -> Path:
    """Write a minimal but structurally faithful policy export."""

    root.mkdir(parents=True, exist_ok=True)
    count = int(sum(segment_lengths))
    segment_id = np.concatenate(
        [np.full(length, index, dtype=np.int64) for index, length in enumerate(segment_lengths)]
    )
    rows = np.arange(count, dtype=np.float32)
    rgb_shape = rgb_shape or (240, 320, 3)
    rgb = np.zeros((count, *rgb_shape), dtype=np.uint8)
    rgb[:, 0, 0, 0] = np.arange(count, dtype=np.uint8)
    np.save(root / "rgb_320x240_rgb8.npy", rgb)

    beam_counts = 396 + (np.arange(count, dtype=np.int64) % 9)
    ranges = np.full((count, MAX_BEAM_COUNT), np.nan, dtype=np.float32)
    for row, beams in enumerate(beam_counts):
        ranges[row, :beams] = 1.0 + 0.001 * row
    np.save(root / "scan_ranges.npy", ranges)

    np.savez(
        root / "vectors.npz",
        segment_id=segment_id,
        goal=np.tile(np.asarray([1.0, 0.0, 1.0], dtype=np.float32), (count, 1))[
            :, :goal_columns
        ],
        robot_state=np.stack((rows * 0.01, rows * -0.02), axis=1).astype(np.float32),
        action=np.stack((rows * 0.001, rows * 0.002), axis=1).astype(np.float32),
        scan_angle_increment_rad=np.asarray(
            [increment_for(int(beams)) for beams in beam_counts], dtype=np.float32
        ),
        scan_beam_count=beam_counts,
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest_for(run_id, view), indent=2), encoding="utf-8"
    )
    return root


class ExportRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_loads_accepted_policy_export(self) -> None:
        path = write_export(self.root / "ok", segment_lengths=[20])
        run = ExportRun(path)
        self.assertEqual(run.count, 20)
        self.assertEqual(run.run_id, "synthetic_run")
        self.assertEqual(run.rgb.shape, (20, 240, 320, 3))

    def test_sensor_view_export_is_refused(self) -> None:
        path = write_export(self.root / "sensor", segment_lengths=[20], view="sensor")
        with self.assertRaisesRegex(ValueError, "not a policy export"):
            ExportRun(path)

    def test_unexpected_rgb_shape_is_refused(self) -> None:
        path = write_export(self.root / "rgb", segment_lengths=[5], rgb_shape=(240, 320, 4))
        with self.assertRaisesRegex(ValueError, "unexpected RGB shape"):
            ExportRun(path)

    def test_cached_training_load_does_not_materialize_rgb(self) -> None:
        path = write_export(self.root / "rgb-light", segment_lengths=[5])
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["outputs"] = {
            "rgb_320x240_rgb8.npy": {"sha256": "A" * 64, "size_bytes": 1}
        }
        manifest_path.write_text(json.dumps(manifest))
        (path / "rgb_320x240_rgb8.npy").unlink()
        run = ExportRun(path, load_rgb=False)
        self.assertIsNone(run.rgb)
        self.assertEqual(run.count, 5)

    def test_unexpected_goal_width_is_refused(self) -> None:
        path = write_export(self.root / "goal", segment_lengths=[5], goal_columns=2)
        with self.assertRaisesRegex(ValueError, r"goal must have shape"):
            ExportRun(path)

    def test_missing_vector_field_is_refused(self) -> None:
        path = write_export(self.root / "partial", segment_lengths=[5])
        loaded = dict(np.load(path / "vectors.npz"))
        loaded.pop("scan_angle_increment_rad")
        np.savez(path / "vectors.npz", **loaded)
        with self.assertRaisesRegex(ValueError, "scan_angle_increment_rad"):
            ExportRun(path)


class WindowConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_window_count_matches_the_exporter_formula(self) -> None:
        # Per segment: len - K - H + 2, and segments shorter than K + H - 1 add none.
        path = write_export(self.root / "counts", segment_lengths=[20, 5, 30])
        dataset = WindowDataset([path], context_k=8, horizon_h=8)
        self.assertEqual(len(dataset), (20 - 14) + 0 + (30 - 14))

    def test_short_segment_alone_yields_no_windows(self) -> None:
        path = write_export(self.root / "short", segment_lengths=[14])
        self.assertEqual(len(WindowDataset([path], context_k=8, horizon_h=8)), 0)

    def test_windows_never_cross_a_segment_boundary(self) -> None:
        path = write_export(self.root / "boundary", segment_lengths=[20, 30])
        dataset = WindowDataset([path], context_k=8, horizon_h=8)
        segment_id = np.load(path / "vectors.npz")["segment_id"]
        for ref in dataset.windows:
            rows = list(ref.context_rows) + list(ref.action_rows)
            self.assertEqual(len(set(segment_id[rows].tolist())), 1)

    def test_window_rows_are_contiguous_and_aligned(self) -> None:
        path = write_export(self.root / "aligned", segment_lengths=[25])
        dataset = WindowDataset([path], context_k=8, horizon_h=8)
        for ref in dataset.windows:
            self.assertEqual(len(ref.context_rows), 8)
            self.assertEqual(len(ref.action_rows), 8)
            self.assertEqual(np.diff(ref.context_rows).tolist(), [1] * 7)
            self.assertEqual(np.diff(ref.action_rows).tolist(), [1] * 7)
            self.assertEqual(ref.origin_row, ref.context_rows[-1])
            # The first commanded action is the one issued at the context origin.
            self.assertEqual(ref.action_rows[0], ref.origin_row)

    def test_targets_are_the_recorded_actions_for_the_action_rows(self) -> None:
        path = write_export(self.root / "targets", segment_lengths=[20])
        dataset = WindowDataset([path], context_k=8, horizon_h=8)
        actions = np.load(path / "vectors.npz")["action"]
        ref = dataset.windows[0]
        targets = dataset.targets(ref)
        self.assertEqual(targets.shape, (8, 2))
        np.testing.assert_allclose(targets, actions[list(ref.action_rows)])

    def test_windows_are_ordered_per_run_for_deterministic_selection(self) -> None:
        first = write_export(self.root / "run_a", segment_lengths=[20], run_id="a")
        second = write_export(self.root / "run_b", segment_lengths=[24], run_id="b")
        dataset = WindowDataset([first, second], context_k=8, horizon_h=8)
        indices = [ref.run_index for ref in dataset.windows]
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(indices.count(0), 6)
        self.assertEqual(indices.count(1), 10)

    def test_non_contiguous_segment_rows_are_refused(self) -> None:
        path = write_export(self.root / "scrambled", segment_lengths=[20, 20])
        loaded = dict(np.load(path / "vectors.npz"))
        segment_id = loaded["segment_id"].copy()
        segment_id[5] = 1
        loaded["segment_id"] = segment_id
        np.savez(path / "vectors.npz", **loaded)
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            WindowDataset([path], context_k=8, horizon_h=8)

    def test_non_positive_window_dimensions_are_refused(self) -> None:
        path = write_export(self.root / "dims", segment_lengths=[20])
        with self.assertRaisesRegex(ValueError, "must be positive"):
            WindowDataset([path], context_k=0, horizon_h=8)


class PreprocessingTests(unittest.TestCase):
    def test_output_contract(self) -> None:
        image = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        tensor = preprocess_rgb(image)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags["C_CONTIGUOUS"])

    def test_letterbox_bands_are_normalized_zero_and_content_is_not(self) -> None:
        image = np.full((240, 320, 3), 200, dtype=np.uint8)
        tensor = preprocess_rgb(image)
        np.testing.assert_array_equal(tensor[:, :28, :], 0.0)
        np.testing.assert_array_equal(tensor[:, 196:, :], 0.0)
        expected = (200.0 / 255.0 - RGB_MEAN) / RGB_STD
        for channel in range(3):
            np.testing.assert_allclose(
                tensor[channel, 28:196, :], expected[channel], rtol=0, atol=1e-5
            )

    def test_full_field_of_view_is_retained_rather_than_cropped(self) -> None:
        # A marker in each source corner must survive; a centre crop would lose them.
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        for row in (0, 239):
            for column in (0, 319):
                image[row, column] = 255
        tensor = preprocess_rgb(image)
        self.assertGreater(tensor[0, 28, 0], tensor[0, 28, 112])
        self.assertGreater(tensor[0, 28, 223], tensor[0, 28, 112])
        self.assertGreater(tensor[0, 195, 0], tensor[0, 112, 112])
        self.assertGreater(tensor[0, 195, 223], tensor[0, 112, 112])

    def test_preprocessing_is_deterministic(self) -> None:
        image = np.random.default_rng(1).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        np.testing.assert_array_equal(preprocess_rgb(image), preprocess_rgb(image))

    def test_wrong_dtype_or_shape_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            preprocess_rgb(np.zeros((240, 320, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            preprocess_rgb(np.zeros((240, 320, 4), dtype=np.uint8))

    def test_source_to_model_pixel_mapping_matches_the_letterbox(self) -> None:
        u, v = source_to_model_pixels(
            np.asarray([0.0, 160.0, 320.0]), np.asarray([0.0, 120.0, 240.0])
        )
        np.testing.assert_allclose(u, [0.0, 112.0, 224.0])
        np.testing.assert_allclose(v, [28.0, 112.0, 196.0])


class TransformTests(unittest.TestCase):
    def test_quaternion_matrix_is_orthonormal_and_right_handed(self) -> None:
        rotation = quaternion_matrix_xyzw(ACCEPTED_Q)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

    def test_unnormalized_quaternion_is_normalized(self) -> None:
        scaled = [2.0 * value for value in ACCEPTED_Q]
        np.testing.assert_allclose(
            quaternion_matrix_xyzw(scaled), quaternion_matrix_xyzw(ACCEPTED_Q), atol=1e-12
        )

    def test_zero_quaternion_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero quaternion"):
            quaternion_matrix_xyzw([0.0, 0.0, 0.0, 0.0])

    def test_camera_from_lidar_inverts_the_recorded_transform(self) -> None:
        manifest = manifest_for("t")
        rotation, translation = camera_from_lidar(manifest)
        forward = quaternion_matrix_xyzw(ACCEPTED_Q)
        offset = np.asarray(ACCEPTED_T)
        point_camera = np.asarray([0.31, -0.07, 1.4])
        point_lidar = forward @ point_camera + offset
        np.testing.assert_allclose(rotation @ point_lidar + translation, point_camera, atol=1e-12)


class LidarTokenizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = manifest_for("lidar")

    def scan(self, beam_count: int, value: float = 1.5) -> np.ndarray:
        ranges = np.full(MAX_BEAM_COUNT, np.nan, dtype=np.float32)
        ranges[:beam_count] = value
        return ranges

    def test_nan_padding_is_not_read_as_a_return(self) -> None:
        beam_count = 397
        padded = self.scan(beam_count)
        exact = padded[:beam_count].copy()
        increment = increment_for(beam_count)
        from_padded = tokenize_lidar(padded, beam_count, increment, self.manifest)
        from_exact = tokenize_lidar(exact, beam_count, increment, self.manifest)
        np.testing.assert_array_equal(from_padded.features, from_exact.features)
        # Every sector is fully valid: the padding never entered any sector.
        np.testing.assert_allclose(from_padded.features[:, 3], 1.0)

    def test_bearings_use_the_per_scan_increment(self) -> None:
        # Reusing one scan's increment for another beam count misplaces the far
        # sectors by several beam spacings. Using each scan's own increment
        # keeps the same physical bearing in the same sector to within the
        # sector-boundary rounding, which is under one beam spacing.
        beam_spacing_deg = math.degrees(increment_for(400))
        narrow = tokenize_lidar(self.scan(397), 397, increment_for(397), self.manifest)
        wide = tokenize_lidar(self.scan(402), 402, increment_for(402), self.manifest)
        difference_deg = np.degrees(
            np.abs(narrow.sector_bearing_rad - wide.sector_bearing_rad)
        )
        self.assertLess(float(difference_deg.max()), beam_spacing_deg)

        wrong_increment = tokenize_lidar(self.scan(402), 402, increment_for(397), self.manifest)
        wrong_deg = np.degrees(
            np.abs(narrow.sector_bearing_rad - wrong_increment.sector_bearing_rad)
        )
        self.assertGreater(float(wrong_deg.max()), 4.0 * beam_spacing_deg)

    def test_sector_bearings_span_the_full_circle_in_order(self) -> None:
        tokens = tokenize_lidar(self.scan(399), 399, increment_for(399), self.manifest)
        bearings = tokens.sector_bearing_rad
        self.assertEqual(bearings.shape, (80,))
        self.assertTrue(np.all(np.diff(bearings) > 0))
        self.assertGreaterEqual(float(bearings[0]), ANGLE_MIN_RAD)
        self.assertLess(float(bearings[-1]), ANGLE_MIN_RAD + 2.0 * math.pi)
        np.testing.assert_allclose(
            tokens.features[:, 1], np.sin(bearings), atol=1e-6
        )
        np.testing.assert_allclose(
            tokens.features[:, 2], np.cos(bearings), atol=1e-6
        )

    def test_out_of_frame_returns_are_invalid_and_default_to_clipped_range(self) -> None:
        beam_count = 400
        ranges = self.scan(beam_count, value=2.0)
        ranges[:5] = 0.0  # below range_min: the driver's no-return encoding
        ranges[5:10] = np.inf
        tokens = tokenize_lidar(ranges, beam_count, increment_for(beam_count), self.manifest)
        self.assertLess(float(tokens.features[0, 3]), 1.0)
        # A sector with no valid return reports maximum normalized range and a
        # zero validity fraction; the placeholder range is what the projection
        # then uses, so the mask still describes where such a sector would fall.
        empty = tokenize_lidar(
            np.zeros(beam_count, dtype=np.float32),
            beam_count,
            increment_for(beam_count),
            self.manifest,
        )
        np.testing.assert_allclose(empty.features[:, 0], 1.0)
        np.testing.assert_allclose(empty.features[:, 3], 0.0)
        np.testing.assert_array_equal(empty.visual_mask.any(axis=1), empty.in_fov)

    def test_range_is_normalized_against_the_clip(self) -> None:
        beam_count = 400
        tokens = tokenize_lidar(
            self.scan(beam_count, value=2.5), beam_count, increment_for(beam_count),
            self.manifest, range_clip_m=10.0,
        )
        np.testing.assert_allclose(tokens.features[:, 0], 0.25, atol=1e-6)
        far = tokenize_lidar(
            self.scan(beam_count, value=40.0), beam_count, increment_for(beam_count),
            self.manifest, range_clip_m=10.0,
        )
        np.testing.assert_allclose(far.features[:, 0], 1.0, atol=1e-6)

    def test_invalid_beam_count_or_sector_count_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the stored scan payload"):
            tokenize_lidar(self.scan(400), 500, increment_for(400), self.manifest)
        with self.assertRaisesRegex(ValueError, "outside the stored scan payload"):
            tokenize_lidar(self.scan(400), 0, increment_for(400), self.manifest)
        with self.assertRaisesRegex(ValueError, "sectors must be between"):
            tokenize_lidar(self.scan(400), 400, increment_for(400), self.manifest, sectors=0)

    def test_sector_partition_covers_every_beam_exactly_once(self) -> None:
        for beam_count in range(396, 405):
            edges = [sector * beam_count // 80 for sector in range(81)]
            self.assertEqual(edges[0], 0)
            self.assertEqual(edges[-1], beam_count)
            self.assertTrue(all(b > a for a, b in zip(edges, edges[1:], strict=False)))


class ProjectionMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = manifest_for("projection")
        beam_count = 400
        ranges = np.full(beam_count, 1.5, dtype=np.float32)
        self.tokens = tokenize_lidar(
            ranges, beam_count, increment_for(beam_count), self.manifest
        )

    def test_forward_sectors_are_in_view_and_rear_sectors_are_not(self) -> None:
        bearings = self.tokens.sector_bearing_rad
        forward = int(np.argmin(np.abs(bearings)))
        rear = int(np.argmin(np.abs(bearings - math.pi)))
        self.assertTrue(bool(self.tokens.in_fov[forward]))
        self.assertFalse(bool(self.tokens.in_fov[rear]))

    def test_in_view_sectors_are_a_contiguous_arc_of_plausible_width(self) -> None:
        # ~52 degrees of horizontal coverage out of 360 over 80 sectors.
        visible = int(self.tokens.in_fov.sum())
        self.assertGreaterEqual(visible, 8)
        self.assertLessEqual(visible, 16)
        rolled = np.roll(self.tokens.in_fov, 40)
        boundaries = int(np.sum(rolled[1:] != rolled[:-1]))
        self.assertEqual(boundaries, 2)

    def test_mask_is_true_exactly_where_a_sector_is_in_view(self) -> None:
        # The fusion attention relies on this: an in-view LiDAR token always has
        # at least one compatible visual patch, so its softmax is never empty.
        np.testing.assert_array_equal(self.tokens.visual_mask.any(axis=1), self.tokens.in_fov)

    def test_mask_is_a_local_neighbourhood_not_the_whole_grid(self) -> None:
        counts = self.tokens.visual_mask.sum(axis=1)[self.tokens.in_fov]
        self.assertTrue(bool(np.all(counts >= 4)))
        self.assertTrue(bool(np.all(counts <= 9)))

    def test_mask_radius_zero_selects_a_single_patch(self) -> None:
        beam_count = 400
        tokens = tokenize_lidar(
            np.full(beam_count, 1.5, dtype=np.float32),
            beam_count,
            increment_for(beam_count),
            self.manifest,
            visual_radius=0,
        )
        counts = tokens.visual_mask.sum(axis=1)[tokens.in_fov]
        np.testing.assert_array_equal(counts, 1)

    def test_mask_shape_and_dtype(self) -> None:
        self.assertEqual(self.tokens.visual_mask.shape, (80, 49))
        self.assertEqual(self.tokens.visual_mask.dtype, np.bool_)
        self.assertEqual(self.tokens.in_fov.dtype, np.bool_)
        self.assertEqual(self.tokens.features.shape, (80, 4))
        self.assertEqual(self.tokens.features.dtype, np.float32)

    def test_projection_accounts_for_the_off_centre_principal_point(self) -> None:
        # cx is 63.1 px right of centre, so the visible arc is asymmetric about
        # the robot's forward axis rather than centred on it.
        bearings = self.tokens.sector_bearing_rad[self.tokens.in_fov]
        signed = np.arctan2(np.sin(bearings), np.cos(bearings))
        self.assertGreater(float(signed.max() + signed.min()), 0.05)

    def test_closer_obstacles_move_up_the_image(self) -> None:
        # The camera sits 8.4 cm below the scan plane, so the scan plane is
        # above the optical axis and a nearer return projects higher. This is
        # the sign check that catches a flipped vertical axis or a transform
        # applied in the wrong direction.
        beam_count = 400
        near = tokenize_lidar(
            np.full(beam_count, 0.5, dtype=np.float32), beam_count,
            increment_for(beam_count), self.manifest, visual_radius=0,
        )
        far = tokenize_lidar(
            np.full(beam_count, 6.0, dtype=np.float32), beam_count,
            increment_for(beam_count), self.manifest, visual_radius=0,
        )
        shared = near.in_fov & far.in_fov
        self.assertTrue(bool(shared.any()))
        near_rows = np.argmax(near.visual_mask[shared], axis=1) // 7
        far_rows = np.argmax(far.visual_mask[shared], axis=1) // 7
        self.assertTrue(bool(np.all(near_rows <= far_rows)))
        self.assertTrue(bool(np.any(near_rows < far_rows)))


ACCEPTED_EXPORTS = {
    "keyboard_obstacle_balanced_2026-07-30_05": (
        REPO_ROOT / "artifacts/export/keyboard_obstacle_balanced_policy_git_2e3848c"
    ),
    "keyboard_pipeline_pilot_2026-07-30_06": (
        REPO_ROOT / "artifacts/export/keyboard_pipeline_pilot_2026-07-30_06_policy_git_b40ea11"
    ),
}


@unittest.skipUnless(
    all(path.is_dir() for path in ACCEPTED_EXPORTS.values()),
    "accepted `_05`/`_06` exports are not present (they are gitignored evidence)",
)
class AcceptedExportTests(unittest.TestCase):
    """Cross-check the loader against the two exports the gate actually uses."""

    def test_window_counts_match_the_exporter_manifest(self) -> None:
        for run_id, path in ACCEPTED_EXPORTS.items():
            with self.subTest(run=run_id):
                dataset = WindowDataset([path], context_k=8, horizon_h=8)
                manifest = json.loads((path / "manifest.json").read_text("utf-8"))
                self.assertEqual(manifest["run_id"], run_id)
                self.assertEqual(len(dataset), manifest["contiguity"]["windowable_k8_h8"])

    def test_recorded_beam_counts_stay_inside_the_stored_payload(self) -> None:
        for run_id, path in ACCEPTED_EXPORTS.items():
            with self.subTest(run=run_id):
                run = ExportRun(path)
                beams = np.asarray(run.vectors["scan_beam_count"])
                self.assertTrue(bool(np.all(beams <= run.scan_ranges.shape[1])))
                self.assertTrue(bool(np.all(beams >= 80)))

    def test_first_row_tokenizes_against_the_recorded_calibration(self) -> None:
        for run_id, path in ACCEPTED_EXPORTS.items():
            with self.subTest(run=run_id):
                run = ExportRun(path)
                tokens = tokenize_lidar(
                    run.scan_ranges[0],
                    int(run.vectors["scan_beam_count"][0]),
                    float(run.vectors["scan_angle_increment_rad"][0]),
                    run.manifest,
                )
                self.assertEqual(tokens.features.shape, (80, 4))
                self.assertTrue(bool(np.all(np.isfinite(tokens.features))))
                np.testing.assert_array_equal(
                    tokens.visual_mask.any(axis=1), tokens.in_fov
                )


if __name__ == "__main__":
    unittest.main()
