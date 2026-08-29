import copy
import unittest

from livifuser_nav.export_schema import EXPORT_SCHEMA_VERSION
from livifuser_nav.manifest_schema import (
    REQUIRED_FIELDS,
    assert_manifest_valid,
    validate_manifest,
)


def valid_manifest() -> dict:
    """A minimal manifest satisfying every declared field."""

    return {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "run_id": "stationary_pilot_2026-07-29_01",
        "environment_id": "lab_stationary",
        "domain": "hardware",
        "view": "sensor",
        "code": {
            "git_revision": None,
            "git_state": "no_commits_or_not_a_repository",
            "source_tree_sha256": "A" * 64,
            "source_files": {"src/livifuser_nav/sampling.py": "B" * 64},
        },
        "environment": {
            "python_version": "3.10.12",
            "generated_at_utc": "2026-07-30T00:00:00+00:00",
        },
        "inputs": {"mcap[0] x.mcap": {"sha256": "C" * 64}},
        "effective_configuration": {"view": "sensor"},
        "effective_configuration_sha256": "D" * 64,
        "association_policy": {"streams": {"camera": {}}, "grid_rate_hz": 10.0},
        "action_topic": "/cmd_vel",
        "action_timestamp_source": "bag_receive_timestamp",
        "run_level_codes_retained": [],
        "run_level_codes_downgraded": [],
        "lidar_association_mode": "nearest",
        "lidar_future_selection": {
            "all_grid_ticks": {"count": 121},
            "lidar_eligible_ticks": {"count": 106},
            "accepted_samples": {"count": 106},
        },
        "preprocessing": {
            "capture_size": [320, 240],
            "stored_encoding": "rgb8",
            "resize_applied": False,
            "normalization_applied": False,
            "lidar_tokenization_applied": False,
        },
        "calibration": {
            "recorded_camera_info": {"k": [1.0]},
            "camera_info_message_count": 363,
            "camera_info_distinct_variants": [{"k": [1.0]}],
            "derived_camera_fov": {
                "fx": 316.2,
                "bearing_convention": "camera optical frame; positive=image-right",
                "image_boundary_definition": "continuous u=[0,width]; accept 0 <= u < width",
            },
            "static_transforms": {"base_scan->camera": {}},
            "transform_verification": {"matches": True},
            "lidar_geometry": {"angular_frame_constant": True},
        },
        "counts": {
            "grid_ticks": 121,
            "accepted_samples": 106,
            "rejected_samples": 15,
            "acceptance_rate": 0.876,
            "timestamp_regression_events": {"camera": 0},
        },
        "rejections": {"by_primary_reason": {}, "by_any_reason": {}},
        "contiguity": {"segment_lengths": [106], "windowable_k8_h8": 92},
        "outputs": {"vectors.npz": {"sha256": "E" * 64}},
    }


class ValidManifestTests(unittest.TestCase):
    def test_reference_manifest_validates(self) -> None:
        self.assertEqual(validate_manifest(valid_manifest()), [])

    def test_assert_does_not_raise_on_valid_manifest(self) -> None:
        assert_manifest_valid(valid_manifest())

    def test_nullable_fields_accept_null(self) -> None:
        manifest = valid_manifest()
        manifest["code"]["git_revision"] = None
        manifest["calibration"]["transform_verification"] = None
        manifest["calibration"]["derived_camera_fov"] = None
        manifest["calibration"]["recorded_camera_info"] = None
        self.assertEqual(validate_manifest(manifest), [])


class MissingFieldTests(unittest.TestCase):
    def test_every_declared_field_is_actually_required(self) -> None:
        # Removing any one declared field must produce at least one problem.
        for spec in REQUIRED_FIELDS:
            with self.subTest(path=spec.path):
                manifest = copy.deepcopy(valid_manifest())
                parts = spec.path.split(".")
                container = manifest
                for part in parts[:-1]:
                    container = container[part]
                del container[parts[-1]]
                self.assertTrue(
                    validate_manifest(manifest),
                    f"{spec.path} was declared required but its absence passed",
                )

    def test_missing_top_level_field_is_named(self) -> None:
        manifest = valid_manifest()
        del manifest["run_id"]
        problems = validate_manifest(manifest)
        self.assertTrue(any("run_id" in problem for problem in problems))

    def test_missing_nested_parent_is_reported_not_crashed(self) -> None:
        manifest = valid_manifest()
        del manifest["code"]
        problems = validate_manifest(manifest)
        self.assertTrue(any("code.source_tree_sha256" in p for p in problems))

    def test_assert_raises_and_lists_problems(self) -> None:
        manifest = valid_manifest()
        del manifest["outputs"]
        del manifest["run_id"]
        with self.assertRaises(ValueError) as caught:
            assert_manifest_valid(manifest)
        message = str(caught.exception)
        self.assertIn("outputs", message)
        self.assertIn("run_id", message)

    def test_schema_1_2_requires_fov_bearing_convention(self) -> None:
        manifest = valid_manifest()
        del manifest["calibration"]["derived_camera_fov"]["bearing_convention"]
        problems = validate_manifest(manifest)
        self.assertTrue(any("bearing_convention" in problem for problem in problems))

    def test_historical_schema_1_1_does_not_gain_new_requirements(self) -> None:
        manifest = valid_manifest()
        manifest["export_schema_version"] = "1.1.0"
        manifest["calibration"]["derived_camera_fov"] = {"fx": 316.2}
        self.assertEqual(validate_manifest(manifest), [])

    def test_schema_1_3_requires_domain(self) -> None:
        manifest = valid_manifest()
        del manifest["domain"]
        problems = validate_manifest(manifest)
        self.assertTrue(any("domain" in problem for problem in problems))

    def test_historical_schema_1_2_does_not_require_domain(self) -> None:
        manifest = valid_manifest()
        manifest["export_schema_version"] = "1.2.0"
        del manifest["domain"]
        self.assertEqual(validate_manifest(manifest), [])


class TypeTests(unittest.TestCase):
    def test_wrong_type_is_rejected(self) -> None:
        manifest = valid_manifest()
        manifest["counts"]["grid_ticks"] = "121"
        problems = validate_manifest(manifest)
        self.assertTrue(any("counts.grid_ticks" in problem for problem in problems))

    def test_bool_does_not_satisfy_an_int_field(self) -> None:
        # bool subclasses int in Python; a boolean count is a real bug.
        manifest = valid_manifest()
        manifest["counts"]["accepted_samples"] = True
        problems = validate_manifest(manifest)
        self.assertTrue(any("counts.accepted_samples" in problem for problem in problems))

    def test_int_does_not_satisfy_a_bool_field(self) -> None:
        manifest = valid_manifest()
        manifest["preprocessing"]["resize_applied"] = 0
        problems = validate_manifest(manifest)
        self.assertTrue(any("resize_applied" in problem for problem in problems))

    def test_non_nullable_field_rejects_null(self) -> None:
        manifest = valid_manifest()
        manifest["code"]["source_tree_sha256"] = None
        problems = validate_manifest(manifest)
        self.assertTrue(any("source_tree_sha256" in problem for problem in problems))

    def test_empty_value_is_rejected_where_non_empty_required(self) -> None:
        for path, value in (
            ("run_id", ""),
            ("inputs", {}),
            ("outputs", {}),
        ):
            with self.subTest(path=path):
                manifest = valid_manifest()
                manifest[path] = value
                self.assertTrue(validate_manifest(manifest))

    def test_empty_list_is_allowed_where_not_marked_non_empty(self) -> None:
        manifest = valid_manifest()
        manifest["contiguity"]["segment_lengths"] = []
        manifest["run_level_codes_retained"] = []
        self.assertEqual(validate_manifest(manifest), [])

    def test_float_accepted_for_numeric_rate(self) -> None:
        manifest = valid_manifest()
        manifest["association_policy"]["grid_rate_hz"] = 10
        self.assertEqual(validate_manifest(manifest), [])


class SpecHygieneTests(unittest.TestCase):
    def test_no_duplicate_paths(self) -> None:
        paths = [spec.path for spec in REQUIRED_FIELDS]
        self.assertEqual(len(paths), len(set(paths)))

    def test_every_spec_declares_at_least_one_type(self) -> None:
        for spec in REQUIRED_FIELDS:
            self.assertTrue(spec.types, spec.path)


if __name__ == "__main__":
    unittest.main()
