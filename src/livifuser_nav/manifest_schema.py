"""Structural validation of an export manifest.

The manifest is the only durable record of how a dataset was produced, so a
misspelled or dropped field is a silent provenance loss: nothing fails, and the
gap is discovered months later when the record is needed. This module declares
the fields that must be present and lets the exporter refuse to finish without
them.

This checks structure, not values. Whether a threshold was *correct* is a
research question; whether it was *recorded* is a contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MISSING = object()


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One required manifest field, addressed by dotted path."""

    path: str
    types: tuple[type, ...]
    allow_none: bool = False
    non_empty: bool = False

    def describe(self) -> str:
        names = "/".join(kind.__name__ for kind in self.types)
        suffix = " or null" if self.allow_none else ""
        return f"{self.path}: {names}{suffix}"


#: Every field the research record needs to reconstruct an export.
REQUIRED_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("export_schema_version", (str,), non_empty=True),
    FieldSpec("run_id", (str,), non_empty=True),
    FieldSpec("environment_id", (str,), non_empty=True),
    FieldSpec("view", (str,), non_empty=True),
    # Code identity: git revision may legitimately be absent, but the source
    # hash and the git state description never may.
    FieldSpec("code.git_revision", (str,), allow_none=True),
    FieldSpec("code.git_state", (str,), non_empty=True),
    FieldSpec("code.source_tree_sha256", (str,), non_empty=True),
    FieldSpec("code.source_files", (dict,), non_empty=True),
    FieldSpec("environment.python_version", (str,), non_empty=True),
    FieldSpec("environment.generated_at_utc", (str,), non_empty=True),
    FieldSpec("inputs", (dict,), non_empty=True),
    FieldSpec("effective_configuration", (dict,), non_empty=True),
    FieldSpec("effective_configuration_sha256", (str,), non_empty=True),
    FieldSpec("association_policy.streams", (dict,), non_empty=True),
    FieldSpec("association_policy.grid_rate_hz", (int, float)),
    FieldSpec("action_topic", (str,), non_empty=True),
    FieldSpec("action_timestamp_source", (str,), non_empty=True),
    FieldSpec("run_level_codes_retained", (list,)),
    FieldSpec("run_level_codes_downgraded", (list,)),
    FieldSpec("lidar_association_mode", (str,), non_empty=True),
    FieldSpec("lidar_future_selection.all_grid_ticks", (dict,), non_empty=True),
    FieldSpec("lidar_future_selection.lidar_eligible_ticks", (dict,), non_empty=True),
    FieldSpec("lidar_future_selection.accepted_samples", (dict,), non_empty=True),
    FieldSpec("preprocessing.capture_size", (list,), non_empty=True),
    FieldSpec("preprocessing.stored_encoding", (str,), non_empty=True),
    FieldSpec("preprocessing.resize_applied", (bool,)),
    FieldSpec("preprocessing.normalization_applied", (bool,)),
    FieldSpec("preprocessing.lidar_tokenization_applied", (bool,)),
    FieldSpec("calibration.recorded_camera_info", (dict,), allow_none=True),
    FieldSpec("calibration.camera_info_message_count", (int,)),
    FieldSpec("calibration.camera_info_distinct_variants", (list,)),
    FieldSpec("calibration.derived_camera_fov", (dict,), allow_none=True),
    FieldSpec("calibration.static_transforms", (dict,)),
    FieldSpec("calibration.transform_verification", (dict,), allow_none=True),
    FieldSpec("calibration.lidar_geometry", (dict,), non_empty=True),
    FieldSpec("counts.grid_ticks", (int,)),
    FieldSpec("counts.accepted_samples", (int,)),
    FieldSpec("counts.rejected_samples", (int,)),
    FieldSpec("counts.acceptance_rate", (int, float)),
    FieldSpec("counts.timestamp_regression_events", (dict,), non_empty=True),
    FieldSpec("rejections.by_primary_reason", (dict,)),
    FieldSpec("rejections.by_any_reason", (dict,)),
    FieldSpec("contiguity.segment_lengths", (list,)),
    FieldSpec("contiguity.windowable_k8_h8", (int,)),
    FieldSpec("outputs", (dict,), non_empty=True),
)

# Additive fields introduced in schema 1.2.0. Keeping these conditional lets
# historical 1.1.0 manifests remain structurally valid while requiring every
# newly generated diagnostic to state the frame in which its bearings live.
FOV_DIAGNOSTIC_FIELDS_1_2: tuple[FieldSpec, ...] = (
    FieldSpec(
        "calibration.derived_camera_fov.bearing_convention",
        (str,),
        non_empty=True,
    ),
    FieldSpec(
        "calibration.derived_camera_fov.image_boundary_definition",
        (str,),
        non_empty=True,
    ),
)

# Additive provenance introduced in schema 1.3.0. Simulation and hardware
# episodes must never become indistinguishable after they enter a dataset.
DOMAIN_FIELDS_1_3: tuple[FieldSpec, ...] = (
    FieldSpec("domain", (str,), non_empty=True),
)


def _resolve(manifest: Mapping[str, Any], path: str) -> Any:
    current: Any = manifest
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return MISSING
        current = current[part]
    return current


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return a list of structural problems; empty means the manifest is valid."""

    problems: list[str] = []
    for spec in REQUIRED_FIELDS:
        value = _resolve(manifest, spec.path)
        if value is MISSING:
            problems.append(f"missing field {spec.describe()}")
            continue
        if value is None:
            if not spec.allow_none:
                problems.append(f"field {spec.path} is null but must be {spec.describe()}")
            continue
        # bool is a subclass of int; keep the two from satisfying each other.
        if bool in spec.types and not isinstance(value, bool):
            problems.append(f"field {spec.path} should be bool, got {type(value).__name__}")
            continue
        if bool not in spec.types and isinstance(value, bool):
            problems.append(f"field {spec.path} should be {spec.describe()}, got bool")
            continue
        if not isinstance(value, spec.types):
            problems.append(
                f"field {spec.path} should be {spec.describe()}, "
                f"got {type(value).__name__}"
            )
            continue
        if spec.non_empty and isinstance(value, (str, Sequence, Mapping)) and len(value) == 0:
            problems.append(f"field {spec.path} must not be empty")

    derived_fov = _resolve(manifest, "calibration.derived_camera_fov")
    if manifest.get("export_schema_version") in {"1.2.0", "1.3.0"} and isinstance(
        derived_fov, Mapping
    ):
        for spec in FOV_DIAGNOSTIC_FIELDS_1_2:
            value = _resolve(manifest, spec.path)
            if value is MISSING:
                problems.append(f"missing field {spec.describe()}")
            elif not isinstance(value, spec.types):
                problems.append(
                    f"field {spec.path} should be {spec.describe()}, "
                    f"got {type(value).__name__}"
                )
            elif spec.non_empty and len(value) == 0:
                problems.append(f"field {spec.path} must not be empty")
    if manifest.get("export_schema_version") == "1.3.0":
        for spec in DOMAIN_FIELDS_1_3:
            value = _resolve(manifest, spec.path)
            if value is MISSING:
                problems.append(f"missing field {spec.describe()}")
            elif not isinstance(value, spec.types):
                problems.append(
                    f"field {spec.path} should be {spec.describe()}, "
                    f"got {type(value).__name__}"
                )
            elif spec.non_empty and len(value) == 0:
                problems.append(f"field {spec.path} must not be empty")
    return problems


def assert_manifest_valid(manifest: Mapping[str, Any]) -> None:
    """Raise :class:`ValueError` listing every structural problem found."""

    problems = validate_manifest(manifest)
    if problems:
        raise ValueError(
            f"manifest failed schema validation ({len(problems)} problems):\n  "
            + "\n  ".join(problems)
        )
