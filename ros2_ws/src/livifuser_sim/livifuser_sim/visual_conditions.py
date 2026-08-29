"""Frozen camera-visible scene conditions for the simulation study.

C1 is authored as a scene intervention, not as an image-space corruption.  It
changes illumination and rendered material appearance while leaving geometry,
physics, the analytic LiDAR, expert labels, and camera transport untouched.
The exact JSON bytes are checksum-pinned so a later edit cannot silently change
the confirmatory condition.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

C1_VISUAL_CONTRACT_NAME = "C1_WARM_LOW_LIGHT_V1"
C1_VISUAL_CONTRACT_FILENAME = "c1_visual_condition_v1.json"
C1_VISUAL_CONTRACT_SHA256 = (
    "85A0F2A92E8C9382E865A646940D5F7017AEA3EC963A093427F8FD0F03DE4E4E"
)


def _default_contract_path() -> Path:
    source_path = Path(__file__).resolve().parents[1] / "config" / C1_VISUAL_CONTRACT_FILENAME
    if source_path.is_file():
        return source_path
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError as error:
        raise FileNotFoundError(
            f"cannot locate {C1_VISUAL_CONTRACT_FILENAME} from source or ROS share"
        ) from error
    return (
        Path(get_package_share_directory("livifuser_sim"))
        / "config"
        / C1_VISUAL_CONTRACT_FILENAME
    )


def _finite_vector(value: object, length: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} values must be finite")
    return result


def load_c1_visual_contract(path: Path | None = None) -> dict:
    """Load and validate the checksum-pinned C1 scene contract."""

    contract_path = _default_contract_path() if path is None else Path(path)
    payload_bytes = contract_path.read_bytes()
    digest = hashlib.sha256(payload_bytes).hexdigest().upper()
    if digest != C1_VISUAL_CONTRACT_SHA256:
        raise ValueError(
            f"C1 visual contract checksum {digest} does not match frozen "
            f"{C1_VISUAL_CONTRACT_SHA256}"
        )
    payload = json.loads(payload_bytes)
    if payload.get("schema_version") != 1:
        raise ValueError("C1 visual contract schema_version must be 1")
    if payload.get("name") != C1_VISUAL_CONTRACT_NAME:
        raise ValueError("C1 visual contract name does not match the frozen name")

    scene = payload.get("scene")
    light = payload.get("directional_light")
    transform = payload.get("material_transform")
    if (
        not isinstance(scene, dict)
        or not isinstance(light, dict)
        or not isinstance(transform, dict)
    ):
        raise ValueError("C1 visual contract lacks scene, light, or material transform")
    for field in ("ambient_rgba", "background_rgba"):
        values = _finite_vector(scene.get(field), 4, f"scene.{field}")
        if not all(0.0 <= item <= 1.0 for item in values):
            raise ValueError(f"scene.{field} must be in [0, 1]")
    for field, length in (
        ("pose_xyz_rpy", 6),
        ("diffuse_rgba", 4),
        ("specular_rgba", 4),
        ("direction_xyz", 3),
    ):
        values = _finite_vector(light.get(field), length, f"directional_light.{field}")
        if field.endswith("rgba") and not all(0.0 <= item <= 1.0 for item in values):
            raise ValueError(f"directional_light.{field} must be in [0, 1]")
    permutation = transform.get("rgb_permutation")
    if permutation not in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
        raise ValueError("material_transform.rgb_permutation must be a permutation")
    scale = _finite_vector(transform.get("rgb_scale"), 3, "material_transform.rgb_scale")
    if not all(0.0 <= item <= 1.0 for item in scale):
        raise ValueError("material_transform.rgb_scale must be in [0, 1]")
    if transform.get("alpha") != "preserve":
        raise ValueError("C1 material alpha rule must be preserve")
    return payload


def c1_condition_descriptor() -> dict[str, str]:
    """Return the immutable descriptor embedded in every derived C1 world."""

    load_c1_visual_contract()
    return {
        "name": C1_VISUAL_CONTRACT_NAME,
        "sha256": C1_VISUAL_CONTRACT_SHA256,
    }


def validate_c1_condition_descriptor(value: object) -> None:
    """Reject missing or drifted C1 metadata before materialization."""

    if value != c1_condition_descriptor():
        raise ValueError("world C1 descriptor does not match the frozen visual contract")


def evaluate_c1_development_gate(c0: dict, c1: dict) -> dict:
    """Evaluate the predeclared non-vacuity and non-degeneracy image gate."""

    contract = load_c1_visual_contract()
    thresholds = contract["development_gate"]
    c0_channels = _finite_vector(
        c0.get("channel_mean_rgb_normalized"), 3, "C0 channel means"
    )
    c1_channels = _finite_vector(
        c1.get("channel_mean_rgb_normalized"), 3, "C1 channel means"
    )
    c0_luminance = float(c0.get("luminance_mean_normalized", math.nan))
    c1_luminance = float(c1.get("luminance_mean_normalized", math.nan))
    if not math.isfinite(c0_luminance) or c0_luminance <= 0.0:
        raise ValueError("C0 luminance mean must be finite and positive")
    if not math.isfinite(c1_luminance):
        raise ValueError("C1 luminance mean must be finite")
    channel_delta = sum(
        abs(after - before)
        for before, after in zip(c0_channels, c1_channels, strict=True)
    ) / 3.0
    luminance_ratio = c1_luminance / c0_luminance
    measurements = {
        "mean_absolute_channel_mean_delta": channel_delta,
        "c1_to_c0_luminance_ratio": luminance_ratio,
        "c1_luminance_std": float(c1.get("luminance_std_normalized", math.nan)),
        "c1_near_black_fraction": float(c1.get("near_black_fraction", math.nan)),
        "c1_near_white_fraction": float(c1.get("near_white_fraction", math.nan)),
    }
    issues = []
    comparisons = (
        (
            measurements["mean_absolute_channel_mean_delta"]
            >= thresholds["minimum_mean_absolute_channel_mean_delta"],
            "appearance_shift_too_small",
        ),
        (
            measurements["c1_to_c0_luminance_ratio"]
            >= thresholds["minimum_c1_to_c0_luminance_ratio"],
            "c1_too_dark",
        ),
        (
            measurements["c1_to_c0_luminance_ratio"]
            <= thresholds["maximum_c1_to_c0_luminance_ratio"],
            "illumination_shift_too_small",
        ),
        (
            measurements["c1_luminance_std"]
            >= thresholds["minimum_c1_luminance_std"],
            "c1_contrast_too_low",
        ),
        (
            measurements["c1_near_black_fraction"]
            <= thresholds["maximum_c1_near_black_fraction"],
            "c1_near_black_fraction_too_high",
        ),
        (
            measurements["c1_near_white_fraction"]
            <= thresholds["maximum_c1_near_white_fraction"],
            "c1_near_white_fraction_too_high",
        ),
    )
    for passed, issue in comparisons:
        if not passed:
            issues.append(issue)
    return {
        "valid": not issues,
        "issues": issues,
        "measurements": measurements,
        "thresholds": thresholds,
        "contract": c1_condition_descriptor(),
    }
