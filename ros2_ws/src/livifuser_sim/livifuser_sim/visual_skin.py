"""Checksum-pinned visual skin for geometry-controlled simulation worlds."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

VISUAL_SKIN_NAME = "AWS_SMALL_HOUSE_CONTROLLED_SKIN_V1"
VISUAL_SKIN_CONFIG_FILENAME = "visual_skin_v1.json"
VISUAL_SKIN_CONFIG_SHA256 = (
    "D5312E2EEB304B3FF7FE03FB9DE8B76761D3A63D481EE97FCD27D75BB6DD89B8"
)


def _package_share() -> Path:
    source_root = Path(__file__).resolve().parents[1]
    if (source_root / "config" / VISUAL_SKIN_CONFIG_FILENAME).is_file():
        return source_root
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError as error:
        raise FileNotFoundError("cannot locate livifuser_sim visual skin") from error
    return Path(get_package_share_directory("livifuser_sim"))


@lru_cache(maxsize=1)
def load_visual_skin_contract() -> dict:
    path = _package_share() / "config" / VISUAL_SKIN_CONFIG_FILENAME
    payload_bytes = path.read_bytes()
    digest = hashlib.sha256(payload_bytes).hexdigest().upper()
    if digest != VISUAL_SKIN_CONFIG_SHA256:
        raise ValueError(
            f"visual skin checksum {digest} does not match {VISUAL_SKIN_CONFIG_SHA256}"
        )
    payload = json.loads(payload_bytes)
    if payload.get("schema_version") != 1 or payload.get("name") != VISUAL_SKIN_NAME:
        raise ValueError("visual skin identity does not match the frozen V1 contract")
    styles = payload.get("styles")
    groups = payload.get("group_styles")
    if not isinstance(styles, list) or not styles or not isinstance(groups, dict):
        raise ValueError("visual skin lacks styles or group mapping")
    if set(groups) != {"dev", "train", "val_id", "test_id"}:
        raise ValueError("visual skin must map every generatable world group")
    if not set(groups.values()).issubset(set(styles)):
        raise ValueError("visual skin group mapping names an unknown style")
    gate = payload.get("c4_visibility_gate", {})
    comparisons = (
        (
            "observed_minimum_mean_rgb_distance",
            "minimum_hazard_to_floor_mean_rgb_distance",
        ),
        (
            "observed_minimum_hazard_luminance_std",
            "minimum_hazard_luminance_std",
        ),
    )
    for observed_name, threshold_name in comparisons:
        observed = float(gate.get(observed_name, math.nan))
        threshold = float(gate.get(threshold_name, math.nan))
        if not math.isfinite(observed) or not math.isfinite(threshold) or observed < threshold:
            raise ValueError(f"visual skin fails C4 visibility gate {threshold_name}")
    return payload


def visual_skin_descriptor(group: str) -> dict[str, str]:
    """Return the immutable skin and style bound to a generated world group."""

    contract = load_visual_skin_contract()
    try:
        style = contract["group_styles"][group]
    except KeyError as error:
        raise ValueError(f"visual skin has no style for world group {group!r}") from error
    return {
        "name": VISUAL_SKIN_NAME,
        "sha256": VISUAL_SKIN_CONFIG_SHA256,
        "style": str(style),
    }


def validate_visual_skin_descriptor(value: object, group: str) -> dict:
    expected = visual_skin_descriptor(group)
    if value != expected:
        raise ValueError("world visual_skin descriptor does not match the frozen contract")
    return load_visual_skin_contract()


def model_uri(relative: str) -> str:
    contract = load_visual_skin_contract()
    return f"{contract['model_uri']}/{relative.lstrip('/')}"


def mesh_envelope(role: str) -> dict:
    contract = load_visual_skin_contract()
    try:
        return contract["mesh_envelopes_m"][role]
    except KeyError as error:
        raise ValueError(f"visual skin has no mesh envelope for {role}") from error


def verify_installed_visual_assets() -> dict:
    """Hash every installed visual asset against the pinned configuration."""

    contract = load_visual_skin_contract()
    model_root = _package_share() / "models" / "livifuser_visual_skin_v1"
    issues = []
    for relative, expected in contract["generated_asset_sha256"].items():
        path = model_root / relative
        if not path.is_file():
            issues.append(f"missing:{relative}")
            continue
        observed = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if observed != expected:
            issues.append(f"checksum:{relative}")
    return {
        "valid": not issues,
        "issues": issues,
        "asset_count": len(contract["generated_asset_sha256"]),
        "contract": {
            "name": VISUAL_SKIN_NAME,
            "sha256": VISUAL_SKIN_CONFIG_SHA256,
        },
    }
