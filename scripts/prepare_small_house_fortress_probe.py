#!/usr/bin/env python3
"""Prepare the archived AWS Small House world for an excluded Fortress probe.

The upstream ROS 2 branch targets Gazebo Classic.  Its static models commonly
contain a mass-only ``<inertial>`` block, which SDFormat 13 / Fortress rejects.
This adapter copies the assets into a disposable artifact directory, marks
every imported model static, removes inertial blocks that static models do not
use, inserts the repaired LiViFuser Burger, and adds the project's Fortress
system plugins.  It does not produce a confirmatory world or layered JSON.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

PROBE_WORLD_NAME = "aws_small_house_fortress_probe"
PROBE_ROBOT_POSE = "3.5 1.0 0.002 0 0 0"
PROBE_ROBOT_XY = (3.5, 1.0)
PROBE_ASSET_RADIUS_M = 4.0
ALWAYS_KEEP_MODEL_TOKENS = ("FloorB", "HouseWallB", "RoomWall")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _set_static(model: ET.Element) -> None:
    static = model.find("static")
    if static is None:
        static = ET.SubElement(model, "static")
    static.text = "true"
    for link in model.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            link.remove(inertial)


def _sanitize_models(models_root: Path) -> int:
    count = 0
    for sdf_path in sorted(models_root.glob("*/model.sdf")):
        tree = ET.parse(sdf_path)
        root = tree.getroot()
        models = root.findall("model")
        if len(models) != 1:
            raise ValueError(f"{sdf_path} must contain exactly one top-level model")
        _set_static(models[0])
        ET.indent(tree, space="  ")
        tree.write(sdf_path, encoding="utf-8", xml_declaration=True)
        count += 1
    if count == 0:
        raise ValueError(f"no model.sdf files found under {models_root}")
    return count


def _localize_external_photos(models_root: Path, photos_root: Path) -> int:
    pattern = re.compile(r"\.\./\.\./\.\./\.\./photos/([^<]+)")
    repaired = 0
    for dae_path in sorted(models_root.glob("*/meshes/*.DAE")):
        text = dae_path.read_text(encoding="utf-8")
        names = sorted(set(pattern.findall(text)))
        if not names:
            continue
        for name in names:
            source = photos_root / name
            if not source.is_file():
                raise ValueError(f"referenced portrait does not exist: {source}")
            shutil.copy2(source, dae_path.parent / name)
        dae_path.write_text(pattern.sub(lambda match: match.group(1), text), encoding="utf-8")
        repaired += len(names)
    return repaired


def _flatten_legacy_model_includes(world: ET.Element) -> int:
    converted = 0
    children = list(world)
    for index, child in enumerate(children):
        if child.tag != "model":
            continue
        nested_include = child.find("include")
        if nested_include is None:
            continue
        direct_include = copy.deepcopy(nested_include)
        name = ET.Element("name")
        name.text = child.get("name")
        direct_include.insert(0, name)
        pose = child.find("pose")
        if pose is not None:
            direct_include.append(copy.deepcopy(pose))
        static = ET.Element("static")
        static.text = "true"
        direct_include.append(static)
        world.remove(child)
        world.insert(index, direct_include)
        converted += 1
    return converted


def _filter_probe_includes(world: ET.Element) -> tuple[int, int]:
    kept = 0
    removed = 0
    for include in list(world.findall("include")):
        name = include.findtext("name", default="")
        pose_text = include.findtext("pose", default="")
        fields = pose_text.split()
        if len(fields) != 6:
            raise ValueError(f"imported model {name!r} has no six-field pose")
        x_m, y_m = float(fields[0]), float(fields[1])
        distance_m = math.hypot(x_m - PROBE_ROBOT_XY[0], y_m - PROBE_ROBOT_XY[1])
        keep = distance_m <= PROBE_ASSET_RADIUS_M or any(
            token in name for token in ALWAYS_KEEP_MODEL_TOKENS
        )
        if keep:
            kept += 1
        else:
            world.remove(include)
            removed += 1
    return kept, removed


def _prepare_world(
    source_world: Path, template: Path, output_world: Path
) -> tuple[int, int, int, int]:
    source_tree = ET.parse(source_world)
    source_root = source_tree.getroot()
    template_tree = ET.parse(template)
    template_root = template_tree.getroot()
    source_root.set("version", template_root.get("version", "1.8"))
    world = source_root.find("world")
    if world is None:
        raise ValueError("AWS source has no SDF world")
    world.set("name", PROBE_WORLD_NAME)
    removed_empty_pose_frames = 0
    for pose_element in world.iter("pose"):
        if pose_element.get("frame") == "":
            del pose_element.attrib["frame"]
            removed_empty_pose_frames += 1
    flattened_include_count = _flatten_legacy_model_includes(world)
    if flattened_include_count == 0:
        raise ValueError("AWS source unexpectedly had no nested model includes")
    kept_include_count, removed_include_count = _filter_probe_includes(world)

    gui = world.find("gui")
    if gui is not None:
        world.remove(gui)
    template_world = template_root.find("world")
    if template_world is None:
        raise ValueError("LiViFuser template has no SDF world")
    source_physics = world.find("physics")
    if source_physics is not None:
        world.remove(source_physics)
    template_physics = template_world.find("physics")
    if template_physics is None:
        raise ValueError("LiViFuser template has no physics profile")
    world.insert(0, copy.deepcopy(template_physics))
    insert_at = 1
    for plugin in list(world.findall("plugin")):
        world.remove(plugin)
    for plugin in template_world.findall("plugin"):
        world.insert(insert_at, copy.deepcopy(plugin))
        insert_at += 1

    robot = template_world.find("model[@name='livifuser_burger']")
    if robot is None:
        raise ValueError("LiViFuser template has no repaired Burger")
    robot = copy.deepcopy(robot)
    pose = robot.find("pose")
    if pose is None:
        pose = ET.SubElement(robot, "pose")
    pose.text = PROBE_ROBOT_POSE
    world.append(robot)

    ET.indent(source_tree, space="  ")
    source_tree.write(output_world, encoding="utf-8", xml_declaration=True)
    if removed_empty_pose_frames == 0:
        raise ValueError("AWS source unexpectedly had no legacy empty pose frames")
    return (
        removed_empty_pose_frames,
        flattened_include_count,
        kept_include_count,
        removed_include_count,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    template = args.template.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing probe directory: {output}")

    source_world = source / "worlds" / "small_house.world"
    source_models = source / "models"
    source_photos = source / "photos"
    source_license = source / "LICENSE"
    for required in (
        source_world,
        source_models,
        source_photos,
        source_license,
        template,
    ):
        if not required.exists():
            raise SystemExit(f"required input does not exist: {required}")

    output.mkdir(parents=True)
    shutil.copytree(source_models, output / "models")
    shutil.copy2(source_license, output / "LICENSE.aws-small-house")
    sanitized_count = _sanitize_models(output / "models")
    localized_photo_count = _localize_external_photos(output / "models", source_photos)
    output_world = output / "aws_small_house_fortress_probe.sdf"
    (
        removed_empty_pose_frames,
        flattened_include_count,
        kept_include_count,
        removed_include_count,
    ) = _prepare_world(source_world, template, output_world)

    manifest = {
        "schema_version": "1.0.0",
        "status": "EXCLUDED_VISUAL_COMPATIBILITY_PROBE",
        "confirmatory_use_permitted": False,
        "source": "https://github.com/aws-robotics/aws-robomaker-small-house-world",
        "source_revision": args.source_revision,
        "source_license": "MIT",
        "source_world_sha256": _sha256(source_world),
        "livifuser_template_sha256": _sha256(template),
        "prepared_world_sha256": _sha256(output_world),
        "world_name": PROBE_WORLD_NAME,
        "robot_pose_xyz_rpy": [3.5, 1.0, 0.002, 0.0, 0.0, 0.0],
        "sanitized_static_model_count": sanitized_count,
        "localized_external_photo_count": localized_photo_count,
        "removed_empty_pose_frame_attributes": removed_empty_pose_frames,
        "flattened_legacy_model_includes": flattened_include_count,
        "probe_asset_radius_m": PROBE_ASSET_RADIUS_M,
        "kept_imported_model_instances": kept_include_count,
        "removed_out_of_radius_model_instances": removed_include_count,
        "compatibility_changes": [
            "all imported models forced static",
            "unused mass-only inertial blocks removed from imported static models",
            "legacy external portrait references localized beside their meshes",
            "Gazebo Classic GUI block removed",
            "obsolete empty pose frame attributes removed",
            "legacy model-wrapped includes converted to top-level SDF includes",
            "probe restricted to the textured shell and nearby model instances",
            "Gazebo Classic physics profile replaced by proven Fortress profile",
            "LiViFuser Fortress systems inserted",
            "repaired LiViFuser Burger and measured camera geometry inserted",
        ],
    }
    (output / "PROBE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
