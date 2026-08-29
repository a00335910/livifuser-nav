"""Materialize a schema-2 layered world as Gazebo Fortress SDF.

The analytic scan and the rendered camera must consume the same generated
world.  Keeping JSON and SDF generation separate without a mechanical seam
would make it possible to test one obstacle layout while rendering another.
This module therefore treats the schema-2 JSON as authoritative and uses the
tracked lab SDF only as a robot/sensor/physics template.

Profile-switchable obstacles are rendered at 0.12 m, below the measured
0.172 m scan plane, in both paired members.  C0 is an explicit oracle-LiDAR
control in which the analytic layer reports that otherwise identical object;
C4 removes only that analytic return.  Camera pixels and collision geometry
remain identical, and the C4 member is physically realizable.

C1 is a checksum-pinned camera-visible scene intervention.  It changes scene
illumination and visual material colors after all geometry is materialized, so
collision shapes, camera intrinsics, analytic LiDAR, and expert labels cannot
be affected by its implementation path.
"""

from __future__ import annotations

import copy
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .visual_conditions import (
    load_c1_visual_contract,
    validate_c1_condition_descriptor,
)
from .visual_skin import (
    mesh_envelope,
    model_uri,
    validate_visual_skin_descriptor,
)
from .world_generator import LIDAR_SCAN_HEIGHT_M
from .world_layers import LAYER_CAMERA, LAYER_COLLISION, parse_world

_PRESERVED_MODELS = {"ground", "livifuser_burger"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_]+")


def _number(value: float) -> str:
    return f"{value:.9g}"


def _text(parent: ET.Element, tag: str, value: str) -> ET.Element:
    element = ET.SubElement(parent, tag)
    element.text = value
    return element


def _geometry(parent: ET.Element, item: dict, height_m: float) -> None:
    geometry = ET.SubElement(parent, "geometry")
    if item["type"] == "box":
        box = ET.SubElement(geometry, "box")
        size_x, size_y = item["size_xy_m"]
        _text(
            box,
            "size",
            f"{_number(float(size_x))} {_number(float(size_y))} {_number(height_m)}",
        )
    else:
        cylinder = ET.SubElement(geometry, "cylinder")
        _text(cylinder, "radius", _number(float(item["radius_m"])))
        _text(cylinder, "length", _number(height_m))


def _box_geometry(parent: ET.Element, size_xyz: tuple[float, float, float]) -> None:
    geometry = ET.SubElement(parent, "geometry")
    box = ET.SubElement(geometry, "box")
    _text(box, "size", " ".join(_number(value) for value in size_xyz))


def _visual_material(
    visual: ET.Element,
    rgba: tuple[float, ...],
) -> None:
    material = ET.SubElement(visual, "material")
    color = " ".join(_number(value) for value in rgba)
    _text(material, "ambient", color)
    _text(material, "diffuse", color)


def _mesh_geometry(
    visual: ET.Element,
    uri: str,
    scale: tuple[float, float, float],
) -> None:
    geometry = ET.SubElement(visual, "geometry")
    mesh = ET.SubElement(geometry, "mesh")
    _text(mesh, "uri", uri)
    _text(mesh, "scale", " ".join(_number(value) for value in scale))


def _render_fields(item: dict, *, low_profile: bool) -> tuple[float, tuple[float, ...]]:
    render = item.get("render")
    if not isinstance(render, dict):
        raise ValueError(f"{item['name']} must declare a render object")
    height_m = float(render.get("height_m", math.nan))
    if not math.isfinite(height_m) or height_m <= 0.0:
        raise ValueError(f"{item['name']}.render.height_m must be finite and positive")
    color = render.get("color_rgba")
    if not isinstance(color, list) or len(color) != 4:
        raise ValueError(f"{item['name']}.render.color_rgba must contain four values")
    rgba = tuple(float(value) for value in color)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in rgba):
        raise ValueError(f"{item['name']}.render.color_rgba must be finite in [0, 1]")
    if low_profile and height_m >= LIDAR_SCAN_HEIGHT_M:
        raise ValueError(
            f"{item['name']} is the structural C4 obstacle but reaches "
            f"{height_m:.3f} m, not below the {LIDAR_SCAN_HEIGHT_M:.3f} m scan plane"
        )
    return height_m, rgba


def _wall_visuals(
    link: ET.Element,
    item: dict,
    height_m: float,
    style: str,
) -> None:
    size_x, size_y = (float(value) for value in item["size_xy_m"])
    skirting_height = min(0.14, height_m * 0.25)
    wall_height = height_m - skirting_height
    skirting = ET.SubElement(link, "visual", {"name": "skirting_visual"})
    _text(
        skirting,
        "pose",
        f"0 0 {_number(-height_m / 2.0 + skirting_height / 2.0)} 0 0 0",
    )
    _box_geometry(skirting, (size_x, size_y, skirting_height))
    _visual_material(skirting, (0.20, 0.24, 0.29, 1.0))

    wall = ET.SubElement(link, "visual", {"name": "wall_visual"})
    _text(wall, "pose", f"0 0 {_number(skirting_height / 2.0)} 0 0 0")
    _mesh_geometry(
        wall,
        model_uri(f"meshes/wall_box_{style}_c0.obj"),
        (size_x, size_y, wall_height),
    )


def _mesh_visual(
    link: ET.Element,
    item: dict,
    height_m: float,
    style: str,
    role: str,
) -> None:
    envelope = mesh_envelope(role)
    minimum = tuple(float(value) for value in envelope["min_xyz"])
    maximum = tuple(float(value) for value in envelope["max_xyz"])
    source_size = tuple(maximum[i] - minimum[i] for i in range(3))
    if role == "furniture":
        size_x, size_y = (float(value) for value in item["size_xy_m"])
        scales = (
            0.92 * size_x / source_size[0],
            0.92 * size_y / source_size[1],
            0.92 * height_m / source_size[2],
        )
    else:
        radius = float(item["radius_m"])
        radial_extent = float(envelope["radial_extent_m"])
        planar_scale = 0.92 * radius / radial_extent
        scales = (planar_scale, planar_scale, 0.92 * height_m / source_size[2])
    centres = tuple((minimum[i] + maximum[i]) / 2.0 for i in range(3))
    pose_x = -centres[0] * scales[0]
    pose_y = -centres[1] * scales[1]
    pose_z = -centres[2] * scales[2]
    visual = ET.SubElement(link, "visual", {"name": f"{role}_visual"})
    _text(
        visual,
        "pose",
        f"{_number(pose_x)} {_number(pose_y)} {_number(pose_z)} 0 0 0",
    )
    geometry = ET.SubElement(visual, "geometry")
    mesh = ET.SubElement(geometry, "mesh")
    _text(mesh, "uri", model_uri(f"meshes/{role}_{style}_c0.dae"))
    _text(mesh, "scale", " ".join(_number(value) for value in scales))


def _primitive_visual(
    link: ET.Element,
    item: dict,
    height_m: float,
    style: str,
    role: str,
) -> None:
    visual = ET.SubElement(link, "visual", {"name": f"{role}_visual"})
    if item["type"] == "box":
        size_x, size_y = (float(value) for value in item["size_xy_m"])
        geometry_role = "hazard_box"
        scale = (size_x, size_y, height_m)
    else:
        radius = float(item["radius_m"])
        geometry_role = "hazard_cylinder"
        scale = (2.0 * radius, 2.0 * radius, height_m)
    _mesh_geometry(
        visual,
        model_uri(f"meshes/{geometry_role}_{style}_c0.obj"),
        scale,
    )


def _obstacle_model(item: dict, *, low_profile: bool, style: str) -> ET.Element:
    height_m, _rgba = _render_fields(item, low_profile=low_profile)
    safe_name = _SAFE_NAME.sub("_", str(item["name"])).strip("_")
    if not safe_name:
        raise ValueError("obstacle name does not contain an SDF-safe character")
    model = ET.Element("model", {"name": f"generated_{safe_name}"})
    _text(model, "static", "true")
    centre_x, centre_y = (float(value) for value in item["center_xy_m"])
    yaw = float(item.get("yaw_rad", 0.0))
    _text(
        model,
        "pose",
        f"{_number(centre_x)} {_number(centre_y)} {_number(height_m / 2.0)} 0 0 {_number(yaw)}",
    )
    link = ET.SubElement(model, "link", {"name": "link"})
    if item["layers"][LAYER_COLLISION]:
        collision = ET.SubElement(link, "collision", {"name": "collision"})
        _geometry(collision, item, height_m)
    if item["layers"][LAYER_CAMERA]:
        if low_profile:
            _primitive_visual(link, item, height_m, style, "hazard")
        elif height_m >= 0.9 and item["type"] == "box":
            _wall_visuals(link, item, height_m, style)
        elif item["type"] == "box":
            _mesh_visual(link, item, height_m, style, "furniture")
        else:
            _mesh_visual(link, item, height_m, style, "cylinder")
    return model


def _set_ground(world_element: ET.Element, payload: dict, style: str) -> None:
    ground = world_element.find("model[@name='ground']")
    if ground is None:
        raise ValueError("SDF template has no ground model")
    minimum_x, minimum_y = (float(value) for value in payload["bounds_min_xy_m"])
    maximum_x, maximum_y = (float(value) for value in payload["bounds_max_xy_m"])
    margin = 0.5
    size_x = maximum_x - minimum_x + 2.0 * margin
    size_y = maximum_y - minimum_y + 2.0 * margin
    centre_x = (minimum_x + maximum_x) / 2.0
    centre_y = (minimum_y + maximum_y) / 2.0
    link = ground.find("link")
    if link is None:
        raise ValueError("SDF template ground has no link")
    for parent in (link.find("collision"), link.find("visual")):
        if parent is None:
            raise ValueError("SDF template ground lacks collision or visual geometry")
        size = parent.find("geometry/box/size")
        pose = parent.find("pose")
        if size is None or pose is None:
            raise ValueError("SDF template ground geometry is incomplete")
        size.text = f"{_number(size_x)} {_number(size_y)} 0.1"
        pose.text = f"{_number(centre_x)} {_number(centre_y)} -0.05 0 0 0"
    ground_visual = link.find("visual")
    if ground_visual is None:
        raise ValueError("SDF template ground has no visual")
    geometry = ground_visual.find("geometry")
    if geometry is None:
        raise ValueError("SDF template ground visual has no geometry")
    ground_visual.remove(geometry)
    _mesh_geometry(
        ground_visual,
        model_uri(f"meshes/floor_box_{style}_c0.obj"),
        (size_x, size_y, 0.1),
    )
    material = ground_visual.find("material")
    if material is not None:
        ground_visual.remove(material)


def _set_robot_start(world_element: ET.Element, payload: dict) -> None:
    robot = world_element.find("model[@name='livifuser_burger']")
    if robot is None:
        raise ValueError("SDF template has no livifuser_burger model")
    start = payload.get("start_pose_xy_yaw")
    if not isinstance(start, list) or len(start) != 3:
        raise ValueError("generated world must declare start_pose_xy_yaw")
    x_m, y_m, yaw_rad = (float(value) for value in start)
    pose = robot.find("pose")
    if pose is None:
        pose = ET.SubElement(robot, "pose")
    pose.text = f"{_number(x_m)} {_number(y_m)} 0.002 0 0 {_number(yaw_rad)}"


def _vector_text(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(_number(float(value)) for value in values)


def _required_text(parent: ET.Element, path: str, context: str) -> ET.Element:
    element = parent.find(path)
    if element is None:
        raise ValueError(f"SDF template lacks {context}")
    return element


def _apply_camera_condition(world_element: ET.Element, payload: dict) -> None:
    condition = payload.get("condition", "C0")
    descriptor = payload.get("camera_condition")
    if condition != "C1":
        if descriptor is not None:
            raise ValueError("camera_condition metadata is only valid for C1")
        return

    validate_c1_condition_descriptor(descriptor)
    contract = load_c1_visual_contract()
    scene = contract["scene"]
    light_contract = contract["directional_light"]
    transform = contract["material_transform"]

    scene_element = _required_text(world_element, "scene", "scene")
    _required_text(scene_element, "ambient", "scene ambient").text = _vector_text(
        scene["ambient_rgba"]
    )
    _required_text(scene_element, "background", "scene background").text = (
        _vector_text(scene["background_rgba"])
    )
    _required_text(scene_element, "shadows", "scene shadows").text = str(
        bool(scene["shadows"])
    ).lower()

    light_name = str(light_contract["name"])
    light = world_element.find(f"light[@name='{light_name}']")
    if light is None or light.get("type") != "directional":
        raise ValueError(f"SDF template lacks directional light {light_name}")
    for tag, field in (
        ("pose", "pose_xyz_rpy"),
        ("diffuse", "diffuse_rgba"),
        ("specular", "specular_rgba"),
        ("direction", "direction_xyz"),
    ):
        _required_text(light, tag, f"{light_name} {tag}").text = _vector_text(
            light_contract[field]
        )

    permutation = tuple(int(index) for index in transform["rgb_permutation"])
    scale = tuple(float(value) for value in transform["rgb_scale"])
    textured_mesh_count = 0
    for mesh_uri in world_element.findall(".//visual/geometry/mesh/uri"):
        if mesh_uri.text is None or "livifuser_visual_skin_v1" not in mesh_uri.text:
            continue
        if "_c0." not in mesh_uri.text:
            raise ValueError("textured visual mesh lacks an exact C0 asset suffix")
        mesh_uri.text = mesh_uri.text.replace("_c0.", "_c1.", 1)
        textured_mesh_count += 1
    if textured_mesh_count == 0:
        raise ValueError("C1 textured world contains no texture-aware meshes")
    for material in world_element.findall(".//visual/material"):
        for tag in ("ambient", "diffuse"):
            color = material.find(tag)
            if color is None or color.text is None:
                continue
            source = tuple(float(value) for value in color.text.split())
            if len(source) != 4 or not all(math.isfinite(value) for value in source):
                raise ValueError(f"visual material {tag} must contain four finite values")
            transformed = tuple(
                max(0.0, min(1.0, source[permutation[index]] * scale[index]))
                for index in range(3)
            ) + (source[3],)
            color.text = _vector_text(transformed)


def materialize_world(payload: dict, template_path: Path) -> ET.ElementTree:
    """Return an SDF tree whose camera, collision, and robot match ``payload``."""

    layered_world = parse_world(payload)
    if layered_world.group is None:
        raise ValueError("generated world must declare its group for visual styling")
    validate_visual_skin_descriptor(payload.get("visual_skin"), layered_world.group)
    style = str(payload["visual_skin"]["style"])
    tree = ET.parse(Path(template_path))
    root = tree.getroot()
    world_element = root.find("world")
    if world_element is None:
        raise ValueError("SDF template has no world element")
    world_element.set("name", str(payload["name"]))

    for model in list(world_element.findall("model")):
        if model.get("name") not in _PRESERVED_MODELS:
            world_element.remove(model)
    _set_ground(world_element, payload, style)
    _set_robot_start(world_element, payload)

    low_names = set(payload.get("c4_hidden_from_lidar", []))
    low_names.update(
        str(item["name"])
        for item in payload["obstacles"]
        if item.get("profile_switchable", False)
    )
    for item in payload["obstacles"]:
        if item["layers"][LAYER_COLLISION] or item["layers"][LAYER_CAMERA]:
            world_element.append(
                _obstacle_model(
                    item,
                    low_profile=str(item["name"]) in low_names,
                    style=style,
                )
            )
    _apply_camera_condition(world_element, payload)
    ET.indent(tree, space="  ")
    return tree


def render_world_sdf(payload: dict, template_path: Path) -> str:
    """Serialize one materialized world reproducibly."""

    tree = materialize_world(copy.deepcopy(payload), template_path)
    return ET.tostring(tree.getroot(), encoding="unicode", xml_declaration=True)
