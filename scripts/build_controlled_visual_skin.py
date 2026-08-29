#!/usr/bin/env python3
"""Build the tracked, geometry-safe Small House visual skin.

The source AWS RoboMaker repository is pinned externally and intentionally not
vendored wholesale. This script extracts a small MIT-licensed subset, reduces
textures to the 320x240 camera's useful resolution, creates deterministic
appearance families, and emits exact C1 channel-permuted texture counterparts.
The resulting mesh files are visual-only; collision remains authoritative in
the schema-2 JSON and is never imported from the asset repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SOURCE_REVISION = "ff9631ca6d1db9c1ba656498151464b5ab74aafe"
SOURCE_WORLD_SHA256 = "8EEF40841C462E3F015DC092F24A1B696EA10F8F05B662B529B76DDD720B544F"
SKIN_NAME = "AWS_SMALL_HOUSE_CONTROLLED_SKIN_V1"
MODEL_NAME = "livifuser_visual_skin_v1"
MAX_TEXTURE_EDGE = 512

STYLES = {
    "dev": "train_wood_brick",
    "train": "train_wood_brick",
    "val_id": "val_amber_plaster",
    "test_id": "test_navy_wood",
}

STYLE_SOURCES = {
    "train_wood_brick": {
        "floor": (
            "aws_robomaker_residential_FloorB_01/materials/textures/"
            "aws_FloorB_01.png"
        ),
        "wall": (
            "aws_robomaker_residential_RoomWall_01/materials/textures/"
            "aws_RoomWall_01.png"
        ),
        "rgb_scale": [1.0, 1.0, 1.0],
    },
    "val_amber_plaster": {
        "floor": (
            "aws_robomaker_residential_CoffeeTable_01/materials/textures/"
            "aws_CoffeeTable_01.png"
        ),
        "wall": (
            "aws_robomaker_residential_HouseWallB_01/materials/textures/"
            "aws_HouseWallB_01.png"
        ),
        "rgb_scale": [0.92, 0.98, 1.0],
    },
    "test_navy_wood": {
        "floor": (
            "aws_robomaker_residential_Carpet_01/materials/textures/"
            "aws_Carpet_01.png"
        ),
        "wall": (
            "aws_robomaker_residential_Board_01/materials/textures/"
            "aws_Board_01.png"
        ),
        "rgb_scale": [0.82, 0.92, 1.0],
    },
}

COMMON_TEXTURES = {
    "furniture": (
        "aws_robomaker_residential_CoffeeTable_01/materials/textures/"
        "aws_CoffeeTable_01.png"
    ),
    "cylinder": (
        "aws_robomaker_residential_Trash_01/materials/textures/aws_Trash_01.png"
    ),
}

MESH_SOURCES = {
    "furniture": (
        "aws_robomaker_residential_CoffeeTable_01/meshes/"
        "aws_CoffeeTable_01_visual.DAE"
    ),
    "cylinder": (
        "aws_robomaker_residential_Trash_01/meshes/aws_Trash_01_visual.DAE"
    ),
}

# Physical envelopes include COLLADA unit conversion and node transforms.
MESH_ENVELOPES_M = {
    "furniture": {
        "min_xyz": [-0.65073196, -0.32403709, -0.00000004],
        "max_xyz": [0.66425201, 0.33076843, 0.32169472],
    },
    "cylinder": {
        "min_xyz": [-0.142589, -0.141495, 0.0],
        "max_xyz": [0.142145, 0.141495, 0.343414],
        "radial_extent_m": 0.1428,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _normalized_geometry_sha256(path: Path) -> str:
    if path.suffix.lower() == ".obj":
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith(("mtllib ", "usemtl "))
        ]
        payload = ("\n".join(lines) + "\n").encode()
    elif path.suffix.lower() == ".dae":
        tree = ET.parse(path)
        root = tree.getroot()
        namespace = root.tag.partition("}")[0].lstrip("{")
        for element in root.findall(
            ".//c:library_images/c:image/c:init_from", {"c": namespace}
        ):
            element.text = "TEXTURE_BINDING_REMOVED"
        ET.indent(tree, space="  ")
        payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    else:
        raise ValueError(f"unsupported mesh geometry file: {path}")
    return hashlib.sha256(payload).hexdigest().upper()


def _load_c1_permutation(path: Path) -> tuple[list[int], str]:
    payload = path.read_bytes()
    contract = json.loads(payload)
    permutation = contract["material_transform"]["rgb_permutation"]
    if sorted(permutation) != [0, 1, 2]:
        raise ValueError("C1 RGB permutation is invalid")
    return [int(value) for value in permutation], hashlib.sha256(payload).hexdigest().upper()


def _resample() -> Image.Resampling:
    return Image.Resampling.LANCZOS


def _source_image(path: Path, scale: list[float]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if max(image.size) > MAX_TEXTURE_EDGE:
        ratio = MAX_TEXTURE_EDGE / max(image.size)
        image = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
            _resample(),
        )
    array = np.asarray(image, dtype=np.float32)
    array *= np.asarray(scale, dtype=np.float32)[None, None, :]
    return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), "RGB")


def _hazard_texture() -> Image.Image:
    image = Image.new("RGB", (256, 128), (245, 192, 32))
    draw = ImageDraw.Draw(image)
    stripe = 36
    for offset in range(-image.height, image.width + image.height, stripe * 2):
        draw.polygon(
            [
                (offset, 0),
                (offset + stripe, 0),
                (offset - image.height + stripe, image.height),
                (offset - image.height, image.height),
            ],
            fill=(24, 30, 38),
        )
    return image


def _c1_image(image: Image.Image, permutation: list[int]) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return Image.fromarray(array[:, :, permutation], "RGB")


def _texture_mesh(source: Path, destination: Path, texture_name: str) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    namespace = root.tag.partition("}")[0].lstrip("{")
    # Fortress uses ignition-common4's strict COLLADA loader. ElementTree's
    # default serialization writes ``ns0:COLLADA`` unless the source default
    # namespace is explicitly re-registered; that loader then reports
    # "Missing COLLADA tag" and crashes its render thread. Keep the canonical
    # default namespace used by the upstream AWS asset.
    ET.register_namespace("", namespace)
    ns = {"c": namespace}
    images = root.findall(".//c:library_images/c:image/c:init_from", ns)
    if not images:
        raise ValueError(f"mesh has no embedded texture reference: {source}")
    for image in images:
        image.text = f"../materials/textures/{texture_name}"
    ET.indent(tree, space="  ")
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    # Source XML whitespace can retain Windows newlines even though the tracked
    # asset contract is LF-only. Canonicalize before hashing so the generated
    # manifest matches both the worktree and a clean Git checkout.
    destination.write_bytes(payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))


def _mean_rgb(image: Image.Image) -> list[float]:
    values = np.asarray(image, dtype=np.float32).mean(axis=(0, 1)) / 255.0
    return [round(float(value), 6) for value in values]


def _luminance_std(image: Image.Image) -> float:
    array = np.asarray(image, dtype=np.float32) / 255.0
    luminance = 0.2126 * array[:, :, 0] + 0.7152 * array[:, :, 1] + 0.0722 * array[:, :, 2]
    return round(float(luminance.std()), 6)


def _write_mtl(path: Path, texture_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                "newmtl textured",
                "Ka 1.0 1.0 1.0",
                "Kd 1.0 1.0 1.0",
                "Ks 0.08 0.08 0.08",
                "Ns 16.0",
                f"map_Kd ../materials/textures/{texture_name}",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def _box_obj(mtl_name: str) -> str:
    lines = [
        f"mtllib {mtl_name}",
        "usemtl textured",
        "v -0.5 -0.5 -0.5",
        "v 0.5 -0.5 -0.5",
        "v 0.5 0.5 -0.5",
        "v -0.5 0.5 -0.5",
        "v -0.5 -0.5 0.5",
        "v 0.5 -0.5 0.5",
        "v 0.5 0.5 0.5",
        "v -0.5 0.5 0.5",
        "vt 0 0",
        "vt 1 0",
        "vt 1 1",
        "vt 0 1",
        "f 1/1 2/2 3/3 4/4",
        "f 5/1 8/2 7/3 6/4",
        "f 1/1 5/2 6/3 2/4",
        "f 2/1 6/2 7/3 3/4",
        "f 3/1 7/2 8/3 4/4",
        "f 5/1 1/2 4/3 8/4",
    ]
    return "\n".join(lines) + "\n"


def _cylinder_obj(mtl_name: str, segments: int = 32) -> str:
    lines = [f"mtllib {mtl_name}", "usemtl textured"]
    for index in range(segments + 1):
        angle = 2.0 * math.pi * index / segments
        x_m = 0.5 * math.cos(angle)
        y_m = 0.5 * math.sin(angle)
        lines.append(f"v {x_m:.9g} {y_m:.9g} -0.5")
        lines.append(f"v {x_m:.9g} {y_m:.9g} 0.5")
    lines.extend(["v 0 0 -0.5", "v 0 0 0.5"])
    for index in range(segments + 1):
        u = index / segments
        lines.append(f"vt {u:.9g} 0")
        lines.append(f"vt {u:.9g} 1")
    bottom_center = 2 * (segments + 1) + 1
    top_center = bottom_center + 1
    for index in range(segments):
        bottom = 2 * index + 1
        top = bottom + 1
        next_bottom = bottom + 2
        next_top = top + 2
        lines.append(
            f"f {bottom}/{bottom} {next_bottom}/{next_bottom} "
            f"{next_top}/{next_top} {top}/{top}"
        )
        lines.append(f"f {bottom_center} {next_bottom} {bottom}")
        lines.append(f"f {top_center} {top} {next_top}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-models", type=Path, required=True)
    parser.add_argument("--source-license", type=Path, required=True)
    parser.add_argument("--c1-contract", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (args.model_output.exists() or args.config_output.exists()) and not args.overwrite:
        raise SystemExit("refusing to overwrite an existing visual-skin output")
    if args.overwrite:
        if args.model_output.is_dir():
            shutil.rmtree(args.model_output)
        elif args.model_output.exists():
            raise SystemExit("model output exists but is not a directory")
        if args.config_output.exists():
            args.config_output.unlink()
    permutation, c1_sha256 = _load_c1_permutation(args.c1_contract)
    texture_root = args.model_output / "materials" / "textures"
    mesh_root = args.model_output / "meshes"
    for directory in (texture_root, mesh_root):
        directory.mkdir(parents=True, exist_ok=True)

    source_hashes: dict[str, str] = {}
    statistics: dict[str, dict] = {}
    hazard = _hazard_texture()
    for style, style_data in STYLE_SOURCES.items():
        scale = [float(value) for value in style_data["rgb_scale"]]
        images: dict[str, Image.Image] = {}
        for role in ("floor", "wall"):
            relative = Path(style_data[role])
            source = args.source_models / relative
            source_hashes[relative.as_posix()] = _sha256(source)
            images[role] = _source_image(source, scale)
        for role, relative_text in COMMON_TEXTURES.items():
            relative = Path(relative_text)
            source = args.source_models / relative
            source_hashes[relative.as_posix()] = _sha256(source)
            images[role] = _source_image(source, scale)
        images["hazard"] = hazard

        statistics[style] = {}
        for role, image in images.items():
            c0_path = texture_root / f"{role}_{style}_c0.png"
            c1_path = texture_root / f"{role}_{style}_c1.png"
            image.save(c0_path, optimize=True)
            _c1_image(image, permutation).save(c1_path, optimize=True)
            statistics[style][role] = {
                "c0_mean_rgb": _mean_rgb(image),
                "c0_luminance_std": _luminance_std(image),
            }

    for style in sorted(STYLE_SOURCES):
        for condition in ("c0", "c1"):
            for role, relative_text in MESH_SOURCES.items():
                relative = Path(relative_text)
                source = args.source_models / relative
                source_hashes[relative.as_posix()] = _sha256(source)
                texture_name = f"{role}_{style}_{condition}.png"
                _texture_mesh(
                    source,
                    mesh_root / f"{role}_{style}_{condition}.dae",
                    texture_name,
                )
            for geometry_role, texture_role, builder in (
                ("floor_box", "floor", _box_obj),
                ("wall_box", "wall", _box_obj),
                ("hazard_box", "hazard", _box_obj),
                ("hazard_cylinder", "hazard", _cylinder_obj),
            ):
                stem = f"{geometry_role}_{style}_{condition}"
                texture_name = f"{texture_role}_{style}_{condition}.png"
                _write_mtl(mesh_root / f"{stem}.mtl", texture_name)
                (mesh_root / f"{stem}.obj").write_text(
                    builder(f"{stem}.mtl"), encoding="utf-8", newline="\n"
                )
    license_payload = args.source_license.read_bytes()
    (args.model_output / "LICENSE.aws-small-house").write_bytes(
        license_payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    )
    (args.model_output / "model.config").write_text(
        """<?xml version="1.0"?>
<model>
  <name>LiViFuser Controlled Visual Skin V1</name>
  <version>1.0</version>
  <sdf version="1.8">model.sdf</sdf>
  <description>Visual-only MIT-licensed AWS Small House asset subset.</description>
</model>
""",
        encoding="utf-8",
        newline="\n",
    )
    (args.model_output / "model.sdf").write_text(
        """<?xml version="1.0"?>
<sdf version="1.8">
  <model name="livifuser_visual_skin_v1"><static>true</static></model>
</sdf>
""",
        encoding="utf-8",
        newline="\n",
    )

    generated_hashes = {
        path.relative_to(args.model_output).as_posix(): _sha256(path)
        for path in sorted(args.model_output.rglob("*"))
        if path.is_file()
    }
    geometry_hashes = {
        path.relative_to(args.model_output).as_posix(): _normalized_geometry_sha256(path)
        for path in sorted(mesh_root.iterdir())
        if path.suffix.lower() in {".obj", ".dae"}
    }
    minimum_color_distance = math.inf
    minimum_hazard_std = math.inf
    for style_stats in statistics.values():
        hazard_mean = style_stats["hazard"]["c0_mean_rgb"]
        floor_mean = style_stats["floor"]["c0_mean_rgb"]
        distance = math.sqrt(
            sum((hazard_mean[i] - floor_mean[i]) ** 2 for i in range(3))
        )
        minimum_color_distance = min(minimum_color_distance, distance)
        minimum_hazard_std = min(
            minimum_hazard_std,
            float(style_stats["hazard"]["c0_luminance_std"]),
        )

    config = {
        "schema_version": 1,
        "name": SKIN_NAME,
        "source": "https://github.com/aws-robotics/aws-robomaker-small-house-world",
        "source_revision": SOURCE_REVISION,
        "source_world_sha256": SOURCE_WORLD_SHA256,
        "source_license": "MIT",
        "c1_visual_contract_sha256": c1_sha256,
        "group_styles": STYLES,
        "styles": sorted(STYLE_SOURCES),
        "model_uri": f"model://{MODEL_NAME}",
        "mesh_envelopes_m": MESH_ENVELOPES_M,
        "source_asset_sha256": dict(sorted(source_hashes.items())),
        "generated_asset_sha256": generated_hashes,
        "normalized_mesh_geometry_sha256": geometry_hashes,
        "appearance_statistics": statistics,
        "c4_visibility_gate": {
            "minimum_hazard_to_floor_mean_rgb_distance": 0.20,
            "observed_minimum_mean_rgb_distance": round(minimum_color_distance, 6),
            "minimum_hazard_luminance_std": 0.20,
            "observed_minimum_hazard_luminance_std": round(minimum_hazard_std, 6),
        },
        "geometry_contract": {
            "imported_collision": "forbidden",
            "mesh_visuals": "must remain within authoritative collision envelope",
            "wall_skin": "partition of the authoritative wall box",
            "c0_c4_visuals": "byte-identical",
            "c1_geometry": "normalized mesh geometry unchanged; texture binding only",
        },
    }
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
