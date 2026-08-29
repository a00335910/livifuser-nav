"""Build and verify the deterministic, cache-free RunPod input handoff."""

from __future__ import annotations

import copy
import io
import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from livifuser_nav.backbone_handoff import (
    ZIP_TIMESTAMP,
    _zip_info,
    json_bytes,
    sha256_bytes,
    sha256_file,
)

# NumPy is needed only to *build* the handoff, never to verify or unpack one.
# The unpack script runs on the pod before any virtualenv exists, and NumPy 2.x
# cannot be installed system-wide there: ROS Humble's Python stack on Jammy is
# built against NumPy 1.x. Importing it lazily keeps verification dependency-free.
if TYPE_CHECKING:
    import numpy as np

BUNDLE_ROOT = "livifuser_runpod_input_v1"
MANIFEST_NAME = "RUNPOD_INPUT_MANIFEST.json"
COMPLETE_NAME = "RUNPOD_INPUT_COMPLETE.json"
BACKBONE_SHA256 = "0F9F7CB99A955AE0B817762CC08565F0D3BD820CDD1692D71DC6B05E2CD9E9F3"
POLICY_SHA256 = "3A989EADD0DB8D995993D2042124E34C2C51FAAFC1A9C74EC60106E8A182C162"
SCHEDULE_SELF_SHA256 = "85F48F81D52F82A5CD64D14D704E032617582D7F5462A7B6FD46C5E64899047D"
PROTECTED_CACHE_NAMES = {
    "livifuser_dinov3_splus_cache_v2_bundle.zip",
    "livifuser_dinov3_splus_heldout_cache_v1_bundle.zip",
}


STATIC_FILES = (
    "pyproject.toml",
    "config/runpod_rtx3090_runtime_v1.proposed.json",
    "config/simulation_closed_loop_execution_v1.proposed.json",
    "config/simulation_live_sensor_contract_v1.json",
    "config/simulation_sweep_v1.json",
    "config/dinov3_splus_directml_parity_v1.json",
    "config/calibration/rpi_camera_v3_320x240_2026-07-29.yaml",
    "config/calibration/lidar_camera_extrinsics_2026-07-29.yaml",
    "docs/experiments/PREREGISTRATION_SIM_CLOSED_LOOP_EXECUTION_AMENDMENT_2026-08-24.md",
    "docs/experiments/PREREGISTRATION_SIM_CLOUD_RUNTIME_ROUTE_AMENDMENT_2026-08-24.md",
    "scripts/bootstrap_runpod_runtime.sh",
    "scripts/capture_runpod_provenance.py",
    "scripts/check_runpod_storage.py",
    "scripts/run_live_sim_development_episode.sh",
    # Gate 6 sweep: both topologies x C0/C1/C3b/C4. Development worlds only;
    # the episode runner it calls rejects any non-development world.
    "scripts/run_gate6_development_smoke.sh",
    # Gate 5: replays recorded parity frames and requires bit-identical
    # output. Must run on the execution route, so it ships to the pod.
    "scripts/verify_recorded_input_determinism.py",
    # The confirmatory batch orchestrator and the modules it imports. Without
    # these the pod has no way to execute the frozen plan at all.
    "scripts/run_closed_loop_confirmatory_batch.py",
    "scripts/build_runtime_handoff.py",
    "scripts/audit_runtime_handoff.py",
    "scripts/seal_runtime_attempt.py",
    "scripts/wait_sim_terminal.py",
    "scripts/benchmark_splus_cpu_runtime.py",
    "scripts/benchmark_splus_cuda_runtime.py",
    "scripts/verify_dinov3_splus_backbone_bundle.py",
    "scripts/verify_closed_loop_policy_payload.py",
    "scripts/verify_runpod_input_handoff.py",
    "scripts/unpack_runpod_input_handoff.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_self_hash(value: dict[str, Any], field: str) -> str:
    copied = copy.deepcopy(value)
    copied.pop(field, None)
    return sha256_bytes(json.dumps(copied, sort_keys=True, separators=(",", ":")).encode())


def _npy_bytes(array: np.ndarray) -> bytes:
    import numpy as np

    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _repository_members(root: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    for relative in STATIC_FILES:
        path = root / relative
        _require(path.is_file(), f"required handoff source missing: {relative}")
        members[relative] = path.read_bytes()
    for package_file in (
        "src/livifuser_nav/__init__.py",
        # contracts.py is stdlib-only and unused by the runtime modules, but
        # __init__.py re-exports four dataclasses from it. Omitting it makes
        # the package unimportable on the pod. Ship it rather than editing
        # __init__.py: a divergent copy would misrepresent the source tree.
        "src/livifuser_nav/contracts.py",
        "src/livifuser_nav/confirmatory_plan.py",
        "src/livifuser_nav/runtime_handoff.py",
        "src/livifuser_nav/backbone_handoff.py",
        "src/livifuser_nav/evaluation.py",
        # Imported by scripts/benchmark_splus_cpu_runtime.py for its
        # right-continuous CDF helper. Shipping the module ships code, not
        # held-out data; no held-out artifact is in this bundle.
        "src/livifuser_nav/heldout_evaluation.py",
        "src/livifuser_nav/learning_data.py",
        "src/livifuser_nav/live_association.py",
        "src/livifuser_nav/live_runtime.py",
        "src/livifuser_nav/model.py",
        "src/livifuser_nav/policy_payload.py",
        "src/livifuser_nav/runpod_handoff.py",
        "src/livifuser_nav/simulation_supervision.py",
    ):
        path = root / package_file
        _require(path.is_file(), f"required Python module missing: {package_file}")
        members[package_file] = path.read_bytes()
    ros_root = root / "ros2_ws/src"
    for path in sorted(item for item in ros_root.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(root).as_posix()
        members[relative] = path.read_bytes()
    return members


def _schedule_members(root: Path) -> dict[str, bytes]:
    schedule_path = root / "artifacts/simulation/confirmatory_v3/schedule.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    _require(
        schedule["schedule_sha256_excludes_self"] == SCHEDULE_SELF_SHA256,
        "confirmatory schedule identity drifted",
    )
    episodes = [
        row for row in schedule["episodes"] if 180 <= int(row["ordinal"]) <= 259
    ]
    _require(len(episodes) == 80, "closed-loop schedule subset count drifted")
    _require(
        [int(row["ordinal"]) for row in episodes] == list(range(180, 260)),
        "closed-loop schedule ordinals drifted",
    )
    subset = {
        "schema_version": "1.0.0",
        "status": "FROZEN_CONFIRMATORY_IDENTITIES_NOT_AUTHORIZED_FOR_EXECUTION",
        "source_schedule_sha256_excludes_self": SCHEDULE_SELF_SHA256,
        "source_split": "test_ood",
        "ordinals_inclusive": [180, 259],
        "episode_count": 80,
        "episodes": episodes,
    }
    subset["subset_sha256_excludes_self"] = _canonical_self_hash(
        subset, "subset_sha256_excludes_self"
    )
    members = {
        "artifacts/simulation/closed_loop_schedule_subset_v1.json": json_bytes(subset)
    }
    world_root = root / "artifacts/simulation/confirmatory_v3"
    world_names = sorted(
        {row["world_json"] for row in episodes} | {row["world_sdf"] for row in episodes}
    )
    for relative in world_names:
        path = world_root / relative
        _require(path.is_file(), f"schedule world missing: {relative}")
        key = f"artifacts/simulation/confirmatory_worlds/{Path(relative).name}"
        members[key] = path.read_bytes()
    return members


def _development_world_members(root: Path) -> dict[str, bytes]:
    source = root / "artifacts/simulation/worlds_visual_skin_dev_v1"
    names = (
        "dev_straight_corridor_000.json",
        "dev_straight_corridor_000.sdf",
        "dev_straight_corridor_000.C1.json",
        "dev_straight_corridor_000.C1.sdf",
        "dev_straight_corridor_000.C4.json",
        "dev_straight_corridor_000.C4.sdf",
        "dev_dogleg_corridor_001.json",
        "dev_dogleg_corridor_001.sdf",
        "dev_dogleg_corridor_001.C1.json",
        "dev_dogleg_corridor_001.C1.sdf",
        "dev_dogleg_corridor_001.C4.json",
        "dev_dogleg_corridor_001.C4.sdf",
    )
    output = {}
    for name in names:
        path = source / name
        _require(path.is_file(), f"development world missing: {name}")
        output[f"artifacts/simulation/development_worlds/{name}"] = path.read_bytes()
    return output


def _parity_members(root: Path) -> dict[str, bytes]:
    import numpy as np

    config_path = root / "config/dinov3_splus_directml_parity_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rgbs: list[np.ndarray] = []
    scans: list[np.ndarray] = []
    counts: list[int] = []
    increments: list[float] = []
    goals: list[np.ndarray] = []
    states: list[np.ndarray] = []
    rows_manifest: list[dict[str, Any]] = []
    maximum_beams = 0
    selected: list[tuple[dict[str, Any], int, np.ndarray, np.ndarray, Any]] = []
    for source in config["parity_sources"]:
        source_root = root / source["root"]
        _require(
            sha256_file(source_root / "manifest.json") == source["manifest_sha256"],
            f"parity source manifest drifted: {source['root']}",
        )
        for filename, key in (
            ("rgb_320x240_rgb8.npy", "rgb_sha256"),
            ("scan_ranges.npy", "scan_sha256"),
            ("vectors.npz", "vectors_sha256"),
        ):
            _require(
                sha256_file(source_root / filename) == source[key],
                f"parity source payload drifted: {source['root']}/{filename}",
            )
        rgb = np.load(source_root / "rgb_320x240_rgb8.npy", mmap_mode="r")
        scan = np.load(source_root / "scan_ranges.npy", mmap_mode="r")
        vectors = np.load(source_root / "vectors.npz", allow_pickle=False)
        maximum_beams = max(maximum_beams, int(scan.shape[1]))
        for row in source["rows"]:
            selected.append((source, int(row), rgb, scan, vectors))
    _require(len(selected) == 32, "prospective parity frame count drifted")
    for index, (source, row, rgb, scan, vectors) in enumerate(selected):
        beam_count = int(vectors["scan_beam_count"][row])
        padded = np.full(maximum_beams, np.nan, dtype=np.float32)
        padded[: scan.shape[1]] = np.asarray(scan[row], dtype=np.float32)
        rgbs.append(np.asarray(rgb[row], dtype=np.uint8))
        scans.append(padded)
        counts.append(beam_count)
        increments.append(float(vectors["scan_angle_increment_rad"][row]))
        goals.append(np.asarray(vectors["goal"][row], dtype=np.float32))
        states.append(np.asarray(vectors["robot_state"][row], dtype=np.float32))
        rows_manifest.append(
            {
                "packed_index": index,
                "condition": source["condition"],
                "source_root": source["root"],
                "source_row": row,
            }
        )
    parity_manifest = {
        "schema_version": "1.0.0",
        "status": "FROZEN_EXCLUDED_DEVELOPMENT_PARITY_INPUTS",
        "frame_count": 32,
        "source_config_sha256": sha256_file(config_path),
        "rows": rows_manifest,
        "unavailable_condition": config["unavailable_condition"],
        "heldout_or_confirmatory_inputs_present": False,
    }
    parity_manifest["manifest_sha256_excludes_self"] = _canonical_self_hash(
        parity_manifest, "manifest_sha256_excludes_self"
    )
    prefix = "artifacts/runtime/parity_inputs_v1"
    return {
        f"{prefix}/manifest.json": json_bytes(parity_manifest),
        f"{prefix}/rgb.npy": _npy_bytes(np.stack(rgbs)),
        f"{prefix}/scan_ranges.npy": _npy_bytes(np.stack(scans)),
        f"{prefix}/scan_beam_count.npy": _npy_bytes(np.asarray(counts, dtype=np.int32)),
        f"{prefix}/scan_angle_increment_rad.npy": _npy_bytes(
            np.asarray(increments, dtype=np.float64)
        ),
        f"{prefix}/goal.npy": _npy_bytes(np.stack(goals)),
        f"{prefix}/robot_state.npy": _npy_bytes(np.stack(states)),
    }


LINE_ENDING_SENSITIVE_SUFFIXES = frozenset(
    {".sh", ".py", ".json", ".yaml", ".yml", ".xml", ".cfg", ".msg", ".md", ".txt", ".toml"}
)


def build_runpod_handoff(repository_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite RunPod handoff: {output}")
    members = _repository_members(root)
    members.update(_schedule_members(root))
    members.update(_development_world_members(root))
    members.update(_parity_members(root))
    inputs = {
        "artifacts/livifuser_dinov3_vits16plus_backbone_c93d816_bundle.zip": BACKBONE_SHA256,
        "artifacts/livifuser_sim_closed_loop_policy_payload_v1_bundle.zip": POLICY_SHA256,
    }
    for relative, expected_hash in inputs.items():
        path = root / relative
        _require(path.is_file() and sha256_file(path) == expected_hash, f"input drift: {relative}")
        members[relative] = path.read_bytes()
    _require(
        not any(Path(name).name in PROTECTED_CACHE_NAMES for name in members),
        "protected DINO cache was selected for handoff",
    )
    # A Windows checkout, or a patch script that forgets newline="", can leave
    # CRLF in a shipped text file. On the pod that is not a cosmetic defect:
    # `set -euo pipefail\r` aborts a shell script on its third line, and a
    # CR terminates a shebang so the interpreter is never found. Refuse to build
    # rather than discover it after a 188 MB upload.
    carriage_return_members = sorted(
        name
        for name, payload in members.items()
        if Path(name).suffix in LINE_ENDING_SENSITIVE_SUFFIXES and b"\r" in payload
    )
    _require(
        not carriage_return_members,
        "CRLF line endings in shipped text members: " + ", ".join(carriage_return_members),
    )
    records = [
        {"name": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(members.items())
    ]
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "SEALED_RUNPOD_INPUT_DEVELOPMENT_ONLY",
        "authorization": {
            "confirmatory_launch_authorized": False,
            "physical_robot_allowed": False,
        },
        "outer_inputs": {
            "backbone_bundle_sha256": BACKBONE_SHA256,
            "policy_payload_sha256": POLICY_SHA256,
        },
        "protected_dino_caches_included": False,
        "heldout_predictions_or_features_included": False,
        "members": records,
        "payload_member_count": len(records),
        "zip_contract": {
            "root": BUNDLE_ROOT,
            "compression": "stored",
            "timestamp": "1980-01-01T00:00:00",
            "unix_mode": "100644",
        },
    }
    manifest["manifest_sha256_excludes_self"] = _canonical_self_hash(
        manifest, "manifest_sha256_excludes_self"
    )
    manifest_payload = json_bytes(manifest)
    completion = {
        "schema_version": "1.0.0",
        "status": "complete_development_only",
        "manifest_file_sha256": sha256_bytes(manifest_payload),
        "manifest_sha256_excludes_self": manifest["manifest_sha256_excludes_self"],
        "payload_member_count": len(records),
        "confirmatory_launch_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", allowZip64=True) as archive:
        for name in sorted(members):
            archive.writestr(_zip_info(f"{BUNDLE_ROOT}/{name}"), members[name])
        archive.writestr(_zip_info(f"{BUNDLE_ROOT}/{MANIFEST_NAME}"), manifest_payload)
        archive.writestr(_zip_info(f"{BUNDLE_ROOT}/{COMPLETE_NAME}"), json_bytes(completion))
    return verify_runpod_handoff(output)


def verify_runpod_handoff(bundle_path: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    _require(bundle.is_file(), f"RunPod handoff missing: {bundle}")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)), "duplicate RunPod handoff members")
        _require(archive.testzip() is None, "RunPod handoff CRC failure")
        for info in archive.infolist():
            parts = Path(info.filename).parts
            _require(
                not info.filename.startswith(("/", "\\")) and ".." not in parts,
                f"unsafe RunPod handoff member: {info.filename}",
            )
            _require(
                info.date_time == ZIP_TIMESTAMP
                and info.compress_type == zipfile.ZIP_STORED
                and (info.external_attr >> 16) == 0o100644,
                f"RunPod handoff ZIP metadata drifted: {info.filename}",
            )
        manifest_name = f"{BUNDLE_ROOT}/{MANIFEST_NAME}"
        completion_name = f"{BUNDLE_ROOT}/{COMPLETE_NAME}"
        manifest_payload = archive.read(manifest_name)
        manifest = json.loads(manifest_payload)
        declared = manifest["manifest_sha256_excludes_self"]
        _require(
            declared == _canonical_self_hash(manifest, "manifest_sha256_excludes_self"),
            "RunPod handoff manifest self-hash failed",
        )
        _require(
            manifest["authorization"]["confirmatory_launch_authorized"] is False,
            "RunPod input unexpectedly authorizes confirmatory execution",
        )
        expected = {manifest_name, completion_name}
        for row in manifest["members"]:
            name = f"{BUNDLE_ROOT}/{row['name']}"
            expected.add(name)
            payload = archive.read(name)
            _require(
                len(payload) == row["size_bytes"] and sha256_bytes(payload) == row["sha256"],
                f"RunPod nested member identity failed: {row['name']}",
            )
        _require(set(names) == expected, "RunPod handoff has extra or missing members")
        _require(
            not any(Path(name).name in PROTECTED_CACHE_NAMES for name in names),
            "protected DINO cache present in RunPod handoff",
        )
        completion = json.loads(archive.read(completion_name))
        _require(
            completion["status"] == "complete_development_only"
            and completion["manifest_file_sha256"] == sha256_bytes(manifest_payload)
            and completion["manifest_sha256_excludes_self"] == declared
            and completion["confirmatory_launch_authorized"] is False,
            "RunPod handoff completion marker drifted",
        )
    return {
        "schema_version": "1.0.0",
        "status": "verified_development_only",
        "bundle_path": str(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
        "bundle_sha256": sha256_file(bundle),
        "payload_member_count": len(manifest["members"]),
        "manifest_file_sha256": sha256_bytes(manifest_payload),
        "manifest_sha256_excludes_self": declared,
        "confirmatory_launch_authorized": False,
    }
