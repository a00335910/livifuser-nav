"""Build and independently audit the immutable closed-loop runtime handoff.

Closed-loop execution amendment section 11. The runtime ZIP carries exactly the
material a confirmatory rollout needs: the verified backbone snapshot, the 12
learned checkpoints with their score references and thresholds, the single
training-only Mahalanobis identity, the frozen configs and schedule subset, the
selected-route contract, the executable runner and supervisor code, and the
recorded-input and Gazebo development-gate evidence.

The builder and the auditor are deliberately separate. The builder decides what
belongs in the bundle; the auditor re-derives every identity from the bundle
alone and rejects anything extra or missing. An auditor that reused the
builder's member list could not detect a builder that shipped the wrong thing,
so it reconstructs the expectation from the amendment's requirements instead.

Nothing here may read held-out or confirmatory material, and the audit fails
closed if any appears.
"""

from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path
from typing import Any

from livifuser_nav.backbone_handoff import (
    _zip_info,
    json_bytes,
    sha256_bytes,
    sha256_file,
)

BUNDLE_ROOT = "livifuser_runtime_v1"
MANIFEST_NAME = f"{BUNDLE_ROOT}/RUNTIME_MANIFEST.json"
COMPLETION_NAME = f"{BUNDLE_ROOT}/RUNTIME_COMPLETE.json"

VARIANTS = ("full", "lidar_only", "concat", "rgb_only")
SEEDS = (20260805, 20260806, 20260807)
EXPECTED_POLICY_IDENTITIES = 12

# Executable code a rollout actually runs. Anything that decides whether data is
# valid must be present so the auditor can hash it.
REQUIRED_CODE = (
    "src/livifuser_nav/live_runtime.py",
    "src/livifuser_nav/live_association.py",
    "src/livifuser_nav/simulation_supervision.py",
    "src/livifuser_nav/model.py",
    "src/livifuser_nav/learning_data.py",
    "src/livifuser_nav/policy_payload.py",
    "src/livifuser_nav/backbone_handoff.py",
    "src/livifuser_nav/evaluation.py",
    "src/livifuser_nav/contracts.py",
    "ros2_ws/src/livifuser_sim/livifuser_sim/live_policy_runner_node.py",
    "ros2_ws/src/livifuser_sim/livifuser_sim/simulation_supervisor_node.py",
    "ros2_ws/src/livifuser_sim/livifuser_sim/constant_arm_runner_node.py",
    "scripts/run_live_sim_development_episode.sh",
    "scripts/wait_sim_terminal.py",
    "scripts/seal_runtime_attempt.py",
)

REQUIRED_CONFIG = (
    "config/simulation_closed_loop_execution_v1.proposed.json",
    "config/runpod_rtx3090_runtime_v1.proposed.json",
    "config/simulation_live_sensor_contract_v1.json",
)

REQUIRED_EVIDENCE = (
    "evidence/cuda_route_benchmark_v1.json",
    "evidence/recorded_input_determinism_v1.json",
)

# Names that must never appear anywhere in the bundle. A cached feature bundle
# would let a rollout skip the backbone; a held-out or confirmatory artifact
# would contaminate the one-time evaluation.
FORBIDDEN_SUBSTRINGS = (
    "heldout",
    "held_out",
    "confirmatory_v1",
    "confirmatory_v2",
    "confirmatory_v3",
    "dinov3_splus_cache",
    "cache_v2",
    "cmd_vel",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_self_hash(value: dict[str, Any], field: str) -> str:
    copied = copy.deepcopy(value)
    copied.pop(field, None)
    return sha256_bytes(json.dumps(copied, sort_keys=True, separators=(",", ":")).encode())


def _assert_no_forbidden_names(names: list[str]) -> None:
    offenders = [
        name
        for name in names
        for needle in FORBIDDEN_SUBSTRINGS
        if needle in name.lower()
    ]
    _require(not offenders, f"forbidden members in runtime handoff: {sorted(set(offenders))}")


def _collect_members(root: Path, evidence_root: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    for relative in REQUIRED_CODE + REQUIRED_CONFIG:
        path = root / relative
        _require(path.is_file(), f"runtime handoff input missing: {relative}")
        members[relative] = path.read_bytes()
    for relative in REQUIRED_EVIDENCE:
        path = evidence_root / Path(relative).name
        _require(path.is_file(), f"required evidence missing: {relative} (looked in {path})")
        members[relative] = path.read_bytes()
    return members


def build_runtime_handoff(
    repository_root: str | Path,
    evidence_root: str | Path,
    output_path: str | Path,
    *,
    backbone_bundle: str | Path,
    policy_payload: str | Path,
) -> dict[str, Any]:
    """Assemble the runtime ZIP and bind every nested byte in its manifest."""

    root = Path(repository_root).resolve()
    evidence = Path(evidence_root).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite runtime handoff: {output}")

    backbone = Path(backbone_bundle).resolve()
    policies = Path(policy_payload).resolve()
    for label, path in (("backbone bundle", backbone), ("policy payload", policies)):
        _require(path.is_file(), f"{label} is not a file: {path}")

    members = _collect_members(root, evidence)
    members["artifacts/" + backbone.name] = backbone.read_bytes()
    members["artifacts/" + policies.name] = policies.read_bytes()
    _assert_no_forbidden_names(list(members))

    records = [
        {"name": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(members.items())
    ]
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "SEALED_RUNTIME_HANDOFF_DEVELOPMENT_ONLY",
        "amendment": (
            "docs/experiments/PREREGISTRATION_SIM_CLOSED_LOOP_EXECUTION_AMENDMENT_2026-08-24.md"
        ),
        "authorization": {
            "confirmatory_launch_authorized": False,
            "physical_robot_allowed": False,
            "cmd_vel_allowed": False,
        },
        "policy_identities": {
            "variants": list(VARIANTS),
            "seeds": list(SEEDS),
            "count": EXPECTED_POLICY_IDENTITIES,
        },
        "inputs": {
            "backbone_bundle_sha256": sha256_file(backbone),
            "policy_payload_sha256": sha256_file(policies),
        },
        "member_count": len(records),
        "members": records,
        "forbidden_inputs_present": False,
    }
    manifest["manifest_sha256_excludes_self"] = _canonical_self_hash(
        manifest, "manifest_sha256_excludes_self"
    )
    manifest_bytes = json_bytes(manifest)

    completion = {
        "schema_version": "1.0.0",
        "manifest_file_sha256": sha256_bytes(manifest_bytes),
        "manifest_sha256_excludes_self": manifest["manifest_sha256_excludes_self"],
        "member_count": len(records),
        "readiness": "SEALED_PENDING_INDEPENDENT_AUDIT",
    }
    completion["completion_sha256_excludes_self"] = _canonical_self_hash(
        completion, "completion_sha256_excludes_self"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(_zip_info(f"{BUNDLE_ROOT}/{name}"), payload)
        archive.writestr(_zip_info(MANIFEST_NAME), manifest_bytes)
        archive.writestr(_zip_info(COMPLETION_NAME), json_bytes(completion))

    return {
        "schema_version": "1.0.0",
        "status": "sealed",
        "bundle_path": str(output),
        "bundle_sha256": sha256_file(output),
        "bundle_size_bytes": output.stat().st_size,
        "member_count": len(records),
        "manifest_file_sha256": completion["manifest_file_sha256"],
        "manifest_sha256_excludes_self": manifest["manifest_sha256_excludes_self"],
        "confirmatory_launch_authorized": False,
    }


def audit_runtime_handoff(bundle_path: str | Path) -> dict[str, Any]:
    """Re-derive every identity from the bundle alone, read-only.

    The expectation is reconstructed from this module's requirements rather than
    read from the manifest, so a builder that shipped the wrong member list is
    detected instead of confirmed.
    """

    bundle = Path(bundle_path).resolve()
    _require(bundle.is_file(), f"runtime handoff is not a file: {bundle}")

    findings: list[str] = []
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        _require(MANIFEST_NAME in names, "runtime manifest is absent")
        _require(COMPLETION_NAME in names, "completion marker is absent")

        manifest = json.loads(archive.read(MANIFEST_NAME))
        completion = json.loads(archive.read(COMPLETION_NAME))

        # Self-consistency of the manifest and the marker that binds it.
        _require(
            manifest["manifest_sha256_excludes_self"]
            == _canonical_self_hash(manifest, "manifest_sha256_excludes_self"),
            "runtime manifest self-hash failed",
        )
        _require(
            completion["manifest_file_sha256"] == sha256_bytes(archive.read(MANIFEST_NAME)),
            "completion marker does not bind the manifest file",
        )
        _require(
            completion["completion_sha256_excludes_self"]
            == _canonical_self_hash(completion, "completion_sha256_excludes_self"),
            "completion marker self-hash failed",
        )

        payload_names = sorted(
            name[len(BUNDLE_ROOT) + 1 :]
            for name in names
            if name not in (MANIFEST_NAME, COMPLETION_NAME)
        )
        _assert_no_forbidden_names(payload_names)

        # Every recorded member must be present and hash correctly.
        recorded = {entry["name"]: entry for entry in manifest["members"]}
        for name, entry in sorted(recorded.items()):
            member = f"{BUNDLE_ROOT}/{name}"
            if member not in names:
                findings.append(f"manifest lists a missing member: {name}")
                continue
            payload = archive.read(member)
            if sha256_bytes(payload) != entry["sha256"]:
                findings.append(f"member hash mismatch: {name}")
            if len(payload) != entry["size_bytes"]:
                findings.append(f"member size mismatch: {name}")

        extra = sorted(set(payload_names) - set(recorded))
        findings.extend(f"member present but not in the manifest: {name}" for name in extra)

        # Independently required content, not taken from the manifest.
        for relative in REQUIRED_CODE + REQUIRED_CONFIG + REQUIRED_EVIDENCE:
            if relative not in recorded:
                findings.append(f"required member absent: {relative}")

        _require(
            manifest["policy_identities"]["count"] == EXPECTED_POLICY_IDENTITIES,
            "policy identity count is not 12",
        )
        _require(
            manifest["authorization"]["confirmatory_launch_authorized"] is False
            and manifest["authorization"]["physical_robot_allowed"] is False
            and manifest["authorization"]["cmd_vel_allowed"] is False,
            "runtime handoff claims an authorization it must not carry",
        )

        # The evidence inside must itself say it passed and used no forbidden input.
        route = json.loads(archive.read(f"{BUNDLE_ROOT}/evidence/cuda_route_benchmark_v1.json"))
        if route.get("status") != "CUDA_ROUTE_ACCEPTED":
            findings.append(f"route evidence status is {route.get('status')!r}")
        if not (route.get("parity", {}).get("pass") and route.get("timing", {}).get("pass")):
            findings.append("route evidence does not record both parity and timing passes")

        determinism = json.loads(
            archive.read(f"{BUNDLE_ROOT}/evidence/recorded_input_determinism_v1.json")
        )
        if determinism.get("deterministic") is not True:
            findings.append("recorded-input evidence does not record determinism")
        if determinism.get("policy_identities") != EXPECTED_POLICY_IDENTITIES:
            findings.append("recorded-input evidence does not cover all 12 identities")
        if determinism.get("device", "").startswith("cpu"):
            findings.append(
                "recorded-input determinism was established on CPU; the confirmatory "
                "route is CUDA and the gate requires the execution route"
            )
        for evidence in (route, determinism):
            forbidden = evidence.get("forbidden_inputs_used", {})
            if forbidden.get("heldout") or forbidden.get("confirmatory"):
                findings.append("evidence admits using a forbidden input")

    return {
        "schema_version": "1.0.0",
        "status": "audit_pass" if not findings else "audit_fail",
        "bundle_path": str(bundle),
        "bundle_sha256": sha256_file(bundle),
        "member_count": len(manifest["members"]),
        "manifest_sha256_excludes_self": manifest["manifest_sha256_excludes_self"],
        "findings": findings,
        "confirmatory_launch_authorized": False,
    }
