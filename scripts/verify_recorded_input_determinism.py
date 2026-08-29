#!/usr/bin/env python3
"""Gate 5: the live runtime must be bit-identical on replayed recorded inputs.

Closed-loop execution amendment section 12, gate 5: "the four-variant runner and
all 12 immutable checkpoint identities pass unit and deterministic
recorded-input tests".

The same recorded frames are pushed through a freshly constructed
``ExactPolicyRuntime`` twice for every policy identity, and every output must
match exactly -- not within a tolerance. Determinism is what makes a rollout
reproducible from its recorded inputs, so a near-miss is a failure.

Inputs are the 32 sealed parity frames from the RunPod handoff, which are drawn
only from training, validation, and excluded-development recordings. Held-out
and confirmatory trajectories are never read; the report records this.

Run on the device the rollouts will use. A pass on CPU does not establish
determinism on CUDA: cuBLAS needs a workspace configuration to be deterministic
at all, which ``configure_deterministic_torch`` applies for CUDA devices.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from livifuser_nav.backbone_handoff import json_bytes, sha256_bytes, sha256_file  # noqa: E402
from livifuser_nav.live_runtime import (  # noqa: E402
    CONTEXT_K,
    LiveObservation,
    construct_exact_runtime,
    extract_verified_backbone,
    load_policy_materials,
)
from livifuser_nav.policy_payload import SEEDS  # noqa: E402
from livifuser_nav.runpod_handoff import BUNDLE_ROOT, verify_runpod_handoff  # noqa: E402

VARIANTS = ("full", "lidar_only", "concat", "rgb_only")
IDENTITIES = tuple((variant, seed) for variant in VARIANTS for seed in SEEDS)
PARITY_PREFIX = f"{BUNDLE_ROOT}/artifacts/runtime/parity_inputs_v1"

# Every field of a decision that must reproduce exactly.
COMPARED_ARRAYS = ("mean_h8", "log_variance_h8", "proposed_action")
COMPARED_SCALARS = ("aleatoric", "mahalanobis", "z_aleatoric", "z_mahalanobis", "combined")
COMPARED_FLAGS = ("ready", "status", "aleatoric_flag", "mahalanobis_flag", "combined_intervention")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_parity_frames(bundle: Path) -> dict[str, Any]:
    """Read the sealed parity frames straight from the verified bundle."""

    with zipfile.ZipFile(bundle) as archive:

        def array(name: str) -> np.ndarray:
            return np.load(io.BytesIO(archive.read(f"{PARITY_PREFIX}/{name}")), allow_pickle=False)

        manifest = json.loads(archive.read(f"{PARITY_PREFIX}/manifest.json"))
        frames = {
            "manifest": manifest,
            "rgb": array("rgb.npy"),
            "scan_ranges": array("scan_ranges.npy"),
            "scan_beam_count": array("scan_beam_count.npy"),
            "scan_angle_increment_rad": array("scan_angle_increment_rad.npy"),
            "goal": array("goal.npy"),
            "robot_state": array("robot_state.npy"),
        }
    require(
        manifest["heldout_or_confirmatory_inputs_present"] is False,
        "parity inputs claim forbidden scope",
    )
    require(frames["rgb"].shape[0] == 32, "expected 32 recorded parity frames")
    return frames


def observation_at(frames: dict[str, Any], row: int) -> LiveObservation:
    return LiveObservation(
        rgb=np.asarray(frames["rgb"][row]),
        scan_ranges=np.asarray(frames["scan_ranges"][row]),
        scan_beam_count=int(frames["scan_beam_count"][row]),
        scan_angle_increment_rad=float(frames["scan_angle_increment_rad"][row]),
        goal=np.asarray(frames["goal"][row], dtype=np.float32),
        robot_state=np.asarray(frames["robot_state"][row], dtype=np.float32),
    )


def replay(runtime: Any, frames: dict[str, Any]) -> list[dict[str, Any]]:
    """Push every recorded frame through one runtime and capture each decision."""

    captured = []
    for row in range(frames["rgb"].shape[0]):
        decision = runtime.accept(observation_at(frames, row))
        captured.append(
            {
                **{name: np.asarray(getattr(decision, name)).copy() for name in COMPARED_ARRAYS},
                **{name: float(getattr(decision, name)) for name in COMPARED_SCALARS},
                **{name: getattr(decision, name) for name in COMPARED_FLAGS},
            }
        )
    return captured


def compare(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> list[str]:
    """Return a description of every difference; empty means bit-identical."""

    differences: list[str] = []
    for row, (a, b) in enumerate(zip(first, second, strict=True)):
        for name in COMPARED_ARRAYS:
            if not np.array_equal(a[name], b[name]):
                worst = float(np.max(np.abs(a[name] - b[name])))
                differences.append(f"frame {row}: {name} differs, max |delta| {worst:.3e}")
        for name in COMPARED_SCALARS + COMPARED_FLAGS:
            if a[name] != b[name]:
                differences.append(f"frame {row}: {name} {a[name]!r} != {b[name]!r}")
    return differences


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite determinism report: {args.output}")

    bundle_report = verify_runpod_handoff(args.bundle)
    frames = load_parity_frames(args.bundle)
    snapshot = extract_verified_backbone(args.backbone, args.extract_root)
    materials = load_policy_materials(args.policies, IDENTITIES)

    identity_reports: list[dict[str, Any]] = []
    for variant, seed in IDENTITIES:
        material = materials[(variant, seed)]
        passes = []
        for _ in range(2):
            # A fresh runtime each pass: reusing one would let state from the
            # first pass mask a nondeterministic construction path.
            runtime = construct_exact_runtime(
                backbone_snapshot=snapshot,
                policy_material=material,
                sensor_contract_path=args.sensor_contract,
                device=args.device,
            )
            passes.append(replay(runtime, frames))
            del runtime
        differences = compare(passes[0], passes[1])
        ready = sum(1 for row in passes[0] if row["ready"])
        require(
            ready >= frames["rgb"].shape[0] - (CONTEXT_K - 1),
            f"{variant}/{seed} produced too few inference-ready frames",
        )
        identity_reports.append(
            {
                "variant": variant,
                "seed": seed,
                "frames": len(passes[0]),
                "inference_ready_frames": ready,
                "deterministic": not differences,
                "differences": differences[:10],
            }
        )
        print(f"  {variant}/{seed}: {'deterministic' if not differences else 'DIVERGED'}")

    import torch
    import transformers

    report = {
        "schema_version": "1.0.0",
        "gate": "closed_loop_amendment_section_12_gate_5",
        "status": "complete",
        "device": args.device,
        "recorded_frames": int(frames["rgb"].shape[0]),
        "policy_identities": len(IDENTITIES),
        "passes_per_identity": 2,
        "identities": identity_reports,
        "deterministic": all(entry["deterministic"] for entry in identity_reports),
        "inputs": {
            "bundle_sha256": bundle_report["bundle_sha256"],
            "parity_manifest_self_sha256": frames["manifest"]["manifest_sha256_excludes_self"],
            "backbone_bundle_sha256": sha256_file(args.backbone),
            "policy_payload_sha256": sha256_file(args.policies),
            "code_sha256": sha256_file(Path(__file__)),
        },
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "forbidden_inputs_used": {"heldout": False, "confirmatory": False},
        "scope": "excluded_development_recorded_input_determinism_only",
    }
    report["report_sha256_excludes_self"] = sha256_bytes(json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle", type=Path, default=ROOT / "artifacts/livifuser_runpod_input_v1_bundle.zip"
    )
    parser.add_argument(
        "--backbone",
        type=Path,
        default=ROOT / "artifacts/livifuser_dinov3_vits16plus_backbone_c93d816_bundle.zip",
    )
    parser.add_argument(
        "--policies",
        type=Path,
        default=ROOT / "artifacts/livifuser_sim_closed_loop_policy_payload_v1_bundle.zip",
    )
    parser.add_argument(
        "--sensor-contract",
        type=Path,
        default=ROOT / "config/simulation_live_sensor_contract_v1.json",
    )
    parser.add_argument("--extract-root", type=Path, default=ROOT / "artifacts/runtime/backbone")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/runtime/recorded_input_determinism_v1.json"
    )
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json_bytes(report))
    print(
        json_bytes(
            {
                "output": str(args.output.resolve()),
                "device": report["device"],
                "deterministic": report["deterministic"],
                "identities": report["policy_identities"],
                "report_sha256_excludes_self": report["report_sha256_excludes_self"],
            }
        ).decode(),
        end="",
    )
    return 0 if report["deterministic"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
