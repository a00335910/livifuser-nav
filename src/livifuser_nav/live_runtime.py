"""Exact float32 live inference core for the frozen closed-loop policies."""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from livifuser_nav.backbone_handoff import (
    BUNDLE_ROOT as BACKBONE_ROOT,
)
from livifuser_nav.backbone_handoff import (
    EXPECTED_MODEL_FILES,
    sha256_file,
    verify_bundle,
    verify_snapshot,
)
from livifuser_nav.evaluation import mahalanobis_distances
from livifuser_nav.learning_data import preprocess_rgb, tokenize_lidar
from livifuser_nav.policy_payload import (
    BUNDLE_ROOT as POLICY_ROOT,
)
from livifuser_nav.policy_payload import (
    CONFIG_SHA256,
    SEEDS,
    verify_policy_payload,
)

LIVE_VARIANTS = ("full", "lidar_only", "concat", "rgb_only")
CONTEXT_K = 8
HORIZON_H = 8
VALIDATION_REFERENCE_COUNT = 9459


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_deterministic_torch(torch: Any, device: str) -> None:
    """Apply the prospective deterministic float32 execution contract."""

    if device.startswith("cuda"):
        # cuBLAS GEMMs are not deterministic under CUDA >= 10.2 without a
        # dedicated workspace, and torch.use_deterministic_algorithms(True)
        # refuses to run them rather than returning irreproducible numbers.
        # cuBLAS reads this when it creates its handle, lazily at the first
        # matmul, so setting it here -- before any model is constructed or run
        # -- takes effect. An inherited value is left alone but must be one of
        # the two documented deterministic settings.
        existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if existing is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
        else:
            _require(
                existing in (":4096:8", ":16:8"),
                f"CUBLAS_WORKSPACE_CONFIG={existing!r} is not a deterministic setting",
            )
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if device.startswith("cuda"):
        _require(torch.cuda.is_available(), "CUDA route requested but CUDA is unavailable")


def extract_verified_backbone(bundle_path: str | Path, destination: str | Path) -> Path:
    """Verify then extract the exact snapshot without accepting extra files."""

    bundle = Path(bundle_path).resolve()
    destination = Path(destination).resolve()
    verify_bundle(bundle)
    snapshot = destination / "snapshot"
    if snapshot.exists():
        verify_snapshot(snapshot)
        return snapshot
    staging = destination / "snapshot.partial"
    if staging.exists():
        raise FileExistsError(f"partial backbone extraction exists: {staging}")
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(bundle) as archive:
            for name in sorted(EXPECTED_MODEL_FILES):
                target = staging / name
                target.write_bytes(archive.read(f"{BACKBONE_ROOT}/{name}"))
                expected = EXPECTED_MODEL_FILES[name]
                _require(target.stat().st_size == expected["size_bytes"], f"size drift: {name}")
                _require(sha256_file(target) == expected["sha256"], f"hash drift: {name}")
        verify_snapshot(staging)
        staging.rename(snapshot)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return snapshot


@dataclass(frozen=True, slots=True)
class PolicyMaterial:
    variant: str
    seed: int
    checkpoint: bytes
    aleatoric_cdf: np.ndarray
    mahalanobis_cdf: np.ndarray
    mahalanobis_mean: np.ndarray
    mahalanobis_precision: np.ndarray
    thresholds: dict[str, float]


def load_policy_material(
    bundle_path: str | Path, variant: str, seed: int
) -> PolicyMaterial:
    """Verify the complete immutable payload before selecting one identity."""

    _require(variant in LIVE_VARIANTS, f"variant is outside frozen live scope: {variant}")
    _require(seed in SEEDS, f"seed is outside frozen live scope: {seed}")
    return load_policy_materials(bundle_path, ((variant, seed),))[(variant, seed)]


def load_policy_materials(
    bundle_path: str | Path,
    identities: tuple[tuple[str, int], ...] | None = None,
) -> dict[tuple[str, int], PolicyMaterial]:
    """Verify once, then read one or more exact policy identities."""

    requested = identities or tuple(
        (variant, seed) for variant in LIVE_VARIANTS for seed in SEEDS
    )
    _require(len(requested) == len(set(requested)), "duplicate requested policy identity")
    for variant, seed in requested:
        _require(variant in LIVE_VARIANTS, f"variant is outside frozen live scope: {variant}")
        _require(seed in SEEDS, f"seed is outside frozen live scope: {seed}")
    bundle = Path(bundle_path).resolve()
    verify_policy_payload(bundle)
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(
            archive.read(f"{POLICY_ROOT}/POLICY_PAYLOAD_MANIFEST.json")
        )
        mean_payload = archive.read(f"{POLICY_ROOT}/mahalanobis/mean.npy")
        precision_payload = archive.read(f"{POLICY_ROOT}/mahalanobis/precision.npy")
        common_mean = np.load(io.BytesIO(mean_payload), allow_pickle=False)
        common_precision = np.load(io.BytesIO(precision_payload), allow_pickle=False)
        output: dict[tuple[str, int], PolicyMaterial] = {}
        for variant, seed in requested:
            rows = [
                row
                for row in manifest["records"]
                if row["variant"] == variant and int(row["seed"]) == seed
            ]
            _require(len(rows) == 1, "policy identity is not unique")
            row = rows[0]
            checkpoint = archive.read(f"{POLICY_ROOT}/{row['checkpoint']['name']}")
            score_payload = archive.read(f"{POLICY_ROOT}/{row['score']['name']}")
            with np.load(io.BytesIO(score_payload), allow_pickle=False) as score:
                aleatoric_cdf = np.asarray(
                    score["aleatoric_cdf_sorted"], dtype=np.float64
                ).copy()
                mahalanobis_cdf = np.asarray(
                    score["mahalanobis_cdf_sorted"], dtype=np.float64
                ).copy()
            thresholds = {
                name: float(value) for name, value in row["thresholds"].items()
            }
            _require(
                thresholds["combined"] == 1.0,
                "combined uncertainty threshold drifted",
            )
            output[(variant, seed)] = PolicyMaterial(
                variant=variant,
                seed=seed,
                checkpoint=checkpoint,
                aleatoric_cdf=aleatoric_cdf,
                mahalanobis_cdf=mahalanobis_cdf,
                mahalanobis_mean=common_mean.copy(),
                mahalanobis_precision=common_precision.copy(),
                thresholds=thresholds,
            )
    return output


def right_continuous_cdf(reference: np.ndarray, value: float) -> float:
    values = np.asarray(reference, dtype=np.float64)
    _require(values.shape == (VALIDATION_REFERENCE_COUNT,), "validation CDF size drifted")
    _require(math.isfinite(value), "uncertainty score is non-finite")
    return float(np.searchsorted(values, value, side="right") / values.size)


@dataclass(frozen=True, slots=True)
class LiveObservation:
    rgb: np.ndarray
    scan_ranges: np.ndarray
    scan_beam_count: int
    scan_angle_increment_rad: float
    goal: np.ndarray
    robot_state: np.ndarray


@dataclass(frozen=True, slots=True)
class RuntimeDecision:
    ready: bool
    status: str
    mean_h8: np.ndarray
    log_variance_h8: np.ndarray
    proposed_action: np.ndarray
    aleatoric: float
    mahalanobis: float
    z_aleatoric: float
    z_mahalanobis: float
    combined: float
    aleatoric_flag: bool
    mahalanobis_flag: bool
    combined_intervention: bool
    stage_ms: dict[str, float]


class ExactPolicyRuntime:
    """Stateful K=8 inference window with no padding across invalid input."""

    def __init__(
        self,
        *,
        backbone: Any,
        policy: Any,
        material: PolicyMaterial,
        sensor_contract: dict[str, Any],
        torch: Any,
        device: str,
    ) -> None:
        _require(material.variant in LIVE_VARIANTS, "invalid live variant")
        self.backbone = backbone
        self.policy = policy
        self.material = material
        self.sensor_contract = sensor_contract
        self.torch = torch
        self.device = device
        self.history: deque[tuple[np.ndarray, ...]] = deque(maxlen=CONTEXT_K)

    def clear_history(self) -> None:
        self.history.clear()

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def accept(self, observation: LiveObservation) -> RuntimeDecision:
        _validate_observation(observation)
        start_ns = time.perf_counter_ns()
        rgb = preprocess_rgb(observation.rgb)
        preprocess_end_ns = time.perf_counter_ns()

        pixels = self.torch.from_numpy(rgb).unsqueeze(0).to(self.device)
        self._sync()
        backbone_start_ns = time.perf_counter_ns()
        with self.torch.inference_mode():
            outputs = self.backbone(pixel_values=pixels, return_dict=False)
        self._sync()
        backbone_end_ns = time.perf_counter_ns()
        hidden, pooled = outputs[0], outputs[1]
        _require(tuple(hidden.shape) == (1, 201, 384), "S+/16 hidden shape drifted")
        _require(tuple(pooled.shape) == (1, 384), "S+/16 pooler shape drifted")
        spatial = hidden[:, 5:, :].reshape(1, 14, 14, 384)
        patches = (
            spatial.reshape(1, 7, 2, 7, 2, 384)
            .mean(dim=(2, 4))
            .reshape(49, 384)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        pooled_array = (
            pooled[0].detach().cpu().numpy().astype(np.float32, copy=False)
        )
        _require(np.all(np.isfinite(patches)), "non-finite S+/16 patch tokens")
        _require(np.all(np.isfinite(pooled_array)), "non-finite S+/16 pooled feature")

        token_start_ns = time.perf_counter_ns()
        tokenizer = self.sensor_contract["tokenizer"]
        tokens = tokenize_lidar(
            observation.scan_ranges,
            observation.scan_beam_count,
            observation.scan_angle_increment_rad,
            self.sensor_contract,
            sectors=int(tokenizer["sectors"]),
            range_clip_m=float(tokenizer["range_clip_m"]),
            visual_radius=int(tokenizer["visual_radius"]),
        )
        token_end_ns = time.perf_counter_ns()
        self.history.append(
            (
                np.ascontiguousarray(patches),
                np.ascontiguousarray(tokens.features),
                np.ascontiguousarray(tokens.visual_mask),
                np.ascontiguousarray(tokens.in_fov),
                np.ascontiguousarray(observation.goal, dtype=np.float32),
                np.ascontiguousarray(observation.robot_state, dtype=np.float32),
            )
        )
        base_stage_ms = {
            "rgb_preprocess": (preprocess_end_ns - start_ns) / 1e6,
            "splus_forward_and_pool": (backbone_end_ns - backbone_start_ns) / 1e6,
            "lidar_tokenize": (token_end_ns - token_start_ns) / 1e6,
        }
        if len(self.history) < CONTEXT_K:
            return _warmup_decision(len(self.history), base_stage_ms)

        policy_start_ns = time.perf_counter_ns()
        history = list(self.history)
        arrays = [np.stack([item[index] for item in history]) for index in range(6)]
        inputs = {
            "visual_tokens": self.torch.from_numpy(arrays[0]).unsqueeze(0).to(self.device),
            "lidar_features": self.torch.from_numpy(arrays[1]).unsqueeze(0).to(self.device),
            "visual_mask": self.torch.from_numpy(arrays[2]).unsqueeze(0).to(self.device),
            "in_fov": self.torch.from_numpy(arrays[3]).unsqueeze(0).to(self.device),
            "goal": self.torch.from_numpy(arrays[4]).unsqueeze(0).to(self.device),
            "robot_state": self.torch.from_numpy(arrays[5]).unsqueeze(0).to(self.device),
        }
        self._sync()
        with self.torch.inference_mode():
            policy_outputs = self.policy(**inputs)
        self._sync()
        mean = policy_outputs["mean"].detach().cpu().numpy().astype(np.float32, copy=False)
        log_variance = (
            policy_outputs["log_variance"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        policy_end_ns = time.perf_counter_ns()
        _require(mean.shape == (1, HORIZON_H, 2), "policy mean shape drifted")
        _require(log_variance.shape == (1, HORIZON_H, 2), "policy logvar shape drifted")
        _require(np.all(np.isfinite(mean)), "non-finite policy mean")
        _require(np.all(np.isfinite(log_variance)), "non-finite policy log-variance")

        uncertainty_start_ns = time.perf_counter_ns()
        mean_h8 = np.ascontiguousarray(mean[0])
        log_variance_h8 = np.ascontiguousarray(log_variance[0])
        aleatoric = float(np.mean(np.exp(np.clip(log_variance_h8, -5.0, 2.0))))
        mahalanobis = float(
            mahalanobis_distances(
                pooled_array[None],
                self.material.mahalanobis_mean,
                self.material.mahalanobis_precision,
            )[0]
        )
        z_a = right_continuous_cdf(self.material.aleatoric_cdf, aleatoric)
        z_m = right_continuous_cdf(self.material.mahalanobis_cdf, mahalanobis)
        combined = max(z_a, z_m)
        thresholds = self.material.thresholds
        uncertainty_end_ns = time.perf_counter_ns()
        stage_ms = {
            **base_stage_ms,
            "policy_stack_and_forward": (policy_end_ns - policy_start_ns) / 1e6,
            "uncertainty": (uncertainty_end_ns - uncertainty_start_ns) / 1e6,
            "complete_path": (uncertainty_end_ns - start_ns) / 1e6,
        }
        return RuntimeDecision(
            ready=True,
            status="inference",
            mean_h8=mean_h8,
            log_variance_h8=log_variance_h8,
            proposed_action=np.ascontiguousarray(mean_h8[0]),
            aleatoric=aleatoric,
            mahalanobis=mahalanobis,
            z_aleatoric=z_a,
            z_mahalanobis=z_m,
            combined=combined,
            aleatoric_flag=z_a > thresholds["aleatoric"],
            mahalanobis_flag=z_m > thresholds["mahalanobis"],
            combined_intervention=combined > thresholds["combined"],
            stage_ms=stage_ms,
        )


def construct_exact_runtime(
    *,
    backbone_snapshot: str | Path,
    policy_material: PolicyMaterial,
    sensor_contract_path: str | Path,
    device: str,
) -> ExactPolicyRuntime:
    """Construct models only after both immutable bundle verifiers have run."""

    import torch
    import transformers

    from livifuser_nav.model import LiViFuserPolicy

    configure_deterministic_torch(torch, device)
    snapshot = Path(backbone_snapshot).resolve()
    verify_snapshot(snapshot)
    contract = json.loads(Path(sensor_contract_path).read_text(encoding="utf-8"))
    _require(
        contract.get("status") == "FROZEN_FROM_ACCEPTED_TRAINING_EXPORT_CONTRACT",
        "live sensor contract status drifted",
    )
    checkpoint = torch.load(
        io.BytesIO(policy_material.checkpoint), map_location="cpu", weights_only=True
    )
    _require(checkpoint["variant"] == policy_material.variant, "checkpoint variant drifted")
    _require(int(checkpoint["seed"]) == policy_material.seed, "checkpoint seed drifted")
    _require(checkpoint["config_sha256"] == CONFIG_SHA256, "checkpoint config drifted")
    backbone = transformers.AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).eval().to(device=device, dtype=torch.float32)
    policy = LiViFuserPolicy(variant=policy_material.variant)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval().to(device=device, dtype=torch.float32)
    _require(next(backbone.parameters()).dtype == torch.float32, "backbone is not float32")
    _require(next(policy.parameters()).dtype == torch.float32, "policy is not float32")
    return ExactPolicyRuntime(
        backbone=backbone,
        policy=policy,
        material=policy_material,
        sensor_contract=contract,
        torch=torch,
        device=device,
    )


def _validate_observation(observation: LiveObservation) -> None:
    _require(
        isinstance(observation.rgb, np.ndarray)
        and observation.rgb.shape == (240, 320, 3)
        and observation.rgb.dtype == np.uint8,
        "invalid live RGB payload",
    )
    ranges = np.asarray(observation.scan_ranges)
    _require(ranges.ndim == 1, "scan ranges must be one-dimensional")
    _require(
        80 <= int(observation.scan_beam_count) <= ranges.size,
        "invalid live scan beam count",
    )
    _require(
        math.isfinite(float(observation.scan_angle_increment_rad))
        and float(observation.scan_angle_increment_rad) > 0.0,
        "invalid live scan angle increment",
    )
    _require(
        np.asarray(observation.goal).shape == (3,)
        and np.all(np.isfinite(observation.goal)),
        "invalid live goal",
    )
    _require(
        np.asarray(observation.robot_state).shape == (2,)
        and np.all(np.isfinite(observation.robot_state)),
        "invalid live robot state",
    )


def _warmup_decision(count: int, stage_ms: dict[str, float]) -> RuntimeDecision:
    zeros = np.zeros((HORIZON_H, 2), dtype=np.float32)
    return RuntimeDecision(
        ready=False,
        status=f"warmup_{count}_of_{CONTEXT_K}",
        mean_h8=zeros.copy(),
        log_variance_h8=zeros.copy(),
        proposed_action=np.zeros(2, dtype=np.float32),
        aleatoric=0.0,
        mahalanobis=0.0,
        z_aleatoric=0.0,
        z_mahalanobis=0.0,
        combined=0.0,
        aleatoric_flag=False,
        mahalanobis_flag=False,
        combined_intervention=False,
        stage_ms={**stage_ms, "policy_stack_and_forward": 0.0, "uncertainty": 0.0},
    )


# ---------------------------------------------------------------------------
# Open-loop constant-training-mean reference arm
#
# Closed-loop execution amendment section 1.1. This is a reference arm, never a
# learned variant. It reads no sensor, loads no backbone and no checkpoint, and
# produces no uncertainty score. Its command is the preregistered trivial
# baseline from the held-out evaluation amendment section 3, computed before
# held-out access from all 56,128 verified training rows across 120 episodes.
#
# The values are written as float64 hexadecimal literals so that a decimal
# transcription error cannot silently change the arm. The decimal forms are
# 0.047447296062892504 m/s and -0.005205255475722954 rad/s.
# ---------------------------------------------------------------------------

CONSTANT_ARM_NAME = "constant_training_mean"
CONSTANT_ARM_LINEAR_X_MPS = float.fromhex("0x1.84b0311bf5c89p-5")
CONSTANT_ARM_ANGULAR_Z_RADPS = float.fromhex("-0x1.5521b2091a221p-8")
CONSTANT_ARM_UNCERTAINTY = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class ConstantArmDecision:
    """Decision record for the sensor-blind constant arm.

    Deliberately not a ``RuntimeDecision``. That type carries aleatoric,
    Mahalanobis, and combined scores, and populating them with zeros here would
    record an inert gate as a gate that did not fire. The amendment forbids
    that, so the uncertainty fields are a single explicit sentinel instead.
    """

    ready: bool
    status: str
    proposed_action: np.ndarray
    uncertainty: str
    combined_intervention: bool


class ConstantActionRuntime:
    """Stateless-except-for-warmup open-loop constant command source.

    ``accept`` takes no observation. Sensor blindness is a structural property
    of the signature, not a convention a caller has to honour.
    """

    def __init__(self) -> None:
        self._accepted = 0

    def clear_history(self) -> None:
        """Re-arm the warmup exactly as a reset does for the learned runtimes."""

        self._accepted = 0

    @property
    def accepted_contexts(self) -> int:
        return self._accepted

    def accept(self) -> ConstantArmDecision:
        self._accepted += 1
        if self._accepted < CONTEXT_K:
            return ConstantArmDecision(
                ready=False,
                status=f"warmup_{self._accepted}_of_{CONTEXT_K}",
                proposed_action=np.zeros(2, dtype=np.float64),
                uncertainty=CONSTANT_ARM_UNCERTAINTY,
                combined_intervention=False,
            )
        return ConstantArmDecision(
            ready=True,
            status="constant_training_mean",
            proposed_action=np.asarray(
                [CONSTANT_ARM_LINEAR_X_MPS, CONSTANT_ARM_ANGULAR_Z_RADPS], dtype=np.float64
            ),
            uncertainty=CONSTANT_ARM_UNCERTAINTY,
            combined_intervention=False,
        )
