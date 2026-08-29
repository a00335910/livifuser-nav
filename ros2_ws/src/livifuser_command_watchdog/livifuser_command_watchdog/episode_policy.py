"""Pure readiness, lifecycle, and command gate for protocol-clean episodes."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from .keyboard_policy import ZERO_COMMAND, KeyboardCommand

IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,79}$")


@dataclass(frozen=True, slots=True)
class EpisodeIdentity:
    episode_id: str
    environment_id: str
    split: str
    route_id: str
    layout_id: str
    code_revision: str

    def __post_init__(self) -> None:
        identifiers = {
            "episode_id": self.episode_id,
            "environment_id": self.environment_id,
            "route_id": self.route_id,
            "layout_id": self.layout_id,
        }
        for name, value in identifiers.items():
            if not IDENTIFIER_PATTERN.fullmatch(value):
                raise ValueError(f"{name} must match [a-z0-9][a-z0-9_-]{{2,79}}")
        if self.split not in {"train", "validation", "test", "development"}:
            raise ValueError("split must be train, validation, test, or development")
        if not re.fullmatch(r"[0-9a-f]{7,40}", self.code_revision):
            raise ValueError("code_revision must be a 7-40 character lowercase Git hash")

    def validate_output_basename(self, basename: str) -> None:
        if basename != self.episode_id:
            raise ValueError("output_path basename must equal episode_id")


class EpisodePhase(str, Enum):
    PREFLIGHT = "preflight"
    RECORDER_STARTING = "recorder_starting"
    ZERO_WARMUP = "zero_warmup"
    RECORDING = "recording"
    ZERO_COOLDOWN = "zero_cooldown"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StreamRequirement:
    topic: str
    minimum_messages: int
    max_age_s: float | None

    def __post_init__(self) -> None:
        if not self.topic.startswith("/"):
            raise ValueError("stream topic must be absolute")
        if self.minimum_messages <= 0:
            raise ValueError("minimum_messages must be positive")
        if self.max_age_s is not None and (
            not math.isfinite(self.max_age_s) or self.max_age_s <= 0.0
        ):
            raise ValueError("max_age_s must be finite and positive when present")


@dataclass(frozen=True, slots=True)
class StreamObservation:
    message_count: int
    last_arrival_monotonic_s: float | None


@dataclass(frozen=True, slots=True)
class ReadinessDecision:
    ready: bool
    reasons: tuple[str, ...]


def evaluate_readiness(
    requirements: tuple[StreamRequirement, ...],
    observations: Mapping[str, StreamObservation],
    *,
    now_monotonic_s: float,
) -> ReadinessDecision:
    """Require enough messages and fresh arrivals for every declared stream."""

    if not math.isfinite(now_monotonic_s):
        raise ValueError("now_monotonic_s must be finite")
    reasons: list[str] = []
    for requirement in requirements:
        observation = observations.get(requirement.topic)
        if observation is None or observation.last_arrival_monotonic_s is None:
            reasons.append(f"stream_missing:{requirement.topic}")
            continue
        if observation.message_count < requirement.minimum_messages:
            reasons.append(
                f"stream_insufficient:{requirement.topic}:"
                f"{observation.message_count}/{requirement.minimum_messages}"
            )
        age_s = now_monotonic_s - observation.last_arrival_monotonic_s
        if not math.isfinite(age_s) or age_s < 0.0:
            reasons.append(f"stream_clock_invalid:{requirement.topic}")
        elif requirement.max_age_s is not None and age_s > requirement.max_age_s:
            reasons.append(f"stream_stale:{requirement.topic}:{age_s:.3f}s")
    return ReadinessDecision(not reasons, tuple(reasons))


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    duration_s: float = 45.0
    zero_warmup_s: float = 2.0
    zero_cooldown_s: float = 2.0
    operator_timeout_s: float = 0.25
    linear_mps: float = 0.08
    angular_radps: float = 0.40

    def __post_init__(self) -> None:
        positive = (
            self.duration_s,
            self.zero_warmup_s,
            self.zero_cooldown_s,
            self.operator_timeout_s,
            self.linear_mps,
            self.angular_radps,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("episode timing and command magnitudes must be finite and positive")
        if self.duration_s > 300.0:
            raise ValueError("duration_s exceeds the 300 s offline safety ceiling")
        if self.linear_mps > 0.10 or self.angular_radps > 0.50:
            raise ValueError("episode command magnitudes exceed watchdog limits")


@dataclass(frozen=True, slots=True)
class LifecycleSnapshot:
    phase: EpisodePhase
    reason: str
    motion_permitted: bool
    recording_elapsed_s: float
    recording_remaining_s: float


class EpisodeLifecycle:
    """Monotonic state machine whose recording deadline cannot be host-delayed."""

    def __init__(self, config: EpisodeConfig, *, now_monotonic_s: float) -> None:
        if not math.isfinite(now_monotonic_s):
            raise ValueError("initial monotonic time must be finite")
        self.config = config
        self.phase = EpisodePhase.PREFLIGHT
        self.reason = "waiting_for_streams"
        self.phase_started_s = now_monotonic_s
        self.recording_started_s: float | None = None
        self._last_now_s = now_monotonic_s

    def _check_now(self, now_monotonic_s: float) -> None:
        if not math.isfinite(now_monotonic_s) or now_monotonic_s < self._last_now_s:
            raise ValueError("episode monotonic clock regressed or became non-finite")
        self._last_now_s = now_monotonic_s

    def begin_recorder(self, *, now_monotonic_s: float) -> None:
        self._check_now(now_monotonic_s)
        if self.phase is not EpisodePhase.PREFLIGHT:
            raise RuntimeError("recorder can begin only after preflight")
        self.phase = EpisodePhase.RECORDER_STARTING
        self.reason = "waiting_for_recorder_subscriptions"
        self.phase_started_s = now_monotonic_s

    def recorder_ready(self, *, now_monotonic_s: float) -> None:
        self._check_now(now_monotonic_s)
        if self.phase is not EpisodePhase.RECORDER_STARTING:
            raise RuntimeError("recorder readiness is valid only while starting")
        self.phase = EpisodePhase.ZERO_WARMUP
        self.reason = "recording_zero_warmup"
        self.phase_started_s = now_monotonic_s

    def advance(self, *, now_monotonic_s: float) -> EpisodePhase:
        self._check_now(now_monotonic_s)
        changed = True
        while changed:
            changed = False
            if (
                self.phase is EpisodePhase.ZERO_WARMUP
                and now_monotonic_s - self.phase_started_s >= self.config.zero_warmup_s
            ):
                self.recording_started_s = self.phase_started_s + self.config.zero_warmup_s
                self.phase = EpisodePhase.RECORDING
                self.reason = "recording"
                self.phase_started_s = self.recording_started_s
                changed = True
            elif (
                self.phase is EpisodePhase.RECORDING
                and self.recording_started_s is not None
                and now_monotonic_s - self.recording_started_s >= self.config.duration_s
            ):
                self.phase = EpisodePhase.ZERO_COOLDOWN
                self.reason = "duration_reached"
                self.phase_started_s = self.recording_started_s + self.config.duration_s
                changed = True
            elif (
                self.phase is EpisodePhase.ZERO_COOLDOWN
                and now_monotonic_s - self.phase_started_s >= self.config.zero_cooldown_s
            ):
                self.phase = EpisodePhase.COMPLETE
                self.reason = self.reason or "complete"
                self.phase_started_s += self.config.zero_cooldown_s
                changed = True
        return self.phase

    def request_stop(self, reason: str, *, now_monotonic_s: float) -> None:
        self._check_now(now_monotonic_s)
        if self.phase is not EpisodePhase.RECORDING:
            return
        if not reason:
            raise ValueError("stop reason must not be empty")
        self.phase = EpisodePhase.ZERO_COOLDOWN
        self.reason = reason
        self.phase_started_s = now_monotonic_s

    def fail(self, reason: str, *, now_monotonic_s: float) -> None:
        self._check_now(now_monotonic_s)
        if not reason:
            raise ValueError("failure reason must not be empty")
        self.phase = EpisodePhase.FAILED
        self.reason = reason
        self.phase_started_s = now_monotonic_s

    def snapshot(self, *, now_monotonic_s: float) -> LifecycleSnapshot:
        self._check_now(now_monotonic_s)
        elapsed_s = 0.0
        if self.recording_started_s is not None:
            elapsed_s = max(0.0, now_monotonic_s - self.recording_started_s)
        elapsed_s = min(elapsed_s, self.config.duration_s)
        return LifecycleSnapshot(
            phase=self.phase,
            reason=self.reason,
            motion_permitted=self.phase is EpisodePhase.RECORDING,
            recording_elapsed_s=elapsed_s,
            recording_remaining_s=max(0.0, self.config.duration_s - elapsed_s),
        )


@dataclass(frozen=True, slots=True)
class OperatorIntent:
    command: KeyboardCommand
    arrival_monotonic_s: float
    structurally_valid: bool = True


@dataclass(frozen=True, slots=True)
class GateDecision:
    command: KeyboardCommand
    reason: str
    permitted: bool


def gate_operator_intent(
    lifecycle: EpisodeLifecycle,
    operator_intent: OperatorIntent | None,
    *,
    now_monotonic_s: float,
) -> GateDecision:
    """Forward only a fresh, exact, forward-only command during RECORDING."""

    snapshot = lifecycle.snapshot(now_monotonic_s=now_monotonic_s)
    if not snapshot.motion_permitted:
        return GateDecision(ZERO_COMMAND, "episode_not_recording", False)
    if operator_intent is None:
        return GateDecision(ZERO_COMMAND, "operator_missing", False)
    age_s = now_monotonic_s - operator_intent.arrival_monotonic_s
    if not math.isfinite(age_s) or age_s < 0.0:
        return GateDecision(ZERO_COMMAND, "operator_clock_invalid", False)
    if age_s > lifecycle.config.operator_timeout_s:
        return GateDecision(ZERO_COMMAND, "operator_stale", False)
    command = operator_intent.command
    if not operator_intent.structurally_valid or not all(
        math.isfinite(value) for value in (command.linear_mps, command.angular_radps)
    ):
        return GateDecision(ZERO_COMMAND, "operator_invalid", False)

    tolerance = 1e-6
    is_zero = (
        abs(command.linear_mps) <= tolerance and abs(command.angular_radps) <= tolerance
    )
    allowed_linear = abs(command.linear_mps - lifecycle.config.linear_mps) <= tolerance
    allowed_angular = any(
        abs(command.angular_radps - value) <= tolerance
        for value in (0.0, lifecycle.config.angular_radps, -lifecycle.config.angular_radps)
    )
    if not is_zero and not (allowed_linear and allowed_angular):
        return GateDecision(ZERO_COMMAND, "operator_command_not_whitelisted", False)
    return GateDecision(command, "operator_fresh", not is_zero)


class GoalReachTracker:
    """Require consecutive near-goal samples before ending an episode early."""

    def __init__(self, *, tolerance_m: float = 0.25, required_samples: int = 3) -> None:
        if not math.isfinite(tolerance_m) or tolerance_m <= 0.0:
            raise ValueError("goal tolerance must be finite and positive")
        if required_samples <= 0:
            raise ValueError("required_samples must be positive")
        self.tolerance_m = tolerance_m
        self.required_samples = required_samples
        self.consecutive_samples = 0

    def update(self, rho_m: float) -> bool:
        if math.isfinite(rho_m) and 0.0 <= rho_m <= self.tolerance_m:
            self.consecutive_samples += 1
        else:
            self.consecutive_samples = 0
        return self.consecutive_samples >= self.required_samples
