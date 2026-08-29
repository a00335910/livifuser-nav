"""Pure deterministic command supervision for simulation-only evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProposalInput:
    stamp_ns: int
    linear_x: float
    angular_z: float
    valid: bool
    inference_ready: bool
    status: str
    combined_intervention: bool


@dataclass(frozen=True, slots=True)
class PrivilegedState:
    available: bool
    collision: bool
    goal_distance_m: float
    clearance_m: float


@dataclass(frozen=True, slots=True)
class SupervisorDecision:
    raw: tuple[float, float]
    clipped: tuple[float, float]
    executed: tuple[float, float]
    reason: str
    terminal_reason: str
    success_samples: int
    control_interval_ms: float


NOMINAL_CONTROL_PERIOD_S = 0.1
MAX_CONTROL_INTERVAL_S = 0.25


class SimulationSupervisor:
    """Frozen priority, clipping, slew, success, and termination state."""

    def __init__(
        self,
        *,
        scientific_deadline_sec: float = 120.0,
        success_distance_m: float = 0.25,
        success_samples: int = 3,
    ) -> None:
        if scientific_deadline_sec <= 0.0 or success_samples < 1:
            raise ValueError("invalid supervisor limits")
        self.deadline_ns = int(scientific_deadline_sec * 1e9)
        self.success_distance_m = float(success_distance_m)
        self.required_success_samples = int(success_samples)
        self.previous_command = (0.0, 0.0)
        self.last_stamp_ns: int | None = None
        self.start_stamp_ns: int | None = None
        self.consecutive_success_samples = 0
        self.terminal_reason = ""
        # Recovered stretched intervals are counted so an episode that
        # limps is visible in the evidence rather than silently smoothed.
        self.stretched_interval_count = 0

    def force_terminal(self, reason: str) -> None:
        if not reason:
            raise ValueError("terminal reason is empty")
        if not self.terminal_reason:
            self.terminal_reason = reason
        self.previous_command = (0.0, 0.0)

    def step(
        self,
        proposal: ProposalInput,
        privileged: PrivilegedState,
        *,
        emergency_stop: bool = False,
    ) -> SupervisorDecision:
        if proposal.stamp_ns < 0:
            raise ValueError("proposal stamp is negative")
        interval_s = NOMINAL_CONTROL_PERIOD_S
        if self.last_stamp_ns is not None:
            if proposal.stamp_ns <= self.last_stamp_ns:
                # Not a missing input: proposals are ordered by construction, so
                # a stamp that fails to advance means two publishers on one topic
                # or a corrupted clock. Neither is recoverable by clearing
                # history, so this stays terminal.
                self.force_terminal("operational_failure_proposal_stamp_regression")
            else:
                interval_s = (proposal.stamp_ns - self.last_stamp_ns) / 1e9
        self.last_stamp_ns = proposal.stamp_ns
        if self.start_stamp_ns is None:
            self.start_stamp_ns = proposal.stamp_ns

        # Amendment section 3: a missing or stale input clears the K-context
        # history and commands zero *for that tick*. Section 6 reserves
        # termination for non-finite commands, exceptions, integrity loss, clock
        # failure, and watchdog expiry. A stretched interval is a missing input,
        # so recover and continue rather than ending the episode.
        stretched_interval = not 0.0 < interval_s <= MAX_CONTROL_INTERVAL_S
        if stretched_interval:
            self.stretched_interval_count += 1
            # The measured gap must not reach the slew limiter: a long gap would
            # otherwise authorise a proportionally large velocity step.
            interval_s = NOMINAL_CONTROL_PERIOD_S

        raw = (float(proposal.linear_x), float(proposal.angular_z))
        finite_raw = all(math.isfinite(value) for value in raw)
        clipped = (
            min(max(raw[0], -0.10), 0.10) if finite_raw else 0.0,
            min(max(raw[1], -0.50), 0.50) if finite_raw else 0.0,
        )
        if not finite_raw:
            self.force_terminal("operational_failure_nonfinite_proposal")

        if privileged.available and privileged.goal_distance_m <= self.success_distance_m:
            self.consecutive_success_samples += 1
        else:
            self.consecutive_success_samples = 0
        scientific_timeout = (
            proposal.stamp_ns - self.start_stamp_ns >= self.deadline_ns
        )

        terminal = ""
        reason = "learned_command"
        stop = False
        if emergency_stop:
            reason = terminal = "emergency_operator_stop"
            stop = True
        elif self.terminal_reason:
            reason = self.terminal_reason
            stop = True
        elif stretched_interval:
            # Command zero for this tick and let the runner rebuild its context.
            reason = "control_interval_recovered"
            stop = True
        elif not proposal.valid or not proposal.inference_ready:
            reason = proposal.status or "input_or_warmup_stop"
            stop = True
        elif not privileged.available:
            reason = "ground_truth_missing"
            stop = True
        elif privileged.collision:
            reason = terminal = "collision"
            stop = True
        else:
            command = self._slew(clipped, interval_s)
            if proposal.combined_intervention:
                reason = terminal = "uncertainty_intervention"
                stop = True
            elif self.consecutive_success_samples >= self.required_success_samples:
                reason = terminal = "success"
                stop = True
            elif scientific_timeout:
                reason = terminal = "scientific_timeout"
                stop = True
        # Episode-level cap. Warmup, missing ground truth, and other non-terminal
        # stops must still end the rollout at the frozen simulated deadline;
        # otherwise a runner that never becomes inference-ready runs until the
        # wall monitor is killed (a7/a10).
        if not terminal and not self.terminal_reason and scientific_timeout:
            reason = terminal = "scientific_timeout"
            stop = True
        if stop:
            command = (0.0, 0.0)
        if terminal:
            self.force_terminal(terminal)
        self.previous_command = command
        return SupervisorDecision(
            raw=raw,
            clipped=clipped,
            executed=command,
            reason=reason,
            terminal_reason=self.terminal_reason,
            success_samples=self.consecutive_success_samples,
            control_interval_ms=interval_s * 1000.0,
        )

    def _slew(self, target: tuple[float, float], interval_s: float) -> tuple[float, float]:
        limits = (0.50 * interval_s, 1.0 * interval_s)
        return tuple(
            previous + min(max(value - previous, -limit), limit)
            for value, previous, limit in zip(
                target, self.previous_command, limits, strict=True
            )
        )
