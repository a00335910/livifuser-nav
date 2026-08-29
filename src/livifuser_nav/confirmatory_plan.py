"""The immutable execution plan for the closed-loop confirmatory batch.

Closed-loop execution amendment section 9. The order of all 1,080 scientific
identities is fixed before the first launch and no outcome may change what comes
next: learned variants in (full, lidar_only, concat, rgb_only) order; within a
variant, seeds ascending; within a seed, the 80 `test_ood` entries in ascending
frozen schedule ordinal. After those 960, the constant reference arm runs the
same 80 entries. After those 1,040, Nav2 runs the C0 and C4 subset in ascending
ordinal.

The rules that protect the result live here rather than in the runner script,
because they decide whether evidence is admissible:

* an identity with an accepted **scientific** terminal outcome is complete and
  is never rerun -- rerunning one would be optional stopping;
* an **operational** failure may retry the same immutable identity, because the
  amendment classes simulator crashes, runner crashes and watchdog expiry as
  infrastructure;
* repeated failure of one identity pauses the batch for a documented audit
  rather than being retried indefinitely;
* no scientific aggregate is derivable from this module. It reports progress as
  identities, attempts and operational state only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VARIANTS = ("full", "lidar_only", "concat", "rgb_only")
SEEDS = (20260805, 20260806, 20260807)
CONSTANT_ARM = "constant_training_mean"
CONSTANT_ARM_SEED = 0
NAV2_ARM = "nav2"
NAV2_CONDITIONS = ("C0", "C4")

LEARNED_ROLLOUTS = 960
CONSTANT_ROLLOUTS = 80
NAV2_ROLLOUTS = 40
TOTAL_ROLLOUTS = LEARNED_ROLLOUTS + CONSTANT_ROLLOUTS + NAV2_ROLLOUTS

ORDINAL_FIRST = 180
ORDINAL_LAST = 259

SCHEDULE_SHA256 = "85F48F81D52F82A5CD64D14D704E032617582D7F5462A7B6FD46C5E64899047D"

# An attempt that ends in one of these has produced science. The identity is
# finished and must never run again.
SCIENTIFIC_TERMINALS = frozenset(
    {"success", "collision", "uncertainty_intervention", "scientific_timeout"}
)

# Retrying more than this many times means something systematic is wrong, and
# the amendment requires a documented audit rather than more retries.
# Three attempts is the frozen default. It exists to stop the batch looping on
# an unknown fault, so it may be raised only when the fault is identified,
# operational, and known to be intermittent -- never to grind past something
# unexplained. LIVIFUSER_MAX_ATTEMPTS raises it, and every use must be
# documented in an amendment naming the cause. Retrying changes nothing about
# what an episode measures: same world, start, goal, seed, checkpoint and
# thresholds. Only an attempt that produces an accepted scientific outcome ends
# an identity, and that outcome remains permanent.
MAX_ATTEMPTS_PER_IDENTITY = int(os.environ.get("LIVIFUSER_MAX_ATTEMPTS", "3"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class Identity:
    """One immutable scientific identity: an arm, a seed, and one episode."""

    index: int
    arm: str
    seed: int
    ordinal: int
    episode_id: str
    condition: str
    # The LiDAR observation condition is its own field in the frozen schedule and
    # is NOT derivable from the episode condition: C1 and C4 are camera and
    # obstacle-visibility manipulations that leave the LiDAR nominal (C0), while
    # C3 binds to C3b. Deriving it instead of reading it passed "C1"/"C4" to the
    # analytic LiDAR node, which rejects them, so /scan never existed for half of
    # every arm and those episodes recorded a robot that never moved.
    lidar_condition: str
    world_json: str
    world_sdf: str
    observation_seed: int
    scientific_deadline_sec: float

    @property
    def key(self) -> str:
        return f"{self.arm}/{self.seed}/{self.ordinal}"


def load_schedule(path: str | Path) -> list[dict[str, Any]]:
    """Read the frozen schedule and return the 80 test_ood entries in order.

    Accepts either the full repository schedule or the 80-entry subset the
    handoff ships. Both are checked against the same frozen source identity, so
    neither can be substituted for something else.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "schedule_sha256_excludes_self" in payload:
        source = payload["schedule_sha256_excludes_self"]
    else:
        # The shipped subset records the identity of the schedule it came from.
        source = payload.get("source_schedule_sha256_excludes_self")
    _require(source == SCHEDULE_SHA256, "confirmatory schedule identity drifted")

    entries = [
        entry
        for entry in payload["episodes"]
        if entry.get("split") == "test_ood" and ORDINAL_FIRST <= entry["ordinal"] <= ORDINAL_LAST
    ]
    entries.sort(key=lambda entry: entry["ordinal"])
    _require(len(entries) == 80, f"expected 80 test_ood entries, found {len(entries)}")
    return entries


def locate_schedule(root: str | Path) -> Path:
    """Find the schedule in either the repository or the shipped layout."""

    base = Path(root)
    candidates = (
        base / "artifacts/simulation/confirmatory_v3/schedule.json",
        base / "artifacts/simulation/closed_loop_schedule_subset_v1.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"no confirmatory schedule found under {base}")


def locate_world(root: str | Path, relative: str) -> Path:
    """Resolve a schedule world path in either layout.

    The repository keeps `confirmatory_v3/worlds/<name>`; the bundle flattens to
    `confirmatory_worlds/<name>`.
    """

    base = Path(root)
    name = Path(relative).name
    candidates = (
        base / "artifacts/simulation/confirmatory_v3" / relative,
        base / "artifacts/simulation/confirmatory_worlds" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"world not found in either layout: {relative}")


def build_plan(schedule_path: str | Path) -> list[Identity]:
    """Expand the frozen order into the complete list of scientific identities."""

    entries = load_schedule(schedule_path)
    plan: list[Identity] = []

    def append(arm: str, seed: int, entry: dict[str, Any]) -> None:
        plan.append(
            Identity(
                index=len(plan),
                arm=arm,
                seed=seed,
                ordinal=int(entry["ordinal"]),
                episode_id=str(entry["episode_id"]),
                condition=str(entry["condition"]),
                lidar_condition=str(entry["lidar_condition"]),
                world_json=str(entry["world_json"]),
                world_sdf=str(entry["world_sdf"]),
                observation_seed=int(entry["observation_seed"]),
                scientific_deadline_sec=float(entry["scientific_simulated_deadline_sec"]),
            )
        )

    for variant in VARIANTS:
        for seed in SEEDS:
            for entry in entries:
                append(variant, seed, entry)
    for entry in entries:
        append(CONSTANT_ARM, CONSTANT_ARM_SEED, entry)
    for entry in entries:
        if entry["condition"] in NAV2_CONDITIONS:
            append(NAV2_ARM, 0, entry)

    _require(len(plan) == TOTAL_ROLLOUTS, f"plan is {len(plan)}, expected {TOTAL_ROLLOUTS}")
    return plan


def select_subset(
    plan: list[Identity],
    *,
    arms: tuple[str, ...] | None = None,
    conditions: tuple[str, ...] | None = None,
    max_per_cell: int | None = None,
) -> list[Identity]:
    """A deterministic subset of the frozen plan, in frozen order.

    The confirmatory budget was cut short by a fixed compute allowance, so only
    part of the plan can be executed. Which part must be decided by a rule
    written down in advance, never by which identities happen to look good: this
    takes the FIRST max_per_cell identities of each (arm, seed, condition) cell
    in the plan's own frozen order, so the selection is reproducible from the
    schedule alone and carries no outcome information.

    Order within the subset is the plan's order, so execution stays sequential
    and resumable exactly as before.
    """

    chosen: list[Identity] = []
    per_cell: dict[tuple[str, int, str], int] = {}
    for identity in plan:
        if arms is not None and identity.arm not in arms:
            continue
        if conditions is not None and identity.condition not in conditions:
            continue
        cell = (identity.arm, identity.seed, identity.condition)
        if max_per_cell is not None and per_cell.get(cell, 0) >= max_per_cell:
            continue
        per_cell[cell] = per_cell.get(cell, 0) + 1
        chosen.append(identity)
    return chosen


def classify_attempt(attempt_dir: str | Path) -> str:
    """Return "scientific", "operational", or "absent" for one attempt."""

    directory = Path(attempt_dir)
    terminal = directory / "terminal.json"
    if terminal.is_file():
        try:
            record = json.loads(terminal.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return "operational"
        reason = record.get("terminal_reason", "")
        # An episode in which the policy never produced a single control
        # decision measured nothing, whatever reason it terminated with. This
        # is not hypothetical: passing an unsupported LiDAR condition killed the
        # analytic LiDAR node, /scan never existed, association never completed,
        # and the robot sat still until the scientific deadline expired. Those
        # episodes were recorded as scientific_timeout -- an accepted outcome --
        # so the plan retired them and would never have retried them. Half of
        # every arm would have entered the analysis as a policy that "failed to
        # reach the goal" when the policy was never running at all.
        if int(record.get("context_sequence", 0) or 0) <= 0:
            return "operational"
        if record.get("terminal") and reason in SCIENTIFIC_TERMINALS:
            return "scientific"
        # A terminal record naming an operational failure is still operational.
        return "operational"
    if (directory / "operational_failure.json").is_file():
        return "operational"
    return "absent"


AUDIT_CLEARANCE_NAME = "audit_cleared.json"


def identity_state(root: str | Path, identity: Identity) -> dict[str, Any]:
    """Summarise every attempt already made for one identity.

    An exhausted identity may be released by placing an audit clearance beside
    its attempts. The amendment requires that repeated failure "pauses the batch
    for a documented audit" and that the reason "be resolved without changing
    scientific inputs" -- so the release is a deliberate, recorded act, the
    failed attempts stay exactly where they are, and the budget resets only for
    attempts made after the clearance.
    """

    directory = Path(root) / identity.arm / str(identity.seed) / f"{identity.ordinal:04d}"
    attempts = sorted(directory.glob("attempt_*")) if directory.is_dir() else []
    outcomes = [classify_attempt(attempt) for attempt in attempts]

    cleared_after = 0
    clearance = directory / AUDIT_CLEARANCE_NAME
    if clearance.is_file():
        try:
            record = json.loads(clearance.read_text(encoding="utf-8"))
            cleared_after = int(record.get("attempts_at_clearance", 0))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            cleared_after = 0

    # Only attempts made since the clearance count against the budget.
    chargeable = max(len(attempts) - cleared_after, 0)
    return {
        "directory": directory,
        "attempts": len(attempts),
        "outcomes": outcomes,
        "complete": "scientific" in outcomes,
        "audit_cleared_after": cleared_after,
        "exhausted": chargeable >= MAX_ATTEMPTS_PER_IDENTITY and "scientific" not in outcomes,
    }


def next_identity(root: str | Path, plan: list[Identity]) -> tuple[Identity | None, str]:
    """Return the next identity to launch, and why.

    Walks the frozen order and stops at the first identity that is neither
    complete nor exhausted. An exhausted identity halts the batch: the amendment
    requires a documented audit, and skipping past it would silently drop a
    scientific identity from the design.
    """

    for identity in plan:
        state = identity_state(root, identity)
        if state["complete"]:
            continue
        if state["exhausted"]:
            return None, (
                f"identity {identity.key} failed {state['attempts']} times without a "
                "scientific outcome; the batch is paused for a documented audit"
            )
        return identity, ("retry" if state["attempts"] else "first attempt")
    return None, "every identity has an accepted scientific outcome"


def progress(root: str | Path, plan: list[Identity]) -> dict[str, Any]:
    """Operational progress only: identities, attempts, and state.

    Deliberately reports no scientific aggregate. The amendment forbids showing
    one before the planned set is complete and sealed, and a runner that
    displayed success rates mid-batch would invite stopping on a favourable
    trend.
    """

    complete = 0
    attempts = 0
    exhausted: list[str] = []
    for identity in plan:
        state = identity_state(root, identity)
        attempts += state["attempts"]
        if state["complete"]:
            complete += 1
        elif state["exhausted"]:
            exhausted.append(identity.key)
    return {
        "identities_total": len(plan),
        "identities_complete": complete,
        "identities_remaining": len(plan) - complete,
        "attempts_made": attempts,
        "exhausted_identities": exhausted,
        "batch_complete": complete == len(plan),
    }
