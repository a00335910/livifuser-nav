"""Analysis of the sealed closed-loop confirmatory batch.

Closed-loop execution amendment section 8 and preregistration section 14. This
module is written **before** the batch completes and before any outcome has been
seen, so the estimator, the resampling scheme and the claim criteria cannot be
tuned toward a result. That ordering is the point: analysis code written after
looking at data invites choices a reviewer cannot distinguish from honest ones.

Structure follows the amendment exactly:

* the episode is the analysis unit, worlds are the generalization unit, and
  training seeds are nuisance factors -- so resampling draws worlds first, then
  paired episodes within them, and never treats episodes as independent;
* every rate is reported per variant and per condition with its exact
  denominator, and every seed separately;
* the constant arm carries no seed dimension, is never pooled into a
  learned-variant mean, and is excluded from intervention rates;
* Nav2 is reported separately under its competence limits;
* operational failures and timeouts are counted, never silently dropped.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from livifuser_nav.confirmatory_plan import (
    CONSTANT_ARM,
    NAV2_ARM,
    SCIENTIFIC_TERMINALS,
    VARIANTS,
)

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260824
PERCENTILES = (2.5, 97.5)
CONDITIONS = ("C0", "C1", "C3b", "C4")

# Conditions are recorded as C3 in the frozen schedule and serialized as C3b in
# evaluation output; the amendment fixes this mapping.
CONDITION_ALIASES = {"C3": "C3b"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class Episode:
    """One accepted scientific outcome, with everything needed to cluster it."""

    arm: str
    seed: int
    ordinal: int
    world: str
    condition: str
    terminal_reason: str
    success: bool
    collision: bool
    uncertainty_intervention: bool
    goal_distance_m: float
    clearance_m: float
    stretched_intervals: int


def _normalise_condition(value: str) -> str:
    return CONDITION_ALIASES.get(value, value)


def load_episodes(root: str | Path, plan: list[Any]) -> list[Episode]:
    """Read every accepted scientific outcome, keyed to its planned identity.

    An identity without an accepted outcome is omitted rather than imputed, and
    the caller is expected to check the denominators: a silently short table
    would understate the true denominator and inflate every rate.
    """

    by_key = {identity.key: identity for identity in plan}
    episodes: list[Episode] = []
    for terminal in sorted(Path(root).rglob("terminal.json")):
        record = json.loads(terminal.read_text(encoding="utf-8"))
        reason = record.get("terminal_reason", "")
        if not record.get("terminal") or reason not in SCIENTIFIC_TERMINALS:
            continue
        # An episode in which the policy never produced a control decision
        # measured nothing, whatever it terminated with. `classify_attempt`
        # already refuses these; this loader had its own filter that did not,
        # so 260 episodes where /scan never existed would have entered the
        # analysis as policies that "failed to reach the goal".
        if int(record.get("context_sequence", 0) or 0) <= 0:
            continue
        # attempt_00N / <ordinal> / <seed> / <arm>
        ordinal = int(terminal.parent.parent.name)
        seed = int(terminal.parent.parent.parent.name)
        arm = terminal.parent.parent.parent.parent.name
        identity = by_key.get(f"{arm}/{seed}/{ordinal}")
        _require(identity is not None, f"outcome outside the frozen plan: {terminal}")
        episodes.append(
            Episode(
                arm=arm,
                seed=seed,
                ordinal=ordinal,
                world=identity.episode_id.rsplit("_c", 1)[0],
                condition=_normalise_condition(identity.condition),
                terminal_reason=reason,
                success=bool(record.get("success")),
                collision=bool(record.get("collision")),
                uncertainty_intervention=bool(record.get("uncertainty_intervention")),
                goal_distance_m=float(record.get("ground_truth_goal_distance_m", float("nan"))),
                clearance_m=float(record.get("ground_truth_clearance_m", float("nan"))),
                stretched_intervals=int(record.get("stretched_interval_count", 0)),
            )
        )
    return episodes


def count_attempts(root: str | Path) -> dict[str, int]:
    """Operational counts that may never be dropped from the record."""

    base = Path(root)
    return {
        "attempts_total": sum(1 for _ in base.rglob("attempt_*") if _.is_dir()),
        "operational_failures": sum(1 for _ in base.rglob("operational_failure.json")),
        "teardown_leaks": sum(1 for _ in base.rglob("teardown_leak.json")),
        "audit_clearances": sum(1 for _ in base.rglob("audit_cleared.json")),
    }


def rate_table(episodes: list[Episode], metric: str) -> dict[tuple[str, str, int | None], Any]:
    """Rate per (arm, condition, seed) with its exact denominator.

    The constant arm and Nav2 carry no seed dimension, so their key holds None
    rather than a fabricated seed.
    """

    buckets: dict[tuple[str, str, int | None], list[bool]] = defaultdict(list)
    for episode in episodes:
        seed = None if episode.arm in (CONSTANT_ARM, NAV2_ARM) else episode.seed
        buckets[(episode.arm, episode.condition, seed)].append(bool(getattr(episode, metric)))
    return {
        key: {
            "numerator": int(sum(values)),
            "denominator": len(values),
            "rate": float(sum(values) / len(values)) if values else float("nan"),
        }
        for key, values in sorted(buckets.items(), key=lambda item: str(item[0]))
    }


def _world_clustered_replicate(
    rng: np.random.Generator,
    by_world: dict[str, list[float]],
) -> float:
    """One hierarchical replicate: resample worlds, then episodes within each.

    Worlds are the generalization unit, so the outer draw is over worlds. An
    interval built by resampling episodes directly would be far too narrow,
    because episodes inside one world are not independent.
    """

    worlds = list(by_world)
    drawn = rng.choice(len(worlds), size=len(worlds), replace=True)
    means: list[float] = []
    for index in drawn:
        values = by_world[worlds[index]]
        if not values:
            continue
        picks = rng.choice(len(values), size=len(values), replace=True)
        means.append(float(np.mean([values[p] for p in picks])))
    return float(np.mean(means)) if means else float("nan")


def cluster_bootstrap(
    paired: dict[str, list[float]],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Frozen hierarchical cluster bootstrap over worlds, then paired episodes."""

    _require(bool(paired), "bootstrap requires at least one world")
    rng = np.random.default_rng(seed)
    draws = np.array(
        [_world_clustered_replicate(rng, paired) for _ in range(replicates)], dtype=np.float64
    )
    finite = draws[np.isfinite(draws)]
    _require(finite.size > 0, "every bootstrap replicate was non-finite")
    low, high = np.percentile(finite, PERCENTILES, method="linear")
    point = float(np.mean([float(np.mean(v)) for v in paired.values() if v]))
    return {
        "point": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "replicates": int(finite.size),
        "excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def sign_test_two_worlds(paired: dict[str, list[float]]) -> dict[str, Any]:
    """Exact two-sided sign test on the paired world-mean signs, ties excluded.

    With two worlds the only informative outcome is agreement, and the exact
    two-sided p-value is 0.5. This is reported rather than dressed up: a test
    with two clusters cannot carry much, and pretending otherwise would be
    worse than stating its limit.
    """

    signs = [np.sign(np.mean(values)) for values in paired.values() if values]
    kept = [s for s in signs if s != 0]
    if not kept:
        return {"worlds": len(signs), "ties_excluded": len(signs), "agree": False, "p_value": 1.0}
    agree = all(s == kept[0] for s in kept)
    return {
        "worlds": len(signs),
        "ties_excluded": len(signs) - len(kept),
        "agree": bool(agree),
        # Exact two-sided binomial for n kept signs at p=0.5.
        "p_value": float(2.0 * 0.5 ** len(kept)) if agree else 1.0,
        "note": "two worlds cannot yield p below 0.5; reported for completeness",
    }


def paired_contrast(
    episodes: list[Episode],
    metric: str,
    arm_a: str,
    arm_b: str,
    condition: str,
) -> dict[str, list[float]]:
    """Per-world paired differences (arm_a minus arm_b) on one condition.

    Pairing is by world, ordinal and seed, which is the frozen
    world/episode/condition/observation-seed structure. An unpaired difference
    would discard exactly the structure the design was built to exploit.
    """

    def index(arm: str) -> dict[tuple[str, int, int | None], float]:
        return {
            (e.world, e.ordinal, None if e.arm in (CONSTANT_ARM, NAV2_ARM) else e.seed): float(
                getattr(e, metric)
            )
            for e in episodes
            if e.arm == arm and e.condition == condition
        }

    left, right = index(arm_a), index(arm_b)
    by_world: dict[str, list[float]] = defaultdict(list)
    for key, value in left.items():
        world, _, seed = key
        # The constant arm has no seed, so a learned identity pairs against its
        # seedless counterpart on the same world and ordinal.
        partner = right.get(key) if seed is None else right.get((key[0], key[1], seed))
        if partner is None and seed is not None:
            partner = right.get((key[0], key[1], None))
        if partner is not None:
            by_world[world].append(value - partner)
    return dict(by_world)


def summarise(root: str | Path, plan: list[Any]) -> dict[str, Any]:
    """Everything section 8 requires, with denominators and no silent drops."""

    episodes = load_episodes(root, plan)
    operational = count_attempts(root)
    learned = [e for e in episodes if e.arm in VARIANTS]
    return {
        "schema_version": "1.0.0",
        "episodes_analysed": len(episodes),
        "identities_planned": len(plan),
        "coverage_complete": len(episodes) == len(plan),
        "operational": operational,
        "rates": {
            metric: rate_table(episodes, metric)
            for metric in ("success", "collision", "uncertainty_intervention")
        },
        "terminal_reasons": {
            reason: sum(1 for e in episodes if e.terminal_reason == reason)
            for reason in sorted({e.terminal_reason for e in episodes})
        },
        "stretched_interval_episodes": sum(1 for e in episodes if e.stretched_intervals),
        "learned_episodes": len(learned),
        "note": (
            "rates carry exact denominators; the constant arm and Nav2 carry no seed "
            "dimension and are never pooled into a learned-variant mean"
        ),
    }
