#!/usr/bin/env python3
"""Execute the closed-loop confirmatory batch in the frozen order.

Closed-loop execution amendment section 9. This script does no science: it walks
the immutable plan from `livifuser_nav.confirmatory_plan`, launches one episode
at a time, and stops. Every decision about what may run next, what counts as a
completed identity, and when the batch must pause lives in that module, where it
is unit tested.

Three refusals are deliberate and are checked before anything launches:

* `--authorize` must be given explicitly. Section 12 makes approval a recorded
  act, and section 9 makes an accepted scientific outcome permanent, so a batch
  started by accident cannot be undone.
* the frozen schedule, execution config, and runtime bundle identities are
  verified first; a drifted input invalidates every rollout that follows it.
* no scientific aggregate is printed. Progress is identities, attempts, and
  operational state only, because a visible success rate mid-batch invites
  stopping on a favourable trend.

Resuming is the normal case: rerun the same command and it continues from the
first identity without an accepted scientific outcome.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from livifuser_nav.backbone_handoff import json_bytes, sha256_file  # noqa: E402
from livifuser_nav.confirmatory_plan import (  # noqa: E402
    SCHEDULE_SHA256,
    Identity,
    build_plan,
    identity_state,
    load_schedule,
    locate_schedule,
    locate_world,
    next_identity,
    progress,
    select_subset,
)

EPISODE_SCRIPT = ROOT / "scripts" / "run_live_sim_development_episode.sh"
CONFIRMATORY_ROOT = ROOT / "artifacts" / "simulation" / "confirmatory_v3"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"refusing to run: {message}")


def verify_inputs(runtime_bundle: Path) -> dict[str, str]:
    """Re-derive every frozen identity before the first launch."""

    schedule = locate_schedule(ROOT)
    # load_schedule re-checks the frozen source identity for either layout.
    require(len(load_schedule(schedule)) == 80, "confirmatory schedule subset drifted")
    require(EPISODE_SCRIPT.is_file(), "episode runner is absent")
    require(runtime_bundle.is_file(), f"runtime bundle is absent: {runtime_bundle}")

    from livifuser_nav.runtime_handoff import audit_runtime_handoff

    audit = audit_runtime_handoff(runtime_bundle)
    require(
        audit["status"] == "audit_pass",
        f"runtime bundle failed its audit: {audit['findings']}",
    )
    return {
        "schedule_sha256_excludes_self": SCHEDULE_SHA256,
        "runtime_bundle_sha256": audit["bundle_sha256"],
        "episode_runner_sha256": sha256_file(EPISODE_SCRIPT),
        "plan_module_sha256": sha256_file(ROOT / "src/livifuser_nav/confirmatory_plan.py"),
    }


def launch(identity: Identity, root: Path, attempt: int, dry_run: bool) -> int:
    """Run one episode. Returns the runner's exit status."""

    directory = root / identity.arm / str(identity.seed) / f"{identity.ordinal:04d}"
    attempt_dir = directory / f"attempt_{attempt:03d}"
    world_sdf = locate_world(ROOT, identity.world_sdf)
    world_json = locate_world(ROOT, identity.world_json)

    command = [
        "bash",
        str(EPISODE_SCRIPT),
        str(attempt_dir),
        str(world_sdf),
        str(world_json),
        identity.lidar_condition,
        str(identity.observation_seed),
        identity.arm,
        str(identity.seed),
    ]
    if dry_run:
        print("  would run:", " ".join(command[2:]))
        return 0
    completed = subprocess.run(command, cwd=ROOT, check=False)  # noqa: S603
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/workspace/livifuser/evidence/confirmatory_closed_loop_v1"),
    )
    parser.add_argument(
        "--runtime-bundle", type=Path, default=ROOT / "artifacts/livifuser_runtime_v1_bundle.zip"
    )
    parser.add_argument(
        "--authorize",
        action="store_true",
        help="required; section 12 makes approval an explicit recorded act",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="stop after this many launches (0 = run to completion)",
    )
    parser.add_argument(
        "--arms", default=None, help="comma-separated arms to run; omit for the whole plan"
    )
    parser.add_argument(
        "--conditions", default=None, help="comma-separated conditions; omit for all"
    )
    parser.add_argument(
        "--max-per-cell",
        type=int,
        default=None,
        help="cap identities per (arm, seed, condition); frozen order, first N",
    )
    args = parser.parse_args()

    plan = build_plan(locate_schedule(ROOT))
    full_plan_size = len(plan)
    if args.arms or args.conditions or args.max_per_cell:
        plan = select_subset(
            plan,
            arms=tuple(a.strip() for a in args.arms.split(",")) if args.arms else None,
            conditions=(
                tuple(c.strip() for c in args.conditions.split(","))
                if args.conditions
                else None
            ),
            max_per_cell=args.max_per_cell,
        )
        # A reduced scope is a deviation and must be visible in the log it writes.
        print(
            f"REDUCED SCOPE: {len(plan)} of {full_plan_size} identities "
            f"(arms={args.arms or 'all'}, conditions={args.conditions or 'all'}, "
            f"max_per_cell={args.max_per_cell or 'all'})"
        )
    identities = verify_inputs(args.runtime_bundle)

    state = progress(args.root, plan)
    print(
        f"plan {state['identities_total']} identities; "
        f"complete {state['identities_complete']}; "
        f"remaining {state['identities_remaining']}; "
        f"attempts so far {state['attempts_made']}"
    )
    if state["exhausted_identities"]:
        print("paused identities:", ", ".join(state["exhausted_identities"]))

    if not args.authorize and not args.dry_run:
        raise SystemExit(
            "refusing to run: --authorize is required. Section 9 makes an accepted "
            "scientific outcome permanent, so this cannot be undone."
        )

    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "batch_inputs.json").write_bytes(json_bytes(identities))

    launched = 0
    started = time.time()
    # A dry run writes nothing, so next_identity would return the same identity
    # forever. Track what a real run would have completed, in memory only.
    previewed: set[str] = set()
    while True:
        identity, reason = next_identity(args.root, plan)
        if args.dry_run:
            identity, reason = next(
                (
                    (candidate, "preview")
                    for candidate in plan
                    if candidate.key not in previewed
                    and not identity_state(args.root, candidate)["complete"]
                ),
                (None, "preview complete"),
            )
        if identity is None:
            print(reason)
            break
        if args.dry_run:
            previewed.add(identity.key)
        if args.max_episodes and launched >= args.max_episodes:
            print(f"stopping after {launched} launches as requested")
            break
        attempt = identity_state(args.root, identity)["attempts"] + 1
        print(
            f"[{launched + 1}] {identity.key} attempt {attempt} "
            f"({identity.condition}, {reason})"
        )
        status = launch(identity, args.root, attempt, args.dry_run)
        launched += 1
        if status != 0:
            # The runner already sealed the attempt and recorded why. Whether the
            # identity retries is decided by the plan, not by this exit code.
            print(f"    runner exit {status}; the plan decides what happens next")

    final = progress(args.root, plan)
    elapsed = time.time() - started
    print(
        f"launched {launched} episodes in {elapsed / 60:.1f} min; "
        f"complete {final['identities_complete']}/{final['identities_total']}"
    )
    if final["batch_complete"]:
        print("every identity has an accepted scientific outcome; seal the batch before analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
