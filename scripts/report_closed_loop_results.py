"""Report the sealed closed-loop confirmatory results.

Runs the estimator written before any outcome was seen. Section 14.1 asks
whether full-model navigation success exceeds every one of LiDAR-only, RGB-only
and concatenation, on C0 and on at least one of C1 or C4, with the paired
cluster-bootstrap interval excluding zero and the sign holding in all three
seeds. C4 was never collected, so C1 is the only corruption condition available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from livifuser_nav.closed_loop_analysis import (  # noqa: E402
    cluster_bootstrap,
    count_attempts,
    load_episodes,
    paired_contrast,
    rate_table,
    sign_test_two_worlds,
)
from livifuser_nav.confirmatory_plan import build_plan, locate_schedule  # noqa: E402

BASELINES = ("lidar_only", "rgb_only", "concat")
SEEDS = (20260805, 20260806, 20260807)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    plan = build_plan(locate_schedule(ROOT))
    episodes = load_episodes(args.root, plan)
    report: dict = {"episodes": len(episodes), "operational": count_attempts(args.root)}

    print("=" * 74)
    print("SUCCESS RATE per arm x condition x seed (numerator/denominator)")
    print("=" * 74)
    table = rate_table(episodes, "success")
    for cond in ("C0", "C1", "C3b"):
        print(f"\n  {cond}")
        for arm in ("full",) + BASELINES:
            cells = [(k, v) for k, v in table.items() if k[0] == arm and k[1] == cond]
            if not cells:
                print(f"    {arm:12s}  (no episodes)")
                continue
            parts = []
            for (_, _, seed), v in sorted(cells, key=lambda kv: str(kv[0][2])):
                parts.append(f"s{str(seed)[-1]}: {v['numerator']:2d}/{v['denominator']:2d}")
            tot_n = sum(v["numerator"] for _, v in cells)
            tot_d = sum(v["denominator"] for _, v in cells)
            allrate = tot_n / tot_d if tot_d else float("nan")
            print(
                f"    {arm:12s}  " + "  ".join(parts)
                + f"   | all {tot_n}/{tot_d} = {allrate:.3f}"
            )
    report["success_rates"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in table.items()}

    print("\n" + "=" * 74)
    print("COLLISION RATE per arm x condition (all seeds pooled for readability)")
    print("=" * 74)
    coll = rate_table(episodes, "collision")
    for cond in ("C0", "C1", "C3b"):
        row = []
        for arm in ("full",) + BASELINES:
            cells = [v for k, v in coll.items() if k[0] == arm and k[1] == cond]
            if not cells:
                row.append(f"{arm}: -")
                continue
            n = sum(c["numerator"] for c in cells)
            d = sum(c["denominator"] for c in cells)
            row.append(f"{arm}: {n}/{d}={n/d:.2f}")
        print(f"  {cond:4s}  " + "   ".join(row))
    report["collision_rates"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in coll.items()}

    print("\n" + "=" * 74)
    print("SECTION 14.1  full - baseline, paired by world/ordinal/seed")
    print("  10,000-replicate hierarchical cluster bootstrap, worlds then episodes")
    print("=" * 74)
    report["contrasts"] = {}
    for cond in ("C0", "C1"):
        print(f"\n  condition {cond}")
        for base in BASELINES:
            paired = paired_contrast(episodes, "success", "full", base, cond)
            if not paired:
                print(f"    full - {base:12s}  (no paired episodes)")
                continue
            npairs = sum(len(v) for v in paired.values())
            ci = cluster_bootstrap(paired)
            sign = sign_test_two_worlds(paired)
            verdict = "EXCLUDES ZERO" if ci["excludes_zero"] else "includes zero"
            print(
                f"    full - {base:12s}  diff {ci['point']:+.3f}"
                f"  95% CI [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]"
                f"  {verdict}   pairs={npairs} worlds={len(paired)}"
            )
            report["contrasts"][f"{cond}|full-{base}"] = {**ci, "sign_test": sign, "pairs": npairs}

    print("\n" + "=" * 74)
    print("SECTION 14.1 VERDICT")
    print("=" * 74)
    for cond in ("C0", "C1"):
        checks = []
        for base in BASELINES:
            key = f"{cond}|full-{base}"
            c = report["contrasts"].get(key)
            if c is None:
                checks.append((base, None, None))
            else:
                checks.append((base, c["point"] > 0, c["excludes_zero"]))
        all_exceed = all(x[1] for x in checks if x[1] is not None) and len(checks) == 3
        all_excl = all(x[2] for x in checks if x[2] is not None) and len(checks) == 3
        print(f"\n  {cond}:")
        for base, exceeds, excl in checks:
            mark = "?" if exceeds is None else ("yes" if exceeds else "NO")
            emark = "?" if excl is None else ("yes" if excl else "NO")
            print(
                f"    full exceeds {base:12s}: {mark:4s}"
                f"   interval excludes zero: {emark}"
            )
        print(
            f"    -> exceeds all three: {all_exceed};"
            f"  all intervals exclude zero: {all_excl}"
        )
    print("\n  C4 was never collected, so 14.1 rests on C1 alone.")

    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
            newline="\n",
        )
        print(f"\n  written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
