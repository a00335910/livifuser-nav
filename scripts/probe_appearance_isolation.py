"""Post-hoc diagnostic: isolate appearance from geometry in frozen features.

STATUS: EXPLORATORY POST-HOC ANALYSIS. Performed after the one-time held-out
evaluation was unblinded. It carries no preregistered acceptance criterion and
may not be reported as a confirmatory finding. It trains nothing, fits no
threshold, and alters no artifact.

Why this exists
---------------
An earlier probe compared training worlds against validation worlds of the
same archetype and found a 3.19x separation ratio. That comparison does not
isolate appearance: train and validation worlds of one archetype still differ
in dimensions, obstacle placement, start pose, goal, and seed, so geometry and
appearance move together.

The C1 condition provides the missing control. C1 changes only the rendered
scene (ambient and directional lighting, and an exact channel permutation of
every visual material). Geometry, collision, privileged expert labels, the
analytic LiDAR, camera intrinsics, resolution, encoding, and rate are all
unchanged, and each C1 episode shares its world, start, goal, episode index,
and observation seed with a C0 partner. Comparing paired C0/C1 episodes
therefore varies appearance with geometry held fixed.

Two contrasts are computed on the same scale:

  appearance : C0 vs C1, same episode identity      (geometry fixed)
  geometry   : C0 vs C0, different episode or world (appearance fixed)

Caveat, stated in the report: paired episodes do not contain identical frame
counts, because causal association rejects a slightly different set of ticks
under each rendering. The comparison is therefore distributional over an
episode rather than frame-by-frame.

Usage
-----
    uv run python scripts/probe_appearance_isolation.py \
        --output artifacts/experiments/appearance_isolation_probe_v1.json
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

BUNDLE_ROOT = "livifuser_dinov3_splus_heldout_cache_v1"
POOLED = "pooled_features_float32.npy"
PATCH = "patch_tokens_7x7_float16.npy"

#: The sealed held-out cache. Any other bundle is refused.
BUNDLE_SHA256 = "7FB323948427AB6FC1F5F82F2CEF5E66DDB51F056C031132B2EE8C9B9F0484E5"

WORLDS = ("dogleg_corridor_001", "straight_corridor_000")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def _open_shard(bundle: zipfile.ZipFile, world: str, condition: str) -> zipfile.ZipFile:
    target = f"test_ood_{world}_{condition}.dinov3_vits16plus_cache.zip"
    name = next(n for n in bundle.namelist() if n.endswith(target))
    return zipfile.ZipFile(io.BytesIO(bundle.read(name)))


def _episodes(shard: zipfile.ZipFile) -> dict[str, str]:
    """Map episode ordinal ('e000') to its directory prefix."""
    found: dict[str, str] = {}
    for name in shard.namelist():
        if name.startswith("episodes/") and name.endswith(POOLED):
            directory = name[: -len(POOLED)]
            ordinal = directory.split("/")[1].split("_")[-2]
            found[ordinal] = directory
    return found


def separation(a: np.ndarray, b: np.ndarray) -> float:
    """Mean separation in units of pooled within-set standard deviation."""
    delta = a.mean(axis=0) - b.mean(axis=0)
    pooled = np.sqrt(0.5 * (a.var(axis=0) + b.var(axis=0))) + 1e-12
    return float(np.linalg.norm(delta / pooled) / np.sqrt(len(delta)))


def token_center(patch: np.ndarray) -> np.ndarray:
    return patch - patch.mean(axis=1, keepdims=True)


def _sample(count: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    if count <= limit:
        return np.arange(count)
    return np.sort(rng.choice(count, size=limit, replace=False))


def collect(
    bundle: zipfile.ZipFile, frames: int, rng: np.random.Generator
) -> dict[tuple[str, str, str], dict[str, np.ndarray]]:
    """Return {(world, condition, ordinal): {pooled, centered}}."""
    out: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for world in WORLDS:
        for condition in ("c0", "c1"):
            shard = _open_shard(bundle, world, condition)
            for ordinal, directory in _episodes(shard).items():
                pooled = _load(shard.read(directory + POOLED)).astype(np.float64)
                patch = _load(shard.read(directory + PATCH)).astype(np.float64)
                rows = _sample(len(pooled), frames, rng)
                centered = token_center(patch[rows]).reshape(len(rows), -1)
                out[(world, condition, ordinal)] = {
                    "pooled": pooled[rows],
                    "centered": centered,
                }
    return out


def contrasts(data: dict, field: str) -> dict[str, list[float]]:
    """Appearance-isolated and appearance-fixed geometry contrasts."""
    appearance: list[float] = []
    geometry_within: list[float] = []
    geometry_cross: list[float] = []

    for world in WORLDS:
        ordinals = sorted(o for (w, c, o) in data if w == world and c == "c0")
        # Appearance: same episode identity, C0 versus C1.
        for ordinal in ordinals:
            key0, key1 = (world, "c0", ordinal), (world, "c1", ordinal)
            if key1 in data:
                appearance.append(separation(data[key0][field], data[key1][field]))
        # Geometry, appearance held fixed: different episodes, same world and skin.
        for left, right in itertools.combinations(ordinals, 2):
            geometry_within.append(
                separation(data[(world, "c0", left)][field], data[(world, "c0", right)][field])
            )

    # Geometry across archetypes, appearance held fixed.
    left_world, right_world = WORLDS
    for left in sorted(o for (w, c, o) in data if w == left_world and c == "c0"):
        for right in sorted(o for (w, c, o) in data if w == right_world and c == "c0"):
            geometry_cross.append(
                separation(
                    data[(left_world, "c0", left)][field],
                    data[(right_world, "c0", right)][field],
                )
            )

    return {
        "appearance_isolated": appearance,
        "geometry_within_world": geometry_within,
        "geometry_cross_archetype": geometry_cross,
    }


def bootstrap_ratio(
    appearance: list[float], geometry: list[float], rng: np.random.Generator, replicates: int
) -> dict[str, float]:
    app = np.asarray(appearance)
    geo = np.asarray(geometry)
    draws = np.empty(replicates)
    for i in range(replicates):
        a = rng.choice(app, size=len(app), replace=True).mean()
        g = rng.choice(geo, size=len(geo), replace=True).mean()
        draws[i] = a / g if g > 0 else np.nan
    finite = draws[np.isfinite(draws)]
    return {
        "ratio": float(app.mean() / geo.mean()),
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "replicates": int(len(finite)),
    }


def summarise(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    actual = _sha256(args.bundle)
    if actual != BUNDLE_SHA256:
        raise SystemExit(
            f"held-out cache SHA-256 mismatch\n  expected {BUNDLE_SHA256}\n  actual   {actual}"
        )
    bundle = zipfile.ZipFile(args.bundle)

    primary_rng = np.random.default_rng(args.seed)
    data = collect(bundle, args.frames_per_episode, primary_rng)

    report: dict[str, Any] = {
        "status": "EXPLORATORY_POST_HOC_ANALYSIS_NOT_CONFIRMATORY",
        "bundle_sha256": actual,
        "seed": args.seed,
        "frames_per_episode": args.frames_per_episode,
        "note": (
            "Paired C0/C1 episodes share world, start, goal, episode index and "
            "observation seed, and differ only in the rendered scene. Frame "
            "counts differ slightly because causal association rejects a "
            "different set of ticks under each rendering, so the comparison is "
            "distributional over an episode rather than frame-by-frame."
        ),
    }

    for field in ("pooled", "centered"):
        raw = contrasts(data, field)
        boot_rng = np.random.default_rng(args.seed + 1)
        report[field] = {
            "appearance_isolated": summarise(raw["appearance_isolated"]),
            "geometry_within_world": summarise(raw["geometry_within_world"]),
            "geometry_cross_archetype": summarise(raw["geometry_cross_archetype"]),
            "ratio_vs_within_world": bootstrap_ratio(
                raw["appearance_isolated"],
                raw["geometry_within_world"],
                boot_rng,
                args.replicates,
            ),
            "ratio_vs_cross_archetype": bootstrap_ratio(
                raw["appearance_isolated"],
                raw["geometry_cross_archetype"],
                np.random.default_rng(args.seed + 2),
                args.replicates,
            ),
        }

    # Sensitivity: sampler seed and frames per episode.
    sensitivity: list[dict[str, Any]] = []
    for frames in args.sensitivity_frames:
        for offset in range(args.sensitivity_seeds):
            rng = np.random.default_rng(args.seed + 100 + offset)
            sample = collect(bundle, frames, rng)
            raw = contrasts(sample, "pooled")
            sensitivity.append(
                {
                    "frames_per_episode": frames,
                    "sampler_seed": args.seed + 100 + offset,
                    "appearance_mean": float(np.mean(raw["appearance_isolated"])),
                    "geometry_within_mean": float(np.mean(raw["geometry_within_world"])),
                    "ratio_vs_within_world": float(
                        np.mean(raw["appearance_isolated"])
                        / np.mean(raw["geometry_within_world"])
                    ),
                }
            )
    report["sensitivity"] = sensitivity
    ratios = [entry["ratio_vs_within_world"] for entry in sensitivity]
    report["sensitivity_summary"] = {
        "ratio_min": float(np.min(ratios)),
        "ratio_max": float(np.max(ratios)),
        "ratio_mean": float(np.mean(ratios)),
        "configurations": len(ratios),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("artifacts/livifuser_dinov3_splus_heldout_cache_v1_bundle.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/appearance_isolation_probe_v1.json"),
    )
    parser.add_argument("--frames-per-episode", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--sensitivity-frames", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument("--sensitivity-seeds", type=int, default=3)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {args.output}")

    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), "utf-8")

    print(f"wrote {args.output}")
    for field in ("pooled", "centered"):
        block = report[field]
        app = block["appearance_isolated"]["mean"]
        within = block["geometry_within_world"]["mean"]
        cross = block["geometry_cross_archetype"]["mean"]
        print(f"  [{field}]")
        print(f"    appearance, geometry fixed   : {app:.4f}")
        print(f"    geometry within world        : {within:.4f}")
        print(f"    geometry cross archetype     : {cross:.4f}")
        for label, key in (
            ("vs within-world  ", "ratio_vs_within_world"),
            ("vs cross-archetype", "ratio_vs_cross_archetype"),
        ):
            r = block[key]
            lo, hi = r["ci_low"], r["ci_high"]
            print(f"    ratio {label}: {r['ratio']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    s = report["sensitivity_summary"]
    lo, hi, n = s["ratio_min"], s["ratio_max"], s["configurations"]
    print(f"  sensitivity ratio: {lo:.3f} to {hi:.3f} over {n} configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
