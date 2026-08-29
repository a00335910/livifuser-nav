"""Exploratory diagnostic: what do the frozen DINOv3 S+/16 features encode?

STATUS: EXPLORATORY DIAGNOSTIC. Not confirmatory. Reads only the already
verified train/validation feature cache bundle. It never opens a held-out
cache, never trains a policy, never fits or alters a frozen threshold, and
never writes into an existing artifact directory.

Question
--------
The frozen-feature Mahalanobis audit found that all 9,459 designated
in-distribution validation windows exceed the *maximum* training distance.
Two explanations are compatible with that:

  (A) appearance -- the visual skin differs across splits (train wood/brick,
      val amber/plaster), and the frozen backbone encodes surface appearance;
  (B) geometry -- the validation worlds are different room layouts.

The confirmatory dataset happens to permit a clean separation of the two,
because validation reuses two *archetypes* that also appear in training
(`dogleg_corridor`, `straight_corridor`) under a different skin:

  same archetype, different skin   -> train_dogleg   vs val_id_dogleg
  different archetype, same skin   -> train_dogleg   vs train_straight

If (A) dominates, the cross-skin distance is large while the cross-archetype
distance is small. If (B) dominates, the reverse. This is decidable from the
cached features alone, with no training.

The script also evaluates one candidate remedy. `visual_projection` begins
with `nn.LayerNorm(384)`, which normalises across the feature dimension of a
single token; it cannot remove a component shared by all 49 tokens of a frame.
If world identity is carried as such a frame-global component, subtracting the
per-frame token mean should suppress it while preserving spatial structure.
Both raw and token-centred statistics are therefore reported.

Usage
-----
    uv run python scripts/probe_visual_representation.py \
        --bundle artifacts/livifuser_dinov3_splus_cache_v2_bundle.zip \
        --output artifacts/experiments/visual_representation_probe_v1.json
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

BUNDLE_ROOT = "livifuser_dinov3_splus_cache_v2"
SHARD_PREFIX = f"{BUNDLE_ROOT}/shards/"
PATCH_MEMBER = "patch_tokens_7x7_float16.npy"
POOLED_MEMBER = "pooled_features_float32.npy"

#: Archetype suffix shared between a training world and a validation world.
#: These are the two comparisons that separate appearance from geometry.
SHARED_ARCHETYPES = ("dogleg_corridor_001", "straight_corridor_000")


def _load_npy(data: bytes) -> np.ndarray:
    """Load a .npy payload from memory with pickle disabled."""
    return np.load(io.BytesIO(data), allow_pickle=False)


def _shard_names(bundle: zipfile.ZipFile) -> list[str]:
    return sorted(
        name
        for name in bundle.namelist()
        if name.startswith(SHARD_PREFIX) and name.endswith(".zip")
    )


def _world_of(shard_name: str) -> str:
    stem = shard_name[len(SHARD_PREFIX) :]
    return stem.split(".dinov3")[0]


def _episode_dirs(shard: zipfile.ZipFile) -> list[str]:
    seen = set()
    for name in shard.namelist():
        if name.startswith("episodes/") and name.endswith(POOLED_MEMBER):
            seen.add(name[: -len(POOLED_MEMBER)])
    return sorted(seen)


def _sample_rows(count: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    if count <= limit:
        return np.arange(count)
    return np.sort(rng.choice(count, size=limit, replace=False))


def collect_world(
    bundle: zipfile.ZipFile,
    shard_name: str,
    frames_per_episode: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Return pooled and patch-token samples for one world shard."""
    shard = zipfile.ZipFile(io.BytesIO(bundle.read(shard_name)))
    pooled_chunks: list[np.ndarray] = []
    patch_chunks: list[np.ndarray] = []
    episodes = _episode_dirs(shard)
    for episode in episodes:
        pooled = _load_npy(shard.read(episode + POOLED_MEMBER))
        patch = _load_npy(shard.read(episode + PATCH_MEMBER))
        if pooled.ndim != 2 or pooled.shape[1] != 384:
            raise ValueError(f"unexpected pooled shape in {episode}")
        if patch.ndim != 3 or patch.shape[1:] != (49, 384):
            raise ValueError(f"unexpected patch shape in {episode}")
        rows = _sample_rows(pooled.shape[0], frames_per_episode, rng)
        pooled_chunks.append(np.asarray(pooled[rows], dtype=np.float64))
        patch_chunks.append(np.asarray(patch[rows], dtype=np.float64))
    return {
        "world": _world_of(shard_name),
        "split": "val_id" if _world_of(shard_name).startswith("val_id") else "train",
        "episodes": len(episodes),
        "pooled": np.concatenate(pooled_chunks, axis=0),
        "patch": np.concatenate(patch_chunks, axis=0),
    }


def token_center(patch: np.ndarray) -> np.ndarray:
    """Remove the frame-global component shared by all 49 spatial tokens."""
    return patch - patch.mean(axis=1, keepdims=True)


def variance_split(patch: np.ndarray) -> dict[str, float]:
    """Split total patch-token variance into frame-global and spatial parts.

    The frame mean over tokens carries whatever is common to the whole image;
    the residual carries where things are within it.
    """
    frame_mean = patch.mean(axis=1)  # (rows, 384)
    residual = patch - frame_mean[:, None, :]
    global_var = float(frame_mean.var(axis=0).sum())
    spatial_var = float(residual.var(axis=(0, 1)).sum())
    total = global_var + spatial_var
    return {
        "frame_global_variance": global_var,
        "within_frame_spatial_variance": spatial_var,
        "frame_global_fraction": global_var / total if total > 0 else float("nan"),
    }


def _shrinkage_gaussian(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a mean and shrinkage precision, matching the sweep's estimator shape."""
    mean = features.mean(axis=0)
    centered = features - mean
    covariance = centered.T @ centered / max(len(features) - 1, 1)
    trace = float(np.trace(covariance)) / covariance.shape[0]
    shrunk = 0.9 * covariance + 0.1 * trace * np.eye(covariance.shape[0])
    return mean, np.linalg.inv(shrunk)


def mahalanobis(features: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    centered = features - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", centered, precision, centered))


def separation(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised mean separation between two feature sets.

    Distance between the two means, expressed in units of their pooled
    within-set standard deviation, so appearance and geometry comparisons
    are on the same scale.
    """
    delta = a.mean(axis=0) - b.mean(axis=0)
    pooled = np.sqrt(0.5 * (a.var(axis=0) + b.var(axis=0))) + 1e-12
    return float(np.linalg.norm(delta / pooled) / np.sqrt(len(delta)))


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    bundle = zipfile.ZipFile(args.bundle)
    worlds = [
        collect_world(bundle, name, args.frames_per_episode, rng)
        for name in _shard_names(bundle)
    ]
    by_world = {world["world"]: world for world in worlds}

    train = [w for w in worlds if w["split"] == "train"]
    validation = [w for w in worlds if w["split"] == "val_id"]
    if not train or not validation:
        raise ValueError("bundle did not contain both train and val_id shards")

    train_pooled = np.concatenate([w["pooled"] for w in train], axis=0)
    val_pooled = np.concatenate([w["pooled"] for w in validation], axis=0)
    train_patch = np.concatenate([w["patch"] for w in train], axis=0)
    val_patch = np.concatenate([w["patch"] for w in validation], axis=0)

    # --- Test 1: what does the representation spend its variance on? --------
    variance = {
        "train": variance_split(train_patch),
        "validation": variance_split(val_patch),
    }

    # --- Test 2: appearance versus geometry --------------------------------
    # Same archetype across the skin boundary vs different archetype within it.
    cross_skin: dict[str, float] = {}
    for archetype in SHARED_ARCHETYPES:
        train_key = f"train_{archetype}"
        val_key = f"val_id_{archetype}"
        if train_key in by_world and val_key in by_world:
            cross_skin[archetype] = separation(
                by_world[train_key]["pooled"], by_world[val_key]["pooled"]
            )
    cross_archetype: dict[str, float] = {}
    train_names = sorted(w["world"] for w in train)
    for i, left in enumerate(train_names):
        for right in train_names[i + 1 :]:
            cross_archetype[f"{left}|{right}"] = separation(
                by_world[left]["pooled"], by_world[right]["pooled"]
            )

    mean_cross_skin = float(np.mean(list(cross_skin.values()))) if cross_skin else float("nan")
    mean_cross_arch = (
        float(np.mean(list(cross_archetype.values()))) if cross_archetype else float("nan")
    )

    # --- Test 3: does token centring close the train/validation gap? -------
    def gap(train_feat: np.ndarray, val_feat: np.ndarray) -> dict[str, float]:
        mean, precision = _shrinkage_gaussian(train_feat)
        train_d = mahalanobis(train_feat, mean, precision)
        val_d = mahalanobis(val_feat, mean, precision)
        above = float((val_d > train_d.max()).mean())
        return {
            "train_median": float(np.median(train_d)),
            "train_p95": float(np.percentile(train_d, 95)),
            "train_max": float(train_d.max()),
            "validation_min": float(val_d.min()),
            "validation_median": float(np.median(val_d)),
            "validation_max": float(val_d.max()),
            "validation_fraction_above_train_max": above,
        }

    raw_gap = gap(train_pooled, val_pooled)
    centered_train = token_center(train_patch).reshape(len(train_patch), -1)
    centered_val = token_center(val_patch).reshape(len(val_patch), -1)
    # Project to a comparable dimensionality before fitting, so the Gaussian
    # estimator is not compared across wildly different feature widths.
    projection = rng.standard_normal((centered_train.shape[1], 384)) / np.sqrt(384)
    centered_gap = gap(centered_train @ projection, centered_val @ projection)

    return {
        "status": "EXPLORATORY_DIAGNOSTIC_NOT_CONFIRMATORY",
        "bundle": str(args.bundle),
        "seed": args.seed,
        "frames_per_episode": args.frames_per_episode,
        "worlds": [
            {
                "world": w["world"],
                "split": w["split"],
                "episodes": w["episodes"],
                "sampled_frames": int(len(w["pooled"])),
            }
            for w in worlds
        ],
        "test_1_variance_decomposition": variance,
        "test_2_appearance_versus_geometry": {
            "cross_skin_same_archetype": cross_skin,
            "cross_archetype_same_skin": cross_archetype,
            "mean_cross_skin": mean_cross_skin,
            "mean_cross_archetype": mean_cross_arch,
            "appearance_dominance_ratio": (
                mean_cross_skin / mean_cross_arch if mean_cross_arch else float("nan")
            ),
        },
        "test_3_mahalanobis_gap": {
            "raw_pooled": raw_gap,
            "token_centered_patch": centered_gap,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("artifacts/livifuser_dinov3_splus_cache_v2_bundle.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/visual_representation_probe_v1.json"),
    )
    parser.add_argument("--frames-per-episode", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {args.output}")

    report = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), "utf-8")

    appearance = report["test_2_appearance_versus_geometry"]
    print(f"wrote {args.output}")
    print(f"  mean cross-skin separation      : {appearance['mean_cross_skin']:.4f}")
    print(f"  mean cross-archetype separation : {appearance['mean_cross_archetype']:.4f}")
    print(f"  appearance dominance ratio      : {appearance['appearance_dominance_ratio']:.3f}")
    gaps = report["test_3_mahalanobis_gap"]
    key = "validation_fraction_above_train_max"
    print(f"  validation beyond train max, raw     : {gaps['raw_pooled'][key]:.4f}")
    print(f"  validation beyond train max, centred : {gaps['token_centered_patch'][key]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
