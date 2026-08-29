#!/usr/bin/env python3
"""Score Mahalanobis separation against held-out evaluation probe recordings.

Uses the Gaussian fitted by `run_baseline_sweep.py` (train-split pooled
features only) so the AUROC here and the sweep's risk–coverage curves share
one fit. In-distribution negatives are validation-split frames; probes are
evaluation-only and must never enter the fit. Sealed test episodes must not be
passed to this script. ``--probe-kind`` keeps purpose-recorded OOD evidence
separate from the weaker, confounded shifted-distribution substitute.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from livifuser_nav.dino_cache import TEMPORARY_BACKBONE_LABEL  # noqa: E402
from livifuser_nav.evaluation import auroc, mahalanobis_distances  # noqa: E402
from livifuser_nav.learning_data import sha256_file  # noqa: E402


def load_pooled(cache_root: Path) -> tuple[str, np.ndarray, str]:
    manifest = json.loads((cache_root / "manifest.json").read_text("utf-8"))
    if manifest["backbone"]["label"] != TEMPORARY_BACKBONE_LABEL:
        raise ValueError(f"{cache_root} backbone label is not the approved baseline")
    pooled = np.load(cache_root / "pooled_features_float32.npy")
    return (
        str(manifest["run_id"]),
        np.asarray(pooled, dtype=np.float64),
        sha256_file(cache_root / "manifest.json"),
    )


def distance_summary(distances: np.ndarray) -> dict[str, float]:
    return {
        "count": int(distances.size),
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(distances.max()),
        "min": float(distances.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-output",
        type=Path,
        required=True,
        help="baseline-sweep output directory holding the fitted Gaussian",
    )
    parser.add_argument("--id-cache", type=Path, action="append", required=True)
    parser.add_argument(
        "--probe-cache",
        "--ood-cache",
        dest="probe_cache",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--probe-kind",
        choices=("purpose_recorded_ood", "shifted_distribution"),
        default="purpose_recorded_ood",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    mean = np.load(args.sweep_output / "mahalanobis_mean.npy")
    precision = np.load(args.sweep_output / "mahalanobis_precision.npy")
    sweep_summary = json.loads((args.sweep_output / "summary.json").read_text("utf-8"))

    id_records = []
    id_distances = []
    for cache_root in args.id_cache:
        run_id, pooled, manifest_hash = load_pooled(cache_root)
        distances = mahalanobis_distances(pooled, mean, precision)
        id_distances.append(distances)
        id_records.append(
            {
                "run_id": run_id,
                "cache": str(cache_root.resolve()),
                "cache_manifest_sha256": manifest_hash,
                "distance": distance_summary(distances),
            }
        )
    probe_records = []
    probe_distances = []
    for cache_root in args.probe_cache:
        run_id, pooled, manifest_hash = load_pooled(cache_root)
        distances = mahalanobis_distances(pooled, mean, precision)
        probe_distances.append(distances)
        probe_records.append(
            {
                "run_id": run_id,
                "cache": str(cache_root.resolve()),
                "cache_manifest_sha256": manifest_hash,
                "distance": distance_summary(distances),
            }
        )
    id_all = np.concatenate(id_distances)
    probe_all = np.concatenate(probe_distances)

    if args.probe_kind == "purpose_recorded_ood":
        disposition = (
            "Mahalanobis OOD separation on the temporary ViT-S/16 backbone; "
            "purpose-recorded probes are evaluation-only and never enter the Gaussian fit"
        )
        limitation = None
    else:
        disposition = (
            "Mahalanobis shifted-recording-context separation on the temporary ViT-S/16 "
            "backbone; probes are evaluation-only and never enter the Gaussian fit"
        )
        limitation = (
            "The probes confound room, layout, and operator interface. This AUROC measures "
            "detection of a different recording context, not novel-object OOD detection."
        )

    result = {
        "disposition": disposition,
        "probe_kind": args.probe_kind,
        "claim_limitation": limitation,
        "sweep_id": sweep_summary["sweep_id"],
        "sweep_config_sha256": sweep_summary["config_sha256"],
        "sweep_git": sweep_summary["git"],
        "mahalanobis_fit": sweep_summary["mahalanobis"],
        "in_distribution": id_records,
        "evaluation_probes": probe_records,
        "auroc_probe_vs_id": auroc(probe_all, id_all),
        "per_probe_auroc_vs_id": {
            record["run_id"]: auroc(distances, id_all)
            for record, distances in zip(probe_records, probe_distances, strict=True)
        },
        "overall": {
            "in_distribution": distance_summary(id_all),
            "probe": distance_summary(probe_all),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "probe_kind": args.probe_kind,
                "auroc_probe_vs_id": result["auroc_probe_vs_id"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
