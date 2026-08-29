#!/usr/bin/env python3
"""Audit the immutable Kaggle simulation-training result bundle in place."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from livifuser_nav.evaluation import per_episode_summary, risk_coverage  # noqa: E402
from livifuser_nav.model import LiViFuserPolicy  # noqa: E402

CONFIG_SHA256 = "76680BCE45A67D5D91F42660D5EC25F90450B281838CE311E329477C4E36F09E"
CLOUD_MANIFEST_SHA256 = "EB98B43B049FA6E2997176A43062D51926968390D1C07314D70DD56E8090ED0A"
ROOT = "livifuser_simulation_sweep_v1/"
PARTITIONS = {
    0: ("full", "rgb_only", "no_fov_mask", "no_temporal"),
    1: ("lidar_only", "concat", "no_gate", "full_mean_only"),
}
ROOT_FILES = (
    "summary.json",
    "execution_summary.json",
    "worker_0.log",
    "worker_1.log",
    "execution_plan.json",
    "RESULT_BUNDLE_MANIFEST.json",
)
WORKER_FILES = (
    "summary.json",
    "mahalanobis_precision.npy",
    "mahalanobis_mean.npy",
    "progress.json",
    "run_context.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(archive: zipfile.ZipFile, member: str) -> dict[str, Any]:
    value = json.loads(archive.read(member))
    require(isinstance(value, dict), f"expected JSON object: {member}")
    return value


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def expected_members(
    runs: dict[str, dict[str, Any]], seeds: list[int]
) -> tuple[set[str], set[str]]:
    files = {ROOT + name for name in ROOT_FILES}
    directories = {ROOT, ROOT + "worker_0/", ROOT + "worker_1/"}
    for worker, names in PARTITIONS.items():
        files.update(ROOT + f"worker_{worker}/{name}" for name in WORKER_FILES)
        for name in names:
            require(name in runs, f"partition names unknown run: {name}")
            run_root = ROOT + f"worker_{worker}/{name}/"
            directories.add(run_root)
            for seed in seeds:
                seed_root = run_root + f"seed_{seed}/"
                directories.add(seed_root)
                files.update((seed_root + "result.json", seed_root + "checkpoint.pt"))
    return files, directories


def validate_episode_summary(
    member: str,
    episode_ids: list[str],
    values: np.ndarray,
    stored: dict[str, dict[str, Any]],
) -> None:
    recomputed = per_episode_summary(episode_ids, values)
    require(set(recomputed) == set(stored), f"episode identities drift: {member}")
    for episode, expected in recomputed.items():
        observed = stored[episode]
        require(
            observed["window_count"] == expected["window_count"],
            f"episode window count drift: {member}:{episode}",
        )
        for field in ("mean", "median", "p95", "max"):
            require(
                close(observed[field], expected[field]),
                f"episode {field} drift: {member}:{episode}",
            )


def validate_risk_curve(
    member: str,
    errors: np.ndarray,
    distances: np.ndarray,
    stored: list[dict[str, float]],
) -> None:
    recomputed = risk_coverage(errors, distances)
    require(len(stored) == len(recomputed), f"risk-coverage length drift: {member}")
    for index, (observed, expected) in enumerate(zip(stored, recomputed, strict=True)):
        for field in ("coverage", "risk", "distance_threshold"):
            require(
                close(observed[field], expected[field]),
                f"risk-coverage {field} drift: {member}:{index}",
            )


def summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["seed"]))
    mse = [float(row["macro_episode_normalized_mse"]) for row in ordered]
    nll = [float(row["macro_episode_nll"]) for row in ordered]
    seconds = [float(row["training_seconds"]) for row in ordered]
    loss = str(ordered[0]["loss"])
    return {
        "variant": ordered[0]["variant"],
        "loss": loss,
        "parameter_count": int(ordered[0]["parameter_count"]),
        "per_seed": [
            {
                "seed": int(row["seed"]),
                "macro_episode_normalized_mse": float(row["macro_episode_normalized_mse"]),
                "macro_episode_nll": float(row["macro_episode_nll"]),
                "training_seconds": float(row["training_seconds"]),
            }
            for row in ordered
        ],
        "mse_mean": statistics.mean(mse),
        "mse_sample_sd": statistics.stdev(mse),
        "nll_mean": statistics.mean(nll),
        "nll_sample_sd": statistics.stdev(nll),
        "nll_interpretable": loss != "mean_only",
        "training_seconds_total": sum(seconds),
    }


def audit(archive_path: Path, config_path: Path) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    config_path = config_path.resolve()
    require(archive_path.is_file(), f"missing results archive: {archive_path}")
    require(config_path.is_file(), f"missing frozen config: {config_path}")
    require(sha256_file(config_path) == CONFIG_SHA256, "local frozen config hash drift")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["seeds"]]
    runs = {str(row["name"]): row for row in config["runs"]}
    require(seeds == [20260805, 20260806, 20260807], "frozen seed set drift")
    require(
        set(runs) == {name for names in PARTITIONS.values() for name in names},
        "frozen run identity set drift",
    )
    expected_files, expected_directories = expected_members(runs, seeds)

    archive_sha256 = sha256_file(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        unsafe = sorted(
            name
            for name in names
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        )
        require(not duplicates, f"duplicate ZIP members: {duplicates}")
        require(not unsafe, f"unsafe ZIP members: {unsafe}")
        require(archive.testzip() is None, "ZIP CRC verification failed")
        observed_files = {name for name in names if not name.endswith("/")}
        observed_directories = {name for name in names if name.endswith("/")}
        require(
            observed_files == expected_files,
            "file member set drift: "
            f"extra={sorted(observed_files - expected_files)}, "
            f"missing={sorted(expected_files - observed_files)}",
        )
        require(
            observed_directories == expected_directories,
            "directory member set drift: "
            f"extra={sorted(observed_directories - expected_directories)}, "
            f"missing={sorted(expected_directories - observed_directories)}",
        )
        member_hashes = {name: sha256_bytes(archive.read(name)) for name in sorted(observed_files)}

        manifest = read_json(archive, ROOT + "RESULT_BUNDLE_MANIFEST.json")
        summary_bytes = archive.read(ROOT + "summary.json")
        summary = json.loads(summary_bytes)
        require(manifest["schema_version"] == 1, "result manifest schema drift")
        require(manifest["config_sha256"] == CONFIG_SHA256, "manifest config hash drift")
        require(
            manifest["summary_sha256"] == sha256_bytes(summary_bytes),
            "summary hash does not match result manifest",
        )
        require(manifest["result_count"] == 24, "manifest result count drift")
        require(summary["result_count"] == 24, "summary result count drift")
        require(summary["config_sha256"] == CONFIG_SHA256, "summary config hash drift")
        require(summary["sweep_id"] == config["sweep_id"], "summary sweep identity drift")

        execution_plan = read_json(archive, ROOT + "execution_plan.json")
        execution_summary = read_json(archive, ROOT + "execution_summary.json")
        require(
            execution_plan["config_sha256"] == CONFIG_SHA256,
            "execution-plan config hash drift",
        )
        require(
            execution_plan["data_plan_sha256"] == manifest["data_plan_sha256"],
            "data-plan hash disagreement",
        )
        require(
            execution_plan["partitions"]
            == [list(PARTITIONS[index]) for index in sorted(PARTITIONS)],
            "execution partition drift",
        )
        require(execution_plan["cuda_devices"] == ["0", "1"], "CUDA plan drift")
        cloud = execution_plan["cloud_bundle"]
        require(
            str(cloud["manifest_sha256"]).upper() == CLOUD_MANIFEST_SHA256,
            "cloud manifest identity drift",
        )
        require(
            cloud["file_count"] == 27 and cloud["total_bytes"] == 260164,
            "cloud bundle count drift",
        )
        workers = execution_summary["workers"]
        require(len(workers) == 2, "execution worker count drift")
        for index, worker_record in enumerate(workers):
            require(worker_record["worker"] == index, f"worker index drift: {index}")
            require(
                worker_record["run_names"] == list(PARTITIONS[index]),
                f"worker plan drift: {index}",
            )
            require(
                worker_record["accelerator"] == f"physical_cuda:{index}",
                f"accelerator drift: {index}",
            )
            require(worker_record["returncode"] == 0, f"worker failed: {index}")

        summary_rows = {(str(row["name"]), int(row["seed"])): row for row in summary["results"]}
        expected_identities = {(name, seed) for name in runs for seed in seeds}
        require(
            len(summary_rows) == 24 and set(summary_rows) == expected_identities,
            "exact 24-result identity set mismatch",
        )

        contexts: dict[int, dict[str, Any]] = {}
        gaussian_hashes: dict[str, str] = {}
        gaussian_arrays: dict[str, np.ndarray] = {}
        result_hashes: dict[str, str] = {}
        checkpoint_hashes: dict[str, str] = {}
        reference_window_identity: tuple[list[str], list[int]] | None = None
        reference_distances: list[float] | None = None
        result_details: dict[tuple[str, int], dict[str, Any]] = {}
        worker_summary_pins = {int(row["worker"]): row for row in summary["worker_summaries"]}
        require(set(worker_summary_pins) == {0, 1}, "worker summary pin set drift")

        for worker, names_in_partition in PARTITIONS.items():
            worker_root = ROOT + f"worker_{worker}/"
            worker_summary_bytes = archive.read(worker_root + "summary.json")
            require(
                sha256_bytes(worker_summary_bytes) == worker_summary_pins[worker]["sha256"],
                f"worker summary hash drift: {worker}",
            )
            worker_summary = json.loads(worker_summary_bytes)
            worker_rows = {
                (str(row["name"]), int(row["seed"])): row for row in worker_summary["results"]
            }
            expected_worker_ids = {(name, seed) for name in names_in_partition for seed in seeds}
            require(
                set(worker_rows) == expected_worker_ids,
                f"worker result identity drift: {worker}",
            )
            require(
                all(row == summary_rows[identity] for identity, row in worker_rows.items()),
                f"worker/combined summary disagreement: {worker}",
            )
            progress = read_json(archive, worker_root + "progress.json")
            progress_ids = {(str(row["name"]), int(row["seed"])) for row in progress["completed"]}
            require(
                progress["completed_result_count"] == 12
                and progress["planned_result_count"] == 12
                and progress_ids == expected_worker_ids,
                f"worker progress drift: {worker}",
            )

            context = read_json(archive, worker_root + "run_context.json")
            contexts[worker] = context
            require(context["config_sha256"] == CONFIG_SHA256, "context config hash drift")
            require(context["backbone"] == config["backbone"], "backbone contract drift")
            require(context["backbone_deviation"] is None, "unexpected backbone deviation")
            require(
                context["execution_run_names"] == list(names_in_partition),
                f"context partition drift: {worker}",
            )
            device = context["device"]
            require(
                device["cuda_available"] is True
                and device["device_name"] == "Tesla T4"
                and device["compute_capability"] == [7, 5],
                f"training device provenance drift: {worker}",
            )
            require(
                context["git"]["revision"] == cloud["git_revision"]
                and context["git"]["state"] == "verified_cloud_bundle",
                "cloud Git provenance drift",
            )
            require(
                context["mahalanobis"]["fit_rows"] == 56128
                and close(context["mahalanobis"]["shrinkage"], 0.1),
                "Gaussian fit provenance drift",
            )
            for split, expected_episodes, expected_windows in (
                ("train", 120, 41367),
                ("validation", 30, 9459),
            ):
                split_record = context["splits"][split]
                counts = split_record["episode_window_counts"]
                require(
                    len(split_record["exports"]) == expected_episodes
                    and len(split_record["export_manifest_sha256"]) == expected_episodes
                    and len(split_record["cache_manifest_sha256"]) == expected_episodes
                    and len(counts) == expected_episodes
                    and sum(int(value) for value in counts.values()) == expected_windows,
                    f"{split} provenance/count drift",
                )
                require(
                    not any(episode.startswith(("test_id_", "test_ood_")) for episode in counts),
                    f"held-out identity entered {split}",
                )

            for filename, shape in (
                ("mahalanobis_mean.npy", (384,)),
                ("mahalanobis_precision.npy", (384, 384)),
            ):
                payload = archive.read(worker_root + filename)
                array = np.load(io.BytesIO(payload), allow_pickle=False)
                require(
                    array.shape == shape and array.dtype == np.float64 and np.isfinite(array).all(),
                    f"Gaussian array drift: {worker}:{filename}",
                )
                key = f"worker_{worker}/{filename}"
                gaussian_hashes[key] = sha256_bytes(payload)
                gaussian_arrays[key] = array

            for name in names_in_partition:
                run_config = runs[name]
                variant = str(run_config["variant"])
                model = LiViFuserPolicy(variant=variant)
                model_state = model.state_dict()
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
                for seed in seeds:
                    identity = (name, seed)
                    member_root = worker_root + f"{name}/seed_{seed}/"
                    result_member = member_root + "result.json"
                    result_payload = archive.read(result_member)
                    result = json.loads(result_payload)
                    result_hashes[f"{name}:{seed}"] = sha256_bytes(result_payload)
                    require(result["run"] == run_config, f"run config drift: {identity}")
                    require(result["seed"] == seed, f"result seed drift: {identity}")

                    training = result["training"]
                    require(training["steps"] == 33000, f"step drift: {identity}")
                    require(
                        training["parameter_count"] == parameter_count,
                        f"parameter count drift: {identity}",
                    )
                    require(
                        math.isfinite(training["seconds"]) and training["seconds"] > 0.0,
                        f"training duration drift: {identity}",
                    )
                    expected_history_steps = [1, *range(100, 33001, 100)]
                    history = training["history"]
                    require(
                        [int(row["step"]) for row in history] == expected_history_steps,
                        f"training history cadence drift: {identity}",
                    )
                    for row in history:
                        step = int(row["step"])
                        expected_phase = (
                            "mean"
                            if run_config["loss"] == "mean_only"
                            or step <= int(config["warmup_steps"])
                            else "heteroscedastic_nll"
                        )
                        require(
                            row["phase"] == expected_phase,
                            f"phase drift: {identity}:{step}",
                        )
                        require(
                            all(
                                math.isfinite(float(row[field]))
                                for field in (
                                    "learning_rate",
                                    "loss",
                                    "gradient_norm_before_clip",
                                )
                            ),
                            f"non-finite history: {identity}:{step}",
                        )

                    validation = result["validation"]
                    require(
                        validation["window_count"] == 9459,
                        f"validation window count drift: {identity}",
                    )
                    per_window = validation["per_window"]
                    require(
                        set(per_window)
                        == {
                            "episode_ids",
                            "origin_rows",
                            "normalized_mse",
                            "nll",
                            "mahalanobis_distance",
                        }
                        and all(len(values) == 9459 for values in per_window.values()),
                        f"per-window contract drift: {identity}",
                    )
                    episode_ids = [str(value) for value in per_window["episode_ids"]]
                    origin_rows = [int(value) for value in per_window["origin_rows"]]
                    distances = [float(value) for value in per_window["mahalanobis_distance"]]
                    mse = np.asarray(per_window["normalized_mse"], dtype=np.float64)
                    nll = np.asarray(per_window["nll"], dtype=np.float64)
                    distance_array = np.asarray(distances, dtype=np.float64)
                    require(
                        np.isfinite(mse).all()
                        and np.isfinite(nll).all()
                        and np.isfinite(distance_array).all()
                        and (mse >= 0.0).all()
                        and (distance_array >= 0.0).all(),
                        f"invalid validation values: {identity}",
                    )
                    expected_counts = context["splits"]["validation"]["episode_window_counts"]
                    require(
                        len(set(episode_ids)) == 30
                        and Counter(episode_ids) == Counter(expected_counts),
                        f"validation episode/window drift: {identity}",
                    )
                    current_identity = (episode_ids, origin_rows)
                    if reference_window_identity is None:
                        reference_window_identity = current_identity
                        reference_distances = distances
                    else:
                        require(
                            current_identity == reference_window_identity,
                            f"validation ordering drift: {identity}",
                        )
                        require(
                            distances == reference_distances,
                            f"Mahalanobis values drift across runs: {identity}",
                        )

                    validate_episode_summary(
                        result_member,
                        episode_ids,
                        nll,
                        validation["per_episode_nll"],
                    )
                    validate_episode_summary(
                        result_member,
                        episode_ids,
                        mse,
                        validation["per_episode_normalized_mse"],
                    )
                    macro_nll = statistics.mean(
                        float(row["mean"]) for row in validation["per_episode_nll"].values()
                    )
                    macro_mse = statistics.mean(
                        float(row["mean"])
                        for row in validation["per_episode_normalized_mse"].values()
                    )
                    require(
                        close(macro_nll, validation["macro_episode_nll"])
                        and close(macro_mse, validation["macro_episode_normalized_mse"]),
                        f"macro episode metric drift: {identity}",
                    )
                    require(
                        len(validation["per_horizon_normalized_mse"]) == 8
                        and all(
                            math.isfinite(float(value)) and float(value) >= 0.0
                            for value in validation["per_horizon_normalized_mse"]
                        ),
                        f"per-horizon metric drift: {identity}",
                    )
                    sigma = validation["sigma_coverage"]
                    require(
                        0.0 <= float(sigma["clamp_floor_fraction"]) <= 1.0,
                        f"sigma clamp fraction drift: {identity}",
                    )
                    for channel in ("linear", "angular"):
                        coverages = [float(sigma[channel][f"{level}_sigma"]) for level in (1, 2, 3)]
                        require(
                            all(0.0 <= value <= 1.0 for value in coverages)
                            and coverages == sorted(coverages),
                            f"sigma coverage drift: {identity}:{channel}",
                        )
                    validate_risk_curve(
                        result_member,
                        mse,
                        distance_array,
                        validation["risk_coverage_by_mahalanobis"],
                    )

                    summary_row = summary_rows[identity]
                    require(
                        summary_row["variant"] == run_config["variant"]
                        and summary_row["loss"] == run_config["loss"]
                        and summary_row["parameter_count"] == parameter_count
                        and close(summary_row["training_seconds"], training["seconds"])
                        and close(summary_row["macro_episode_nll"], macro_nll)
                        and close(summary_row["macro_episode_normalized_mse"], macro_mse),
                        f"combined summary row drift: {identity}",
                    )
                    result_details[identity] = result

                    checkpoint_member = member_root + "checkpoint.pt"
                    checkpoint_payload = archive.read(checkpoint_member)
                    checkpoint_hashes[f"{name}:{seed}"] = sha256_bytes(checkpoint_payload)
                    checkpoint = torch.load(
                        io.BytesIO(checkpoint_payload),
                        map_location="cpu",
                        weights_only=True,
                    )
                    require(
                        set(checkpoint) == {"model_state_dict", "variant", "seed", "config_sha256"},
                        f"checkpoint field drift: {identity}",
                    )
                    require(
                        checkpoint["variant"] == variant
                        and checkpoint["seed"] == seed
                        and checkpoint["config_sha256"] == CONFIG_SHA256,
                        f"checkpoint identity drift: {identity}",
                    )
                    state = checkpoint["model_state_dict"]
                    require(
                        set(state) == set(model_state),
                        f"checkpoint key drift: {identity}",
                    )
                    for key, tensor in state.items():
                        expected_tensor = model_state[key]
                        require(
                            tensor.shape == expected_tensor.shape
                            and tensor.dtype == expected_tensor.dtype
                            and bool(torch.isfinite(tensor).all()),
                            f"checkpoint tensor drift: {identity}:{key}",
                        )

        require(
            contexts[0]["splits"] == contexts[1]["splits"],
            "worker split provenance drift",
        )
        require(
            gaussian_hashes["worker_0/mahalanobis_mean.npy"]
            == gaussian_hashes["worker_1/mahalanobis_mean.npy"]
            and gaussian_hashes["worker_0/mahalanobis_precision.npy"]
            == gaussian_hashes["worker_1/mahalanobis_precision.npy"],
            "worker Gaussian files are not byte-identical",
        )
        precision = gaussian_arrays["worker_0/mahalanobis_precision.npy"]
        require(
            np.allclose(precision, precision.T, rtol=0.0, atol=1e-10),
            "Gaussian precision is not symmetric",
        )
        require(
            float(np.linalg.eigvalsh(precision).min()) > 0.0,
            "Gaussian precision is not positive definite",
        )

        variants = {
            name: summarize_variant([summary_rows[(name, seed)] for seed in seeds])
            for name in sorted(runs)
        }
        full_mse = {
            seed: float(summary_rows[("full", seed)]["macro_episode_normalized_mse"])
            for seed in seeds
        }
        for name, record in variants.items():
            per_seed = {
                int(row["seed"]): float(row["macro_episode_normalized_mse"])
                for row in record["per_seed"]
            }
            record["mse_seed_wins_vs_full"] = sum(per_seed[seed] < full_mse[seed] for seed in seeds)
            details = [result_details[(name, seed)] for seed in seeds]
            record["clamp_floor_fraction_per_seed"] = [
                float(row["validation"]["sigma_coverage"]["clamp_floor_fraction"])
                for row in details
            ]
            record["per_horizon_mse_seed_mean"] = (
                np.asarray(
                    [row["validation"]["per_horizon_normalized_mse"] for row in details],
                    dtype=np.float64,
                )
                .mean(axis=0)
                .tolist()
            )

        common_mahalanobis = contexts[0]["mahalanobis"]
        return {
            "schema_version": 1,
            "status": "PASS",
            "archive": {
                "path": str(archive_path),
                "size_bytes": archive_path.stat().st_size,
                "sha256": archive_sha256,
                "member_count": len(infos),
                "file_count": len(observed_files),
                "directory_count": len(observed_directories),
                "uncompressed_bytes": sum(info.file_size for info in infos),
                "compressed_member_bytes": sum(info.compress_size for info in infos),
            },
            "frozen_provenance": {
                "sweep_id": config["sweep_id"],
                "config_sha256": CONFIG_SHA256,
                "data_plan_sha256": manifest["data_plan_sha256"],
                "summary_sha256": manifest["summary_sha256"],
                "cloud_manifest_sha256": CLOUD_MANIFEST_SHA256,
                "cloud_git_revision": cloud["git_revision"],
                "backbone": config["backbone"],
                "runtime": execution_plan["runtime"],
            },
            "integrity": {
                "exact_member_set": True,
                "duplicates": 0,
                "unsafe_members": 0,
                "zip_crc": "PASS",
                "workers_returned_zero": 2,
                "result_count": len(result_hashes),
                "checkpoint_count": len(checkpoint_hashes),
                "all_result_arithmetic_recomputed": True,
                "all_checkpoint_contracts_verified": True,
                "all_training_histories_verified": True,
                "heldout_excluded": True,
            },
            "validation": {
                "episode_count": 30,
                "window_count": 9459,
                "common_window_identity_across_results": True,
                "common_mahalanobis_across_results": True,
            },
            "mahalanobis": {
                **common_mahalanobis,
                "mean_sha256": gaussian_hashes["worker_0/mahalanobis_mean.npy"],
                "precision_sha256": gaussian_hashes["worker_0/mahalanobis_precision.npy"],
                "worker_files_byte_identical": True,
            },
            "variants": variants,
            "result_sha256": result_hashes,
            "checkpoint_sha256": checkpoint_hashes,
            "member_sha256": member_hashes,
        }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + chr(10), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "livifuser_simulation_sweep_v1_results.zip",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "simulation_sweep_v1.json",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit(args.archive, args.config)
    if args.report is not None:
        report_path = args.report.resolve()
        write_json_atomic(report_path, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "archive_sha256": report["archive"]["sha256"],
                    "results": report["integrity"]["result_count"],
                    "checkpoints": report["integrity"]["checkpoint_count"],
                    "report": str(report_path),
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
