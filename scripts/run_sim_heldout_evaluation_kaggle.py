#!/usr/bin/env python3
"""Run and seal the approved one-time simulation held-out evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import io
import json
import os
import platform
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from audit_simulation_sweep_results import (  # noqa: E402
    CONFIG_SHA256,
    PARTITIONS,
    sha256_bytes,
    sha256_file,
)
from evaluate_sim_heldout import (  # noqa: E402
    RESULT_AUDIT_SHA256,
    SCORE_AUDIT_SHA256,
    SCORE_BUNDLE_SHA256,
    SCORE_MANIFEST_FILE_SHA256,
    SCORE_MANIFEST_SELF_SHA256,
    SEEDS,
    validate_plan,
)
from replay_sim_validation_scores import RESULT_ARCHIVE_SHA256  # noqa: E402

from livifuser_nav.cloud_bundle import verify_cloud_bundle  # noqa: E402
from livifuser_nav.heldout_evaluation import (  # noqa: E402
    discrimination_metrics,
    episode_reduce,
    episode_risk_coverage,
    hierarchical_paired_bootstrap,
    macro_sigma_calibration,
)

AMENDMENT = Path(
    "docs/experiments/PREREGISTRATION_SIM_HELDOUT_EVALUATION_EXECUTION_AMENDMENT_2026-08-24.md"
)
AMENDMENT_SHA256 = "2CD7ADE1AC43FBC74975D9987E6C6052F5146B9FAD4CB97B1D46BEC996F4EE55"
REPAIR = Path(
    "docs/experiments/PREREGISTRATION_SIM_HELDOUT_EVALUATION_EXECUTION_REPAIR_2026-08-24.md"
)
REPAIR_SHA256 = "EB19516B2D84D4830A7A34B7EDB56DFBACE7E8C8E17866AEB9605B9929AC9357"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
GROUPS = (
    ("test_id", "C0"),
    ("test_ood", "C0"),
    ("test_ood", "C1"),
    ("test_ood", "C3b"),
    ("test_ood", "C4"),
)
SCORE_NAMES = ("aleatoric", "mahalanobis", "combined")
COMPARATORS = (
    "lidar_only",
    "rgb_only",
    "concat",
    "no_fov_mask",
    "no_gate",
    "no_temporal",
    "full_mean_only",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def self_hash(payload: dict[str, Any], field: str) -> str:
    value = copy.deepcopy(payload)
    value.pop(field, None)
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def zip_payload(members: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, members[name])
    return output.getvalue()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, indent=2) + chr(10)
    if path.exists():
        require(path.read_text(encoding="utf-8") == raw, f"existing JSON drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    require(not temporary.exists(), f"stale partial JSON: {temporary}")
    temporary.write_text(raw, encoding="utf-8", newline=chr(10))
    temporary.replace(path)


def worker_command(
    data_plan: Path,
    results_source: Path,
    result_audit: Path,
    score_source: Path,
    score_audit: Path,
    config: Path,
    output: Path,
    names: tuple[str, ...],
) -> list[str]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "evaluate_sim_heldout.py"),
        "--data-plan",
        str(data_plan),
        "--results-source",
        str(results_source),
        "--result-audit",
        str(result_audit),
        "--score-source",
        str(score_source),
        "--score-audit",
        str(score_audit),
        "--config",
        str(config),
        "--output",
        str(output),
        "--device",
        "cuda:0",
    ]
    for name in names:
        command.extend(["--run-name", name])
    return command


def run_worker(
    index: int,
    cuda_device: str,
    data_plan: Path,
    results_source: Path,
    result_audit: Path,
    score_source: Path,
    score_audit: Path,
    config: Path,
    output_root: Path,
    print_lock: threading.Lock,
) -> dict[str, Any]:
    names = PARTITIONS[index]
    output = output_root / f"worker_{index}"
    output.mkdir(parents=True, exist_ok=True)
    command = worker_command(
        data_plan,
        results_source,
        result_audit,
        score_source,
        score_audit,
        config,
        output,
        names,
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = cuda_device
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    log_path = output_root / f"worker_{index}.log"
    with log_path.open("a", encoding="utf-8", newline=chr(10)) as log:
        log.write(f"ACCELERATOR physical_cuda:{cuda_device}{chr(10)}")
        log.write(f"COMMAND {subprocess.list2cmdline(command)}{chr(10)}")
        log.flush()
        with print_lock:
            print(
                f"starting worker={index} accelerator=physical_cuda:{cuda_device}",
                flush=True,
            )
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            with print_lock:
                print(f"[worker {index}] {line}", end="", flush=True)
        returncode = process.wait()
    return {
        "worker": index,
        "accelerator": f"physical_cuda:{cuda_device}",
        "run_names": list(names),
        "returncode": returncode,
        "output": str(output),
        "log": str(log_path),
    }


def load_npz(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def group_mask(common: dict[str, np.ndarray], split: str, condition: str) -> np.ndarray:
    return (common["splits"].astype(str) == split) & (common["conditions"].astype(str) == condition)


def group_key(split: str, condition: str) -> str:
    return f"{split}:{condition}"


def episode_values(
    common: dict[str, np.ndarray],
    mask: np.ndarray,
    values: np.ndarray,
    reduction: str,
) -> tuple[np.ndarray, np.ndarray]:
    return episode_reduce(
        common["episode_ids"][mask].astype(str).tolist(),
        np.asarray(values)[mask],
        reduction,
    )


def prediction_metrics(
    common: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    heteroscedastic: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split, condition in GROUPS:
        mask = group_mask(common, split, condition)
        identities, episode_mse = episode_values(common, mask, arrays["normalized_mse"], "mean")
        horizon = []
        episode_horizon: dict[str, list[float]] = {identity: [] for identity in identities.tolist()}
        for index in range(8):
            horizon_ids, values = episode_values(
                common,
                mask,
                arrays["per_horizon_squared_error"][:, index],
                "mean",
            )
            require(
                np.array_equal(identities, horizon_ids),
                "horizon episode identity drift",
            )
            horizon.append(float(values.mean()))
            for identity, value in zip(identities.tolist(), values, strict=True):
                episode_horizon[identity].append(float(value))
        record: dict[str, Any] = {
            "episode_count": int(identities.size),
            "window_count": int(mask.sum()),
            "macro_episode_normalized_mse": float(episode_mse.mean()),
            "per_horizon_macro_episode_normalized_mse": horizon,
            "per_episode_normalized_mse": {
                identity: float(value)
                for identity, value in zip(identities.tolist(), episode_mse, strict=True)
            },
            "per_episode_per_horizon_normalized_mse": episode_horizon,
        }
        if heteroscedastic:
            nll_ids, episode_nll = episode_values(common, mask, arrays["nll"], "mean")
            require(
                np.array_equal(identities, nll_ids),
                "NLL episode identity drift",
            )
            record["macro_episode_nll"] = float(episode_nll.mean())
            record["per_episode_nll"] = {
                identity: float(value)
                for identity, value in zip(identities.tolist(), episode_nll, strict=True)
            }
            record["sigma_calibration"] = macro_sigma_calibration(
                common["episode_ids"][mask].astype(str).tolist(),
                arrays["mean"][mask],
                arrays["log_variance"][mask],
                common["target"][mask],
            )
        output[group_key(split, condition)] = record
    return output


def trivial_metrics(
    common: dict[str, np.ndarray],
    trivial: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, Any]:
    arrays = {
        "normalized_mse": trivial[f"{prefix}_normalized_mse"],
        "per_horizon_squared_error": trivial[f"{prefix}_per_horizon_squared_error"],
    }
    return prediction_metrics(common, arrays, False)


def episode_metadata(
    common: dict[str, np.ndarray],
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    identities = np.asarray(sorted(set(common["episode_ids"][mask].astype(str))), dtype=np.str_)
    worlds = []
    indices = []
    seeds = []
    for identity in identities:
        selected = mask & (common["episode_ids"].astype(str) == identity)
        for field in ("worlds", "episode_indices", "observation_seeds"):
            require(
                np.unique(common[field][selected]).size == 1,
                f"episode metadata drift: {identity}:{field}",
            )
        worlds.append(str(common["worlds"][selected][0]))
        indices.append(int(common["episode_indices"][selected][0]))
        seeds.append(int(common["observation_seeds"][selected][0]))
    return (
        identities,
        np.asarray(worlds, dtype=np.str_),
        np.asarray(indices, dtype=np.int64),
        np.asarray(seeds, dtype=np.int64),
    )


def uncertainty_metrics(
    common: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    prediction: dict[str, Any],
    thresholds: dict[str, float] | None,
) -> dict[str, Any]:
    discrimination: dict[str, Any] = {}
    risk: dict[str, Any] = {}
    intervention: dict[str, Any] = {}
    for split, condition in GROUPS:
        mask = group_mask(common, split, condition)
        identities, _worlds, _indices, _seeds = episode_metadata(common, mask)
        for score_name, field in (
            ("aleatoric", "z_a"),
            ("mahalanobis", "z_m"),
            ("combined", "combined"),
        ):
            score_ids, scores = episode_values(common, mask, arrays[field], "max")
            require(
                np.array_equal(identities, score_ids),
                "score episode identity drift",
            )
            mse_map = prediction[group_key(split, condition)]["per_episode_normalized_mse"]
            episode_risk = np.asarray(
                [mse_map[identity] for identity in identities],
                dtype=np.float64,
            )
            risk[f"{group_key(split, condition)}:{score_name}"] = episode_risk_coverage(
                identities, episode_risk, scores
            )
            if thresholds is not None:
                threshold = float(thresholds[score_name])
                count = int(np.count_nonzero(scores > threshold))
                require(
                    score_name != "combined" or count == 0,
                    "combined gate fired despite frozen threshold",
                )
                intervention[f"{group_key(split, condition)}:{score_name}"] = {
                    "episode_count": int(identities.size),
                    "threshold": threshold,
                    "comparison": "strict_greater_than",
                    "intervention_count": count,
                    "intervention_rate": float(count / identities.size),
                }

    negative_mask = group_mask(common, "test_ood", "C0")
    (
        negative_ids,
        negative_worlds,
        negative_indices,
        negative_seeds,
    ) = episode_metadata(common, negative_mask)
    negative_keys = list(
        zip(
            negative_worlds.tolist(),
            negative_indices.tolist(),
            negative_seeds.tolist(),
            strict=True,
        )
    )
    require(len(set(negative_keys)) == 20, "negative matching key drift")
    for condition in ("C1", "C3b", "C4"):
        positive_mask = group_mask(common, "test_ood", condition)
        (
            positive_ids,
            positive_worlds,
            positive_indices,
            positive_seeds,
        ) = episode_metadata(common, positive_mask)
        positive_keys = list(
            zip(
                positive_worlds.tolist(),
                positive_indices.tolist(),
                positive_seeds.tolist(),
                strict=True,
            )
        )
        require(
            set(positive_keys) == set(negative_keys),
            f"matched condition key drift: {condition}",
        )
        negative_order = {key: index for index, key in enumerate(negative_keys)}
        aligned_negative = np.asarray(
            [negative_order[key] for key in positive_keys], dtype=np.int64
        )
        for score_name in SCORE_NAMES:
            score_field = {
                "aleatoric": "z_a",
                "mahalanobis": "z_m",
                "combined": "combined",
            }[score_name]
            positive_score_ids, positive_scores = episode_values(
                common, positive_mask, arrays[score_field], "max"
            )
            negative_score_ids, negative_scores = episode_values(
                common, negative_mask, arrays[score_field], "max"
            )
            require(
                np.array_equal(positive_ids, positive_score_ids)
                and np.array_equal(negative_ids, negative_score_ids),
                "discrimination episode ordering drift",
            )
            negative_scores = negative_scores[aligned_negative]
            record = discrimination_metrics(positive_scores, negative_scores)
            record["matching"] = "world_name+episode_index+observation_seed"
            record["per_world"] = {}
            for world in sorted(set(positive_worlds.tolist())):
                selected = positive_worlds == world
                require(
                    int(selected.sum()) == 10,
                    f"per-world discrimination count drift: {world}",
                )
                record["per_world"][world] = discrimination_metrics(
                    positive_scores[selected], negative_scores[selected]
                )
            discrimination[f"{condition}:{score_name}"] = record
    return {
        "condition_discrimination": discrimination,
        "risk_coverage": risk,
        "interventions": intervention,
    }


def collect_outputs(
    output_root: Path,
    workers: list[dict[str, Any]],
) -> tuple[
    dict[str, bytes],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    members: dict[str, bytes] = {}
    records: list[dict[str, Any]] = []
    common_payload: bytes | None = None
    trivial_payload: bytes | None = None
    for worker in sorted(workers, key=lambda row: int(row["worker"])):
        require(
            worker["returncode"] == 0,
            f"held-out worker failed: {worker}",
        )
        root = output_root / f"worker_{worker['worker']}"
        summary = json.loads((root / "worker_summary.json").read_text(encoding="utf-8"))
        require(
            summary["run_names"] == list(PARTITIONS[int(worker["worker"])]),
            "worker partition drift",
        )
        current_common = (root / "worker_common.npz").read_bytes()
        current_trivial = (root / "worker_trivial_baselines.npz").read_bytes()
        if common_payload is None:
            common_payload = current_common
            trivial_payload = current_trivial
        else:
            require(
                common_payload == current_common,
                "worker common array byte drift",
            )
            require(
                trivial_payload == current_trivial,
                "worker trivial array byte drift",
            )
        for record in summary["records"]:
            source = root / record["prediction_file"]
            payload = source.read_bytes()
            require(
                len(payload) == int(record["prediction_size_bytes"])
                and sha256_bytes(payload) == record["prediction_sha256"],
                f"prediction member drift: {record['name']}:{record['seed']}",
            )
            name = str(record["prediction_file"])
            require(
                name not in members,
                f"duplicate prediction member: {name}",
            )
            members[name] = payload
            records.append(record)
    require(
        common_payload is not None and trivial_payload is not None,
        "missing common outputs",
    )
    identities = {(row["name"], int(row["seed"])) for row in records}
    expected = {(name, seed) for names in PARTITIONS.values() for name in names for seed in SEEDS}
    require(
        len(records) == len(members) == 24 and identities == expected,
        "24 prediction identity drift",
    )
    members["heldout_common.npz"] = common_payload
    members["trivial_baselines.npz"] = trivial_payload
    return (
        members,
        records,
        load_npz(common_payload),
        load_npz(trivial_payload),
    )


def build_summary(
    prediction_members: dict[str, bytes],
    records: list[dict[str, Any]],
    common: dict[str, np.ndarray],
    trivial: dict[str, np.ndarray],
) -> dict[str, Any]:
    require(
        common["episode_ids"].shape == (34_503,) and common["target"].shape == (34_503, 8, 2),
        "common output shape drift",
    )
    by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        identity = (str(record["name"]), int(record["seed"]))
        arrays = load_npz(prediction_members[record["prediction_file"]])
        heteroscedastic = record["loss"] == "heteroscedastic"
        metrics = prediction_metrics(common, arrays, heteroscedastic)
        uncertainty = (
            uncertainty_metrics(common, arrays, metrics, record["thresholds"])
            if heteroscedastic
            else None
        )
        by_identity[identity] = {
            "name": identity[0],
            "seed": identity[1],
            "loss": record["loss"],
            "prediction": metrics,
            "uncertainty": uncertainty,
        }
    trivial_summary = {
        "repeat_last": trivial_metrics(common, trivial, "repeat_last"),
        "constant_training_mean": trivial_metrics(common, trivial, "constant_training_mean"),
    }
    contrasts: dict[str, Any] = {}
    for split, condition in GROUPS:
        key = group_key(split, condition)
        mask = group_mask(common, split, condition)
        (
            identities,
            worlds,
            _indices,
            _observation_seeds,
        ) = episode_metadata(common, mask)
        full_seed_values = []
        for seed in SEEDS:
            values = by_identity[("full", seed)]["prediction"][key]["per_episode_normalized_mse"]
            full_seed_values.append(
                np.asarray(
                    [values[identity] for identity in identities],
                    dtype=np.float64,
                )
            )
        full_mean = np.mean(full_seed_values, axis=0)
        for comparator in COMPARATORS:
            comparator_seed_values = []
            per_seed = {}
            for seed, full_values in zip(SEEDS, full_seed_values, strict=True):
                values = by_identity[(comparator, seed)]["prediction"][key][
                    "per_episode_normalized_mse"
                ]
                comparator_values = np.asarray(
                    [values[identity] for identity in identities],
                    dtype=np.float64,
                )
                comparator_seed_values.append(comparator_values)
                difference = full_values - comparator_values
                per_seed[str(seed)] = {
                    "mean_full_minus_comparator": float(difference.mean()),
                    "world_mean_full_minus_comparator": {
                        world: float(difference[worlds == world].mean())
                        for world in sorted(set(worlds.tolist()))
                    },
                }
            comparator_mean = np.mean(comparator_seed_values, axis=0)
            contrast = hierarchical_paired_bootstrap(
                worlds, identities, full_mean - comparator_mean
            )
            contrast["definition"] = "three-seed-mean full MSE minus three-seed-mean comparator MSE"
            contrast["per_seed"] = per_seed
            contrasts[f"{key}:full_minus_{comparator}"] = contrast
        for baseline, field in (
            ("repeat_last", "repeat_last_normalized_mse"),
            (
                "constant_training_mean",
                "constant_training_mean_normalized_mse",
            ),
        ):
            baseline_ids, baseline_values = episode_values(common, mask, trivial[field], "mean")
            require(
                np.array_equal(identities, baseline_ids),
                "trivial contrast identity drift",
            )
            contrast = hierarchical_paired_bootstrap(
                worlds, identities, full_mean - baseline_values
            )
            contrast["definition"] = "three-seed-mean full MSE minus trivial baseline MSE"
            contrast["per_seed"] = {
                str(seed): {
                    "mean_full_minus_comparator": float((full_values - baseline_values).mean())
                }
                for seed, full_values in zip(SEEDS, full_seed_values, strict=True)
            }
            contrasts[f"{key}:full_minus_{baseline}"] = contrast
    ordered_results = [
        by_identity[(name, seed)]
        for name in sorted({row["name"] for row in records})
        for seed in SEEDS
    ]
    return {
        "schema_version": 1,
        "status": "SEALED_HELDOUT_OFFLINE_EVALUATION",
        "analysis_unit": "episode",
        "generalization_unit": "world",
        "training_seed_role": "nuisance_factor",
        "groups": [group_key(*group) for group in GROUPS],
        "records": ordered_results,
        "trivial_baselines": trivial_summary,
        "paired_mse_contrasts": contrasts,
        "limitations": [
            (
                "Only two independent worlds are available; cluster "
                "intervals and sign tests are weak."
            ),
            ("The frozen combined threshold is 1.0 and cannot trigger under strict greater-than."),
        ],
    }


def seal_bundle(
    bundle_output: Path,
    members: dict[str, bytes],
    records: list[dict[str, Any]],
    plan_path: Path,
    plan: dict[str, Any],
    result_audit_path: Path,
    score_audit_path: Path,
    cloud: dict[str, Any],
    workers: list[dict[str, Any]],
) -> dict[str, Any]:
    member_rows = [
        {
            "name": name,
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
        for name, payload in sorted(members.items())
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "SEALED_ONE_TIME_HELDOUT_OFFLINE_EVALUATION",
        "amendment": {
            "path": AMENDMENT.as_posix(),
            "sha256": AMENDMENT_SHA256,
            "approval_date": "2026-08-24",
        },
        "execution_repair": {
            "path": REPAIR.as_posix(),
            "sha256": REPAIR_SHA256,
            "recorded_date": "2026-08-24",
            "pre_model_inference": True,
        },
        "identities": {
            "source_handoff_self_sha256": plan["source_handoff_self_sha256"],
            "cache_transport_sha256": plan["cache_transport_sha256"],
            "cache_manifest_sha256": plan["cache_manifest_sha256"],
            "cache_manifest_self_sha256": plan["cache_manifest_self_sha256"],
            "backbone_contract_sha256": plan["backbone_contract_sha256"],
            "result_archive_sha256": RESULT_ARCHIVE_SHA256,
            "result_audit_sha256": sha256_file(result_audit_path),
            "mahalanobis_mean_sha256": (
                "7CA0C5DE3EA1A7A4C2F29142F85DF4BEA441D270888D744A89FCE2300FC853B5"
            ),
            "mahalanobis_precision_sha256": (
                "3E69CD96AE8A69E66EF8223F54DBC3261A46AC61C58A745DEE7C3B7DB3B73A2B"
            ),
            "score_freeze_archive_sha256": SCORE_BUNDLE_SHA256,
            "score_freeze_manifest_file_sha256": (SCORE_MANIFEST_FILE_SHA256),
            "score_freeze_manifest_self_sha256": (SCORE_MANIFEST_SELF_SHA256),
            "score_freeze_audit_sha256": sha256_file(score_audit_path),
            "simulation_config_sha256": CONFIG_SHA256,
            "heldout_data_plan_sha256": sha256_file(plan_path),
            "heldout_code_cloud_manifest_sha256": cloud["manifest_sha256"].upper(),
            "cloud_git_revision": cloud["git_revision"],
        },
        "counts": {
            "episodes": 110,
            "accepted_samples": 47_326,
            "windows": 34_503,
            "prediction_records": 24,
            "heteroscedastic_records": 21,
            "closed_loop_threshold_records": 12,
            "trivial_baselines": 2,
            "exact_bundle_members": 29,
        },
        "metric_contract": {
            "episode_reduction": ("window values reduced within episode first"),
            "discrimination": ("matched test_ood C0 negative; episode maximum score"),
            "risk_coverage": ("episode risk; requested 1.00 to 0.10 by 0.05; ceil count"),
            "intervention": ("strict score > frozen validation threshold"),
            "bootstrap": ("world then paired episodes; 10000; seed 20260824; linear percentile"),
            "c3_machine_label": "C3b",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "workers": [
                {
                    "worker": row["worker"],
                    "accelerator": row["accelerator"],
                    "run_names": row["run_names"],
                }
                for row in sorted(workers, key=lambda value: int(value["worker"]))
            ],
        },
        "prediction_records": sorted(
            records,
            key=lambda row: (str(row["name"]), int(row["seed"])),
        ),
        "members": member_rows,
    }
    field = "manifest_sha256_excludes_self"
    manifest[field] = self_hash(manifest, field)
    manifest_payload = (json.dumps(manifest, indent=2) + chr(10)).encode()
    completion = {
        "schema_version": 1,
        "status": "COMPLETE",
        "manifest_file_sha256": sha256_bytes(manifest_payload),
        "manifest_sha256_excludes_self": manifest[field],
        "prediction_record_count": 24,
        "heteroscedastic_record_count": 21,
        "closed_loop_threshold_record_count": 12,
        "exact_bundle_member_count": 29,
    }
    bundle_members = dict(members)
    bundle_members["HELDOUT_EVAL_MANIFEST.json"] = manifest_payload
    bundle_members["HELDOUT_EVAL_COMPLETE.json"] = (
        json.dumps(completion, indent=2) + chr(10)
    ).encode()
    require(len(bundle_members) == 29, "sealed member count drift")
    payload = zip_payload(bundle_members)
    bundle_output.parent.mkdir(parents=True, exist_ok=True)
    if bundle_output.exists():
        require(
            bundle_output.read_bytes() == payload,
            "existing held-out bundle drift",
        )
    else:
        temporary = bundle_output.with_suffix(bundle_output.suffix + ".tmp")
        require(
            not temporary.exists(),
            f"stale bundle partial: {temporary}",
        )
        temporary.write_bytes(payload)
        temporary.replace(bundle_output)
    return {
        "bundle": str(bundle_output),
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "member_count": len(bundle_members),
        "manifest_file_sha256": sha256_bytes(manifest_payload),
        "manifest_self_sha256": manifest[field],
        "prediction_records": 24,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-plan", type=Path, required=True)
    parser.add_argument("--results-source", type=Path, required=True)
    parser.add_argument("--result-audit", type=Path, required=True)
    parser.add_argument("--score-source", type=Path, required=True)
    parser.add_argument("--score-audit", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/livifuser_sim_heldout_evaluation_v1"),
    )
    parser.add_argument(
        "--bundle-output",
        type=Path,
        default=Path("/kaggle/working/livifuser_sim_heldout_evaluation_v1_bundle.zip"),
    )
    parser.add_argument("--cuda-device", action="append", required=True)
    args = parser.parse_args()
    require(
        sha256_file(REPOSITORY_ROOT / AMENDMENT) == AMENDMENT_SHA256,
        "amendment hash drift",
    )
    require(
        sha256_file(REPOSITORY_ROOT / REPAIR) == REPAIR_SHA256,
        "execution repair hash drift",
    )
    require(
        sha256_file(args.config.resolve()) == CONFIG_SHA256,
        "config hash drift",
    )
    require(
        sha256_file(args.result_audit.resolve()) == RESULT_AUDIT_SHA256,
        "result audit hash drift",
    )
    require(
        sha256_file(args.score_audit.resolve()) == SCORE_AUDIT_SHA256,
        "score audit hash drift",
    )
    plan_path = args.data_plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    cloud = verify_cloud_bundle(REPOSITORY_ROOT)
    devices = [str(value) for value in args.cuda_device]
    require(
        len(devices) == 2 and len(set(devices)) == 2,
        "exactly two unique CUDA devices required",
    )
    require(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "Kaggle T4x2 unavailable",
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    execution = {
        "schema_version": 1,
        "amendment_sha256": AMENDMENT_SHA256,
        "execution_repair_sha256": REPAIR_SHA256,
        "data_plan_sha256": sha256_file(plan_path),
        "result_archive_sha256": RESULT_ARCHIVE_SHA256,
        "score_archive_sha256": SCORE_BUNDLE_SHA256,
        "partitions": {str(key): list(value) for key, value in PARTITIONS.items()},
        "cuda_devices": devices,
        "cloud_bundle": cloud,
    }
    write_json_atomic(output_root / "execution_plan.json", execution)
    print_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                run_worker,
                index,
                devices[index],
                plan_path,
                args.results_source.resolve(),
                args.result_audit.resolve(),
                args.score_source.resolve(),
                args.score_audit.resolve(),
                args.config.resolve(),
                output_root,
                print_lock,
            )
            for index in range(2)
        ]
        workers = [future.result() for future in futures]
    write_json_atomic(
        output_root / "execution_summary.json",
        {"workers": workers},
    )
    members, records, common, trivial = collect_outputs(output_root, workers)
    summary = build_summary(members, records, common, trivial)
    members["summary.json"] = (json.dumps(summary, indent=2) + chr(10)).encode()
    report = seal_bundle(
        args.bundle_output.resolve(),
        members,
        records,
        plan_path,
        plan,
        args.result_audit.resolve(),
        args.score_audit.resolve(),
        cloud,
        workers,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
