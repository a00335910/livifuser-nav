#!/usr/bin/env python3
"""Independently audit and numerically reproduce the sealed held-out evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from run_sim_heldout_evaluation_kaggle import build_summary

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_SIZE_BYTES = 163_135_634
BUNDLE_SHA256 = "EC4736E677D7365461FE1C1F9C607A699D36E1AC9624F8B89D46D864DB8C75C6"
MANIFEST_FILE_SHA256 = "69801277F2DC63468C2691D63D5618AB13AF593B2D9EF9570CA1C239C8EE7A96"
MANIFEST_SELF_SHA256 = "A853AF0A8352AB54DE8D7429688F8C3E9ABAB262179197AECF9FD00EA2C244D5"
AMENDMENT_SHA256 = "2CD7ADE1AC43FBC74975D9987E6C6052F5146B9FAD4CB97B1D46BEC996F4EE55"
REPAIR_SHA256 = "EB19516B2D84D4830A7A34B7EDB56DFBACE7E8C8E17866AEB9605B9929AC9357"
HANDOFF_SELF_SHA256 = "3F48A7E54A1596947A469B59B8D63EE96FD29294C7BD57EFAC924736A984492C"
CACHE_BUNDLE_SHA256 = "7FB323948427AB6FC1F5F82F2CEF5E66DDB51F056C031132B2EE8C9B9F0484E5"
CACHE_MANIFEST_SHA256 = "6E7E51176FE494634303D756BDCF8D9BB5D28C81754CC26F6C11180BFCB1FD42"
CACHE_SELF_SHA256 = "9FC559291790B5FDB1E77F62FD1A160C4498BFD4591B86B420BEC8C89BC4A5F7"
BACKBONE_SHA256 = "2957C78346DE608067DD5AC14D5C3E2F23438CD2BB3B1ECA847F898EBA68894A"
RESULT_BUNDLE_SHA256 = "F5B7D9EAB29DD20CE6710E4B803EAA331A5D7C2E741E9330995A1EAE615B9AC7"
RESULT_AUDIT_SHA256 = "4D1CEA8F2D61EF76E1A48770FB6228F14683DAF6943C4932C06FCE0FB46611B3"
SCORE_BUNDLE_SHA256 = "07116A629E296929D69EDA41E44CB6067CB6C751C735B66FD0A1B736D240751B"
SCORE_MANIFEST_FILE_SHA256 = "BFD5A21F150DCFCF12CD988821DE6901A1558ACC0EA183D7F8223940C2C1A729"
SCORE_MANIFEST_SELF_SHA256 = "AA90B540579C8285F55422DB41EA549305FA6C755FDF391FA3CD06ADC82127BF"
SCORE_AUDIT_SHA256 = "A9071ABF41F25B0FA68209B2AFB94242F33B83E4330DF6B9B563A5FA6ADA3E97"
CONFIG_SHA256 = "76680BCE45A67D5D91F42660D5EC25F90450B281838CE311E329477C4E36F09E"
CODE_MANIFEST_SHA256 = "9E0A6F5176F290F46AC732575459053F5A7E95A8E8A2F53E67F6281B03517F74"
DATA_PLAN_SHA256 = "DFA6A2BDF0776D0DD5ED582932CE6CB7DE6DA6B72E36BCEF591760A4C6B42499"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
SEEDS = (20_260_805, 20_260_806, 20_260_807)
NAMES = (
    "concat",
    "full",
    "full_mean_only",
    "lidar_only",
    "no_fov_mask",
    "no_gate",
    "no_temporal",
    "rgb_only",
)
HETEROSCEDASTIC_NAMES = tuple(name for name in NAMES if name != "full_mean_only")
CLOSED_LOOP_NAMES = ("full", "lidar_only", "concat", "rgb_only")
GROUPS = (
    ("test_id", "C0", 30),
    ("test_ood", "C0", 20),
    ("test_ood", "C1", 20),
    ("test_ood", "C3b", 20),
    ("test_ood", "C4", 20),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_self_hash(payload: dict[str, Any], field: str) -> str:
    value = copy.deepcopy(payload)
    value.pop(field, None)
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in name


def load_npz(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, indent=2) + chr(10)
    if path.exists():
        require(path.read_text(encoding="utf-8") == raw, f"existing audit drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    require(not temporary.exists(), f"stale audit partial: {temporary}")
    temporary.write_text(raw, encoding="utf-8", newline=chr(10))
    temporary.replace(path)


def expected_names() -> set[str]:
    predictions = {f"predictions/{name}_seed_{seed}.npz" for name in NAMES for seed in SEEDS}
    return predictions | {
        "HELDOUT_EVAL_COMPLETE.json",
        "HELDOUT_EVAL_MANIFEST.json",
        "heldout_common.npz",
        "summary.json",
        "trivial_baselines.npz",
    }


def reconstruct_data_plan(handoff: dict[str, Any]) -> str:
    work_root = "/kaggle/working/livifuser_sim_heldout_data_v1"
    episodes = []
    for episode in sorted(handoff["episodes"], key=lambda row: int(row["ordinal"])):
        identity = str(episode["episode_id"])
        episodes.append(
            {
                "episode_id": identity,
                "split": episode["split"],
                "world_name": episode["world_name"],
                "condition": "C3b" if episode["condition"] == "C3" else episode["condition"],
                "episode_index": int(episode["episode_index"]),
                "observation_seed": int(episode["observation_seed"]),
                "ordinal": int(episode["ordinal"]),
                "accepted_samples": int(episode["accepted_samples"]),
                "windows_k8_h8": int(episode["windowable_k8_h8"]),
                "export": f"{work_root}/exports/{identity}",
                "cache": f"{work_root}/caches/{identity}",
            }
        )
    plan = {
        "schema_version": 1,
        "purpose": "approved_one_time_simulation_heldout_evaluation",
        "source_handoff_self_sha256": HANDOFF_SELF_SHA256,
        "cache_transport_sha256": CACHE_BUNDLE_SHA256,
        "cache_manifest_sha256": CACHE_MANIFEST_SHA256,
        "cache_manifest_self_sha256": CACHE_SELF_SHA256,
        "backbone_contract_sha256": BACKBONE_SHA256,
        "heldout_attached": True,
        "allowed_splits": ["test_id", "test_ood"],
        "work_root": work_root,
        "episode_count": len(episodes),
        "accepted_samples": sum(row["accepted_samples"] for row in episodes),
        "windows_k8_h8": sum(row["windows_k8_h8"] for row in episodes),
        "episodes": episodes,
    }
    return sha256_bytes((json.dumps(plan, indent=2) + chr(10)).encode())


def max_difference(left: np.ndarray, right: np.ndarray, label: str) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    require(a.shape == b.shape, f"shape drift: {label}")
    require(np.all(np.isfinite(a)) and np.all(np.isfinite(b)), f"non-finite values: {label}")
    difference = float(np.max(np.abs(a - b))) if a.size else 0.0
    require(difference <= 1e-12, f"numerical drift: {label}: {difference}")
    return difference


def verify_upstream(manifest: dict[str, Any]) -> dict[str, Any]:
    amendment = REPOSITORY_ROOT / manifest["amendment"]["path"]
    repair = REPOSITORY_ROOT / manifest["execution_repair"]["path"]
    result_bundle = REPOSITORY_ROOT / "artifacts" / "livifuser_simulation_sweep_v1_results.zip"
    result_audit_path = REPOSITORY_ROOT / "artifacts" / "simulation_sweep_v1_result_audit.json"
    score_bundle = (
        REPOSITORY_ROOT / "artifacts" / "livifuser_sim_validation_score_freeze_v1_bundle.zip"
    )
    score_audit_path = REPOSITORY_ROOT / "artifacts" / "sim_validation_score_freeze_v1_audit.json"
    config = REPOSITORY_ROOT / "config" / "simulation_sweep_v1.json"
    code_bundle = (
        REPOSITORY_ROOT
        / "artifacts"
        / "cloud"
        / "livifuser_sim_heldout_eval_code_2cd7ade1ac43_r3.zip"
    )
    handoff_path = (
        REPOSITORY_ROOT
        / "artifacts"
        / "simulation"
        / "gpu_handoff_heldout_v1"
        / "handoff_manifest.json"
    )
    cache_bundle = (
        REPOSITORY_ROOT / "artifacts" / "livifuser_dinov3_splus_heldout_cache_v1_bundle.zip"
    )
    for path in (
        amendment,
        repair,
        result_bundle,
        result_audit_path,
        score_bundle,
        score_audit_path,
        config,
        code_bundle,
        handoff_path,
        cache_bundle,
    ):
        require(path.is_file(), f"missing upstream artifact: {path}")
    expected_files = {
        amendment: AMENDMENT_SHA256,
        repair: REPAIR_SHA256,
        result_bundle: RESULT_BUNDLE_SHA256,
        result_audit_path: RESULT_AUDIT_SHA256,
        score_bundle: SCORE_BUNDLE_SHA256,
        score_audit_path: SCORE_AUDIT_SHA256,
        config: CONFIG_SHA256,
        cache_bundle: CACHE_BUNDLE_SHA256,
    }
    for path, expected in expected_files.items():
        require(sha256_file(path) == expected, f"upstream hash drift: {path}")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    require(
        handoff["manifest_sha256_excludes_self"] == HANDOFF_SELF_SHA256
        and canonical_self_hash(handoff, "manifest_sha256_excludes_self") == HANDOFF_SELF_SHA256,
        "held-out handoff self-hash drift",
    )
    require(reconstruct_data_plan(handoff) == DATA_PLAN_SHA256, "held-out data plan hash drift")
    with zipfile.ZipFile(cache_bundle) as cache:
        cache_manifest_raw = cache.read("cache_manifest.json")
        cache_manifest = json.loads(cache_manifest_raw)
        require(
            sha256_bytes(cache_manifest_raw) == CACHE_MANIFEST_SHA256, "cache manifest hash drift"
        )
        require(
            cache_manifest["manifest_sha256_excludes_self"] == CACHE_SELF_SHA256
            and canonical_self_hash(cache_manifest, "manifest_sha256_excludes_self")
            == CACHE_SELF_SHA256,
            "cache manifest self-hash drift",
        )
        require(
            sha256_bytes(cache.read("BACKBONE_CONTRACT.json")) == BACKBONE_SHA256,
            "backbone contract hash drift",
        )
        completion = json.loads(cache.read("CACHE_COMPLETE.json"))
        require(
            completion["cache_manifest_sha256"] == CACHE_MANIFEST_SHA256,
            "cache completion marker drift",
        )
    with zipfile.ZipFile(code_bundle) as code:
        require(
            sha256_bytes(code.read("LiViFuser/cloud_bundle_manifest.json")) == CODE_MANIFEST_SHA256,
            "code cloud manifest hash drift",
        )
    identities = manifest["identities"]
    expected_identities = {
        "source_handoff_self_sha256": HANDOFF_SELF_SHA256,
        "cache_transport_sha256": CACHE_BUNDLE_SHA256,
        "cache_manifest_sha256": CACHE_MANIFEST_SHA256,
        "cache_manifest_self_sha256": CACHE_SELF_SHA256,
        "backbone_contract_sha256": BACKBONE_SHA256,
        "result_archive_sha256": RESULT_BUNDLE_SHA256,
        "result_audit_sha256": RESULT_AUDIT_SHA256,
        "score_freeze_archive_sha256": SCORE_BUNDLE_SHA256,
        "score_freeze_manifest_file_sha256": SCORE_MANIFEST_FILE_SHA256,
        "score_freeze_manifest_self_sha256": SCORE_MANIFEST_SELF_SHA256,
        "score_freeze_audit_sha256": SCORE_AUDIT_SHA256,
        "simulation_config_sha256": CONFIG_SHA256,
        "heldout_data_plan_sha256": DATA_PLAN_SHA256,
        "heldout_code_cloud_manifest_sha256": CODE_MANIFEST_SHA256,
    }
    for key, expected in expected_identities.items():
        require(identities[key] == expected, f"manifest upstream identity drift: {key}")
    return {
        "all_local_hashes_verified": True,
        "heldout_data_plan_reconstructed": True,
        "heldout_cache_transport_rehashed": True,
        "heldout_cache_manifest_and_completion_verified": True,
        "backbone_contract_verified": True,
    }


def audit(bundle_path: Path) -> dict[str, Any]:
    require(bundle_path.stat().st_size == BUNDLE_SIZE_BYTES, "bundle size drift")
    require(sha256_file(bundle_path) == BUNDLE_SHA256, "bundle SHA-256 drift")
    with zipfile.ZipFile(bundle_path) as bundle:
        infos = [info for info in bundle.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)) == 29, "bundle member count or duplicate drift")
        require(set(names) == expected_names(), "bundle exact member set drift")
        require(all(safe_member(name) for name in names), "unsafe bundle member")
        require(bundle.testzip() is None, "bundle CRC failure")
        require(
            all(
                info.date_time == FIXED_ZIP_TIME
                and info.compress_type == zipfile.ZIP_DEFLATED
                and (info.external_attr >> 16) == 0o100644
                for info in infos
            ),
            "deterministic ZIP metadata drift",
        )
        manifest_raw = bundle.read("HELDOUT_EVAL_MANIFEST.json")
        completion_raw = bundle.read("HELDOUT_EVAL_COMPLETE.json")
        require(sha256_bytes(manifest_raw) == MANIFEST_FILE_SHA256, "manifest file hash drift")
        manifest = json.loads(manifest_raw)
        completion = json.loads(completion_raw)
        require(
            manifest["manifest_sha256_excludes_self"] == MANIFEST_SELF_SHA256
            and canonical_self_hash(manifest, "manifest_sha256_excludes_self")
            == MANIFEST_SELF_SHA256,
            "manifest self-hash drift",
        )
        require(
            completion
            == {
                "schema_version": 1,
                "status": "COMPLETE",
                "manifest_file_sha256": MANIFEST_FILE_SHA256,
                "manifest_sha256_excludes_self": MANIFEST_SELF_SHA256,
                "prediction_record_count": 24,
                "heteroscedastic_record_count": 21,
                "closed_loop_threshold_record_count": 12,
                "exact_bundle_member_count": 29,
            },
            "completion marker drift",
        )
        require(
            manifest["status"] == "SEALED_ONE_TIME_HELDOUT_OFFLINE_EVALUATION"
            and manifest["amendment"]["sha256"] == AMENDMENT_SHA256
            and manifest["execution_repair"]["sha256"] == REPAIR_SHA256
            and manifest["execution_repair"]["pre_model_inference"] is True,
            "manifest execution contract drift",
        )
        require(
            manifest["counts"]
            == {
                "episodes": 110,
                "accepted_samples": 47_326,
                "windows": 34_503,
                "prediction_records": 24,
                "heteroscedastic_records": 21,
                "closed_loop_threshold_records": 12,
                "trivial_baselines": 2,
                "exact_bundle_members": 29,
            },
            "manifest counts drift",
        )
        member_rows = manifest["members"]
        require(
            len(member_rows) == 27
            and [row["name"] for row in member_rows]
            == sorted(
                expected_names() - {"HELDOUT_EVAL_MANIFEST.json", "HELDOUT_EVAL_COMPLETE.json"}
            ),
            "manifest member rows drift",
        )
        for row in member_rows:
            payload = bundle.read(row["name"])
            require(
                len(payload) == int(row["size_bytes"]) and sha256_bytes(payload) == row["sha256"],
                f"member hash drift: {row['name']}",
            )

        upstream = verify_upstream(manifest)
        result_audit = json.loads(
            (REPOSITORY_ROOT / "artifacts" / "simulation_sweep_v1_result_audit.json").read_text(
                encoding="utf-8"
            )
        )
        score_path = (
            REPOSITORY_ROOT / "artifacts" / "livifuser_sim_validation_score_freeze_v1_bundle.zip"
        )
        with zipfile.ZipFile(score_path) as score_bundle:
            score_manifest_raw = score_bundle.read("SCORE_FREEZE_MANIFEST.json")
            require(
                sha256_bytes(score_manifest_raw) == SCORE_MANIFEST_FILE_SHA256,
                "score manifest file hash drift",
            )
            score_manifest = json.loads(score_manifest_raw)
            score_records = {
                (str(row["name"]), int(row["seed"])): row for row in score_manifest["records"]
            }
            records = manifest["prediction_records"]
            expected_identities = {(name, seed) for name in NAMES for seed in SEEDS}
            require(
                {(str(row["name"]), int(row["seed"])) for row in records} == expected_identities,
                "prediction identity set drift",
            )
            common = load_npz(bundle.read("heldout_common.npz"))
            expected_common = {
                "conditions",
                "episode_ids",
                "episode_indices",
                "mahalanobis_distance",
                "observation_seeds",
                "origin_rows",
                "splits",
                "target",
                "worlds",
            }
            require(set(common) == expected_common, "heldout common schema drift")
            require(
                common["target"].shape == (34_503, 8, 2)
                and all(common[key].shape == (34_503,) for key in expected_common - {"target"}),
                "heldout common shape drift",
            )
            require(
                all(
                    np.all(np.isfinite(value))
                    for value in common.values()
                    if value.dtype.kind in "fiu"
                ),
                "non-finite common array",
            )
            episode_ids = common["episode_ids"].astype(str)
            require(len(set(episode_ids.tolist())) == 110, "held-out episode identity count drift")
            group_counts: dict[str, dict[str, int]] = {}
            for split, condition, episode_count in GROUPS:
                mask = (common["splits"].astype(str) == split) & (
                    common["conditions"].astype(str) == condition
                )
                require(
                    len(set(episode_ids[mask].tolist())) == episode_count,
                    f"group episode count drift: {split}:{condition}",
                )
                group_counts[f"{split}:{condition}"] = {
                    "episodes": episode_count,
                    "windows": int(mask.sum()),
                }

            trivial = load_npz(bundle.read("trivial_baselines.npz"))
            expected_trivial = {
                "constant_training_mean",
                "constant_training_mean_normalized_mse",
                "constant_training_mean_per_horizon_squared_error",
                "repeat_last_mean",
                "repeat_last_normalized_mse",
                "repeat_last_per_horizon_squared_error",
            }
            require(set(trivial) == expected_trivial, "trivial baseline schema drift")
            scale = np.asarray((0.10, 0.50), dtype=np.float64)
            numerical_max = 0.0
            for prefix, mean_field in (
                ("repeat_last", "repeat_last_mean"),
                ("constant_training_mean", "constant_training_mean"),
            ):
                mean = trivial[mean_field]
                require(mean.shape == (34_503, 8, 2), f"trivial mean shape drift: {prefix}")
                squared = np.square((mean - common["target"]) / scale)
                numerical_max = max(
                    numerical_max,
                    max_difference(
                        squared.mean(axis=(1, 2)),
                        trivial[f"{prefix}_normalized_mse"],
                        f"{prefix} MSE",
                    ),
                    max_difference(
                        squared.mean(axis=2),
                        trivial[f"{prefix}_per_horizon_squared_error"],
                        f"{prefix} horizon MSE",
                    ),
                )
            frozen_mean = np.asarray(
                (0.047447296062892504, -0.005205255475722954), dtype=np.float64
            )
            require(
                np.array_equal(
                    trivial["constant_training_mean"], np.broadcast_to(frozen_mean, (34_503, 8, 2))
                ),
                "constant training mean drift",
            )
            require(
                np.all(trivial["repeat_last_mean"] == trivial["repeat_last_mean"][:, :1, :]),
                "repeat-last horizon drift",
            )

            prediction_payloads: dict[str, bytes] = {}
            for record in records:
                name = str(record["name"])
                seed = int(record["seed"])
                key = f"{name}:{seed}"
                identity = (name, seed)
                prediction_file = f"predictions/{name}_seed_{seed}.npz"
                require(
                    record["prediction_file"] == prediction_file, f"prediction path drift: {key}"
                )
                require(record["window_count"] == 34_503, f"prediction window count drift: {key}")
                require(
                    record["checkpoint_sha256"] == result_audit["checkpoint_sha256"][key]
                    and record["result_sha256"] == result_audit["result_sha256"][key],
                    f"checkpoint/result binding drift: {key}",
                )
                payload = bundle.read(prediction_file)
                require(
                    len(payload) == int(record["prediction_size_bytes"])
                    and sha256_bytes(payload) == record["prediction_sha256"],
                    f"prediction payload binding drift: {key}",
                )
                prediction_payloads[prediction_file] = payload
                arrays = load_npz(payload)
                base_fields = {
                    "log_variance",
                    "mean",
                    "normalized_mse",
                    "per_horizon_squared_error",
                }
                hetero_fields = {
                    "aleatoric_variance",
                    "combined",
                    "first_step_max_sigma",
                    "max_sigma",
                    "nll",
                    "z_a",
                    "z_m",
                }
                heteroscedastic = name in HETEROSCEDASTIC_NAMES
                require(
                    set(arrays) == base_fields | (hetero_fields if heteroscedastic else set()),
                    f"prediction NPZ schema drift: {key}",
                )
                require(
                    arrays["mean"].shape == arrays["log_variance"].shape == (34_503, 8, 2)
                    and arrays["normalized_mse"].shape == (34_503,)
                    and arrays["per_horizon_squared_error"].shape == (34_503, 8),
                    f"prediction array shape drift: {key}",
                )
                require(
                    all(np.all(np.isfinite(value)) for value in arrays.values()),
                    f"non-finite prediction array: {key}",
                )
                error = (arrays["mean"].astype(np.float64) - common["target"]) / scale
                squared = np.square(error)
                numerical_max = max(
                    numerical_max,
                    max_difference(
                        squared.mean(axis=(1, 2)), arrays["normalized_mse"], f"MSE:{key}"
                    ),
                    max_difference(
                        squared.mean(axis=2),
                        arrays["per_horizon_squared_error"],
                        f"horizon MSE:{key}",
                    ),
                )
                if heteroscedastic:
                    score_record = score_records[identity]
                    require(
                        record["validation_score_sha256"] == score_record["score_sha256"]
                        and record["closed_loop_shortlist"] == (name in CLOSED_LOOP_NAMES),
                        f"validation score binding drift: {key}",
                    )
                    require(
                        record["thresholds"]
                        == (score_record["thresholds"] if name in CLOSED_LOOP_NAMES else None),
                        f"threshold binding drift: {key}",
                    )
                    score_arrays = load_npz(score_bundle.read(score_record["score_file"]))
                    clamped = np.clip(arrays["log_variance"].astype(np.float64), -5.0, 2.0)
                    variance = np.exp(clamped)
                    sigma = np.exp(0.5 * clamped)
                    nll = (0.5 * (np.exp(-clamped) * squared + clamped)).mean(axis=(1, 2))
                    aleatoric = variance.mean(axis=(1, 2))
                    z_a = (
                        np.searchsorted(
                            score_arrays["aleatoric_cdf_sorted"], aleatoric, side="right"
                        )
                        / score_arrays["aleatoric_cdf_sorted"].size
                    )
                    z_m = (
                        np.searchsorted(
                            score_arrays["mahalanobis_cdf_sorted"],
                            common["mahalanobis_distance"],
                            side="right",
                        )
                        / score_arrays["mahalanobis_cdf_sorted"].size
                    )
                    checks = {
                        "nll": nll,
                        "aleatoric_variance": aleatoric,
                        "max_sigma": sigma.max(axis=(1, 2)),
                        "first_step_max_sigma": sigma[:, 0, :].max(axis=1),
                        "z_a": z_a,
                        "z_m": z_m,
                        "combined": np.maximum(z_a, z_m),
                    }
                    for field, expected in checks.items():
                        numerical_max = max(
                            numerical_max,
                            max_difference(expected, arrays[field], f"{field}:{key}"),
                        )
                else:
                    require(
                        record["loss"] == "mean_only"
                        and record["validation_score_sha256"] is None
                        and record["thresholds"] is None
                        and record["closed_loop_shortlist"] is False,
                        f"mean-only contract drift: {key}",
                    )

        summary_raw = bundle.read("summary.json")
        summary = json.loads(summary_raw)
        recomputed = build_summary(prediction_payloads, records, common, trivial)
        require(recomputed == summary, "exact summary reproduction failed")
        return {
            "schema_version": 1,
            "status": "PASS",
            "bundle": {
                "path": str(bundle_path),
                "size_bytes": BUNDLE_SIZE_BYTES,
                "sha256": BUNDLE_SHA256,
                "member_count": 29,
                "manifest_file_sha256": MANIFEST_FILE_SHA256,
                "manifest_self_sha256": MANIFEST_SELF_SHA256,
                "zip_crc": "PASS",
                "deterministic_metadata": True,
                "all_member_hashes_verified": True,
            },
            "upstream": upstream,
            "payload": {
                "episodes": 110,
                "windows": 34_503,
                "groups": group_counts,
                "prediction_records": 24,
                "heteroscedastic_records": 21,
                "closed_loop_threshold_records": 12,
                "trivial_baselines": 2,
                "all_npz_pickle_free": True,
                "all_arrays_finite": True,
                "all_raw_metrics_recomputed": True,
                "maximum_absolute_reproduction_difference": numerical_max,
                "summary_exactly_reproduced": True,
                "bootstrap_records_reproduced": len(summary["paired_mse_contrasts"]),
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "livifuser_sim_heldout_evaluation_v1_bundle.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "sim_heldout_evaluation_v1_audit.json",
    )
    args = parser.parse_args()
    report = audit(args.bundle.resolve())
    output = args.output.resolve()
    write_json_atomic(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "audit": str(output),
                "audit_sha256": sha256_file(output),
                "bundle": report["bundle"],
                "payload": report["payload"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
