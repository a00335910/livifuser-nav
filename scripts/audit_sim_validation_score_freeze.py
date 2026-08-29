#!/usr/bin/env python3
"""Independently audit the immutable validation uncertainty score-freeze bundle."""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import struct
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import numpy as np  # noqa: E402
from prepare_sim_training_data import (  # noqa: E402
    BACKBONE_CONTRACT_SHA256,
    CACHE_MANIFEST_SHA256,
    CACHE_SELF_SHA256,
    HANDOFF_SELF_SHA256,
)
from prepare_sim_training_data import (  # noqa: E402
    self_hash as source_self_hash,
)
from replay_sim_validation_scores import (  # noqa: E402
    CLOSED_LOOP_NAMES,
    CONFIG_SHA256,
    HETEROSCEDASTIC_PARTITIONS,
    RESULT_ARCHIVE_SHA256,
    SEEDS,
    checkpoint_member,
    episode_maxima,
    operating_threshold,
    result_member,
    right_continuous_cdf,
    sha256_bytes,
    sha256_file,
)

BUNDLE_SHA256 = "07116A629E296929D69EDA41E44CB6067CB6C751C735B66FD0A1B736D240751B"
BUNDLE_SIZE_BYTES = 6_927_317
MANIFEST_FILE_SHA256 = "BFD5A21F150DCFCF12CD988821DE6901A1558ACC0EA183D7F8223940C2C1A729"
MANIFEST_SELF_SHA256 = "AA90B540579C8285F55422DB41EA549305FA6C755FDF391FA3CD06ADC82127BF"
AUDIT_REPORT_SHA256 = "4D1CEA8F2D61EF76E1A48770FB6228F14683DAF6943C4932C06FCE0FB46611B3"
AMENDMENT_SHA256 = "8760474F1CCC6269BD23A28489DD01076891ECBF9E66A6F39BBF8E2838F6DCD7"
VALIDATION_CODE_BUNDLE_SHA256 = "F7C371D725808F15CAC9D6EC79BFB3C768BDFE8FFC9C8C6B34AA9D76EF58BC53"
VALIDATION_CODE_MANIFEST_SHA256 = "79E21F000B01C02E298C738332E4A8022735E7707137DD7318998C3663403201"
TRAINING_CODE_MANIFEST_SHA256 = "EB98B43B049FA6E2997176A43062D51926968390D1C07314D70DD56E8090ED0A"
TRAINING_DATA_PLAN_SHA256 = "F9977E1A9D6C58AFA99A743C5ABD1A319FD56D58111052804C8327251C9789F0"
VALIDATION_DATA_PLAN_SHA256 = "748A4FD00622448DE5C41EFB6D831E4AAC64ABD7543DEFC1FE0702216F0F0966"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
ARRAY_NAMES = {
    "aleatoric_cdf_sorted",
    "aleatoric_episode_max",
    "aleatoric_variance",
    "combined",
    "combined_episode_max",
    "episode_ids",
    "episode_ids_unique",
    "first_step_max_sigma",
    "mahalanobis_cdf_sorted",
    "mahalanobis_distance",
    "mahalanobis_episode_max",
    "max_sigma",
    "origin_rows",
    "z_a",
    "z_m",
}
COMMON_FIELDS = (
    "episode_ids",
    "origin_rows",
    "mahalanobis_distance",
    "mahalanobis_cdf_sorted",
    "z_m",
    "episode_ids_unique",
    "mahalanobis_episode_max",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_self_hash(payload: dict[str, Any], field: str) -> str:
    value = copy.deepcopy(payload)
    value.pop(field, None)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(raw)


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(path.parts) and not path.is_absolute() and ".." not in path.parts


def expected_identities() -> set[tuple[str, int]]:
    return {
        (name, seed)
        for names in HETEROSCEDASTIC_PARTITIONS.values()
        for name in names
        for seed in SEEDS
    }


def expected_score_members() -> set[str]:
    return {f"scores/{name}_seed_{seed}.npz" for name, seed in expected_identities()}


def load_score_npz(payload: bytes, label: str) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(io.BytesIO(payload)) as nested:
        infos = nested.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)) == len(ARRAY_NAMES), f"{label}: NPZ member drift")
        require(
            set(names) == {f"{name}.npy" for name in ARRAY_NAMES},
            f"{label}: NPZ array set drift",
        )
        require(nested.testzip() is None, f"{label}: NPZ CRC failure")
        for info in infos:
            require(safe_member(info.filename), f"{label}: unsafe NPZ member")
            require(info.date_time == FIXED_ZIP_TIME, f"{label}: NPZ timestamp drift")
            require(info.compress_type == zipfile.ZIP_DEFLATED, f"{label}: NPZ compression drift")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    require(set(arrays) == ARRAY_NAMES, f"{label}: loaded array set drift")
    require(
        not any(array.dtype.hasobject for array in arrays.values()),
        f"{label}: object/pickle array forbidden",
    )
    return arrays


def singleton_npz_from_score(payload: bytes, array_name: str) -> bytes:
    """Rebuild a singleton NPZ while preserving the producer's compressed bytes."""

    member_name = f"{array_name}.npy"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        info = archive.getinfo(member_name)
        local_offset = info.header_offset
        local_header = payload[local_offset : local_offset + 30]
        fields = struct.unpack("<IHHHHHIIIHH", local_header)
        signature, _version, flags, _compression, *_rest, name_length, extra_length = fields
        require(signature == 0x04034B50, f"local NPZ signature drift: {member_name}")
        require(not flags & 0x08, f"NPZ data descriptor unsupported: {member_name}")
        local_length = 30 + name_length + extra_length + info.compress_size
        local_record = payload[local_offset : local_offset + local_length]
        position = archive.start_dir
        central_record: bytearray | None = None
        while payload[position : position + 4] == b"PK":
            central_name_length, central_extra_length, comment_length = struct.unpack(
                "<HHH", payload[position + 28 : position + 34]
            )
            record_length = 46 + central_name_length + central_extra_length + comment_length
            name = payload[position + 46 : position + 46 + central_name_length].decode()
            if name == member_name:
                central_record = bytearray(payload[position : position + record_length])
                break
            position += record_length
    require(central_record is not None, f"NPZ central record missing: {member_name}")
    central_record[42:46] = struct.pack("<I", 0)
    central = bytes(central_record)
    end = struct.pack(
        "<IHHHHIIH",
        0x06054B50,
        0,
        0,
        1,
        1,
        len(central),
        len(local_record),
        0,
    )
    return local_record + central + end


def verify_code_bundle(path: Path) -> dict[str, Any]:
    require(path.stat().st_size > 0, "validation code bundle is empty")
    require(
        sha256_file(path) == VALIDATION_CODE_BUNDLE_SHA256,
        "validation code bundle hash drift",
    )
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)) == 32, "code bundle member count drift")
        require(archive.testzip() is None, "code bundle CRC failure")
        manifest_raw = archive.read("LiViFuser/cloud_bundle_manifest.json")
        require(
            sha256_bytes(manifest_raw) == VALIDATION_CODE_MANIFEST_SHA256,
            "validation code manifest hash drift",
        )
        manifest = json.loads(manifest_raw)
        entries = {str(row["path"]): row for row in manifest["files"]}
        expected = {
            "LiViFuser/cloud_bundle_manifest.json",
            *(f"LiViFuser/{name}" for name in entries),
        }
        require(set(names) == expected, "code bundle exact member set drift")
        for name, row in entries.items():
            raw = archive.read(f"LiViFuser/{name}")
            require(len(raw) == int(row["size_bytes"]), f"code member size drift: {name}")
            require(sha256_bytes(raw) == row["sha256"], f"code member hash drift: {name}")
    return {
        "bundle_sha256": VALIDATION_CODE_BUNDLE_SHA256,
        "manifest_sha256": VALIDATION_CODE_MANIFEST_SHA256,
        "file_count": len(entries),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, indent=2) + chr(10)
    if path.exists():
        require(path.read_text(encoding="utf-8") == raw, f"existing audit report drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(raw, encoding="utf-8", newline=chr(10))
    temporary.replace(path)


def audit(
    bundle_path: Path,
    result_archive_path: Path,
    result_audit_path: Path,
    config_path: Path,
    amendment_path: Path,
    handoff_path: Path,
    code_bundle_path: Path,
) -> dict[str, Any]:
    require(bundle_path.stat().st_size == BUNDLE_SIZE_BYTES, "score bundle size drift")
    require(sha256_file(bundle_path) == BUNDLE_SHA256, "score bundle SHA-256 drift")
    require(sha256_file(result_archive_path) == RESULT_ARCHIVE_SHA256, "result ZIP hash drift")
    require(sha256_file(result_audit_path) == AUDIT_REPORT_SHA256, "result audit hash drift")
    require(sha256_file(config_path) == CONFIG_SHA256, "simulation config hash drift")
    require(sha256_file(amendment_path) == AMENDMENT_SHA256, "amendment hash drift")
    code_verification = verify_code_bundle(code_bundle_path)
    result_audit = json.loads(result_audit_path.read_text(encoding="utf-8"))
    require(result_audit["status"] == "PASS", "upstream result audit is not PASS")
    require(
        result_audit["archive"]["sha256"] == RESULT_ARCHIVE_SHA256,
        "upstream result archive identity drift",
    )

    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    require(
        handoff["manifest_sha256_excludes_self"] == HANDOFF_SELF_SHA256,
        "handoff declared self-hash drift",
    )
    require(
        source_self_hash(handoff, "manifest_sha256_excludes_self") == HANDOFF_SELF_SHA256,
        "handoff self-hash drift",
    )
    expected_validation_ids = [
        row["episode_id"]
        for row in sorted(
            (row for row in handoff["episodes"] if row["split"] == "val_id"),
            key=lambda row: int(row["ordinal"]),
        )
    ]
    require(len(expected_validation_ids) == 30, "handoff validation identity count drift")

    with (
        zipfile.ZipFile(bundle_path) as bundle,
        zipfile.ZipFile(result_archive_path) as results,
    ):
        infos = bundle.infolist()
        names = [info.filename for info in infos]
        expected_members = {
            "SCORE_FREEZE_MANIFEST.json",
            "SCORE_FREEZE_COMPLETE.json",
            *expected_score_members(),
        }
        require(len(names) == len(set(names)) == 23, "bundle member count/duplicate drift")
        require(set(names) == expected_members, "bundle exact member set drift")
        require(all(safe_member(name) for name in names), "unsafe bundle member")
        require(bundle.testzip() is None, "bundle CRC failure")
        for info in infos:
            require(not info.is_dir(), "unexpected bundle directory entry")
            require(info.date_time == FIXED_ZIP_TIME, f"bundle timestamp drift: {info.filename}")
            require(
                info.compress_type == zipfile.ZIP_DEFLATED,
                f"bundle compression drift: {info.filename}",
            )

        manifest_raw = bundle.read("SCORE_FREEZE_MANIFEST.json")
        completion_raw = bundle.read("SCORE_FREEZE_COMPLETE.json")
        require(sha256_bytes(manifest_raw) == MANIFEST_FILE_SHA256, "manifest file hash drift")
        manifest = json.loads(manifest_raw)
        completion = json.loads(completion_raw)
        field = "manifest_sha256_excludes_self"
        require(manifest[field] == MANIFEST_SELF_SHA256, "manifest declared self-hash drift")
        require(
            canonical_self_hash(manifest, field) == MANIFEST_SELF_SHA256,
            "manifest self-hash recomputation failed",
        )
        require(
            completion
            == {
                "schema_version": 1,
                "status": "COMPLETE",
                "manifest_file_sha256": MANIFEST_FILE_SHA256,
                "manifest_sha256_excludes_self": MANIFEST_SELF_SHA256,
                "score_member_count": 21,
                "heteroscedastic_record_count": 21,
                "closed_loop_threshold_record_count": 12,
                "exact_bundle_member_count": 23,
            },
            "completion marker drift",
        )
        require(
            manifest["schema_version"] == 1
            and manifest["status"] == "FROZEN_VALIDATION_ONLY"
            and manifest["purpose"] == "simulation_uncertainty_score_freeze_before_heldout",
            "manifest status/purpose drift",
        )
        require(
            manifest["heldout"]
            == {
                "attached": False,
                "opened": False,
                "hashed": False,
                "excluded_splits": ["test_id", "test_ood"],
            },
            "held-out exclusion declaration drift",
        )
        require(
            manifest["amendment"]["sha256"] == AMENDMENT_SHA256
            and manifest["amendment"]["approval_date"] == "2026-08-24",
            "amendment binding drift",
        )
        identities = manifest["identities"]
        expected_identity_values = {
            "result_archive_sha256": RESULT_ARCHIVE_SHA256,
            "result_audit_report_sha256": AUDIT_REPORT_SHA256,
            "simulation_config_sha256": CONFIG_SHA256,
            "validation_data_plan_sha256": VALIDATION_DATA_PLAN_SHA256,
            "source_handoff_self_sha256": HANDOFF_SELF_SHA256,
            "cache_manifest_sha256": CACHE_MANIFEST_SHA256,
            "cache_manifest_self_sha256": CACHE_SELF_SHA256,
            "backbone_contract_sha256": BACKBONE_CONTRACT_SHA256,
            "training_data_plan_sha256": TRAINING_DATA_PLAN_SHA256,
            "validation_code_cloud_manifest_sha256": VALIDATION_CODE_MANIFEST_SHA256,
            "training_code_cloud_manifest_sha256": TRAINING_CODE_MANIFEST_SHA256,
        }
        for key, value in expected_identity_values.items():
            require(identities.get(key) == value, f"manifest identity drift: {key}")
        require(
            identities["backbone_contract_sha256"]
            == result_audit["frozen_provenance"]["backbone"]["backbone_contract_sha256"],
            "result/backbone binding drift",
        )

        validation = manifest["validation"]
        require(
            validation["episode_count"] == 30
            and validation["window_count"] == 9459
            and validation["episode_ids"] == expected_validation_ids,
            "validation count/episode identity drift",
        )
        require(
            not any(
                value.startswith(("test_id_", "test_ood_")) for value in validation["episode_ids"]
            ),
            "held-out episode in validation manifest",
        )
        contract = manifest["score_contract"]
        require(
            contract["aleatoric"] == "mean(exp(clip(log_var,-5,2))) over Hx2"
            and contract["cdf"] == "right_continuous; count(reference<=x)/N"
            and contract["combined"] == "max(z_a,z_m)"
            and contract["episode_reduction"] == "maximum_window_score"
            and contract["threshold"] == "29th_order_statistic_of_30; strict_greater_than"
            and contract["full_mean_only_excluded"] is True,
            "score contract drift",
        )

        member_rows = {row["name"]: row for row in manifest["members"]}
        require(
            len(member_rows) == 21 and set(member_rows) == expected_score_members(),
            "manifest score member records drift",
        )
        records = manifest["records"]
        identities_observed = {(str(record["name"]), int(record["seed"])) for record in records}
        require(
            len(records) == len(identities_observed) == 21
            and identities_observed == expected_identities(),
            "score record identity set drift",
        )
        require(
            sum(bool(record["closed_loop_shortlist"]) for record in records) == 12,
            "closed-loop record count drift",
        )
        require(
            {record["name"] for record in records if record["closed_loop_shortlist"]}
            == CLOSED_LOOP_NAMES,
            "closed-loop name set drift",
        )

        common: dict[str, np.ndarray] | None = None
        producer_common_hashes: dict[str, str] | None = None
        threshold_records: dict[str, Any] = {}
        reproduction_maxima = {"mse": 0.0, "nll": 0.0, "sigma_coverage": 0.0}
        for record in records:
            name = str(record["name"])
            seed = int(record["seed"])
            identity_key = f"{name}:{seed}"
            score_name = str(record["score_file"])
            require(score_name in member_rows, f"unmanifested score file: {score_name}")
            score_raw = bundle.read(score_name)
            member_row = member_rows[score_name]
            require(
                len(score_raw) == int(record["score_size_bytes"]) == int(member_row["size_bytes"]),
                f"score size drift: {identity_key}",
            )
            score_sha256 = sha256_bytes(score_raw)
            require(
                score_sha256 == record["score_sha256"] == member_row["sha256"],
                f"score hash drift: {identity_key}",
            )
            result_raw = results.read(result_member(name, seed))
            checkpoint_raw = results.read(checkpoint_member(name, seed))
            require(
                sha256_bytes(result_raw)
                == record["result_sha256"]
                == result_audit["result_sha256"][identity_key],
                f"result binding drift: {identity_key}",
            )
            require(
                sha256_bytes(checkpoint_raw)
                == record["checkpoint_sha256"]
                == result_audit["checkpoint_sha256"][identity_key],
                f"checkpoint binding drift: {identity_key}",
            )
            original = json.loads(result_raw)
            require(
                original["run"]["name"] == name
                and original["run"]["variant"] == record["variant"]
                and original["run"]["loss"] == record["loss"] == "heteroscedastic"
                and int(original["seed"]) == seed,
                f"result semantic identity drift: {identity_key}",
            )
            arrays = load_score_npz(score_raw, identity_key)
            window_fields = (
                "episode_ids",
                "origin_rows",
                "aleatoric_variance",
                "max_sigma",
                "first_step_max_sigma",
                "aleatoric_cdf_sorted",
                "mahalanobis_distance",
                "mahalanobis_cdf_sorted",
                "z_a",
                "z_m",
                "combined",
            )
            episode_fields = (
                "episode_ids_unique",
                "aleatoric_episode_max",
                "mahalanobis_episode_max",
                "combined_episode_max",
            )
            require(
                all(arrays[field].shape == (9459,) for field in window_fields),
                f"window array shape drift: {identity_key}",
            )
            require(
                all(arrays[field].shape == (30,) for field in episode_fields),
                f"episode array shape drift: {identity_key}",
            )
            require(
                record["validation_windows"] == 9459 and record["validation_episodes"] == 30,
                f"record count drift: {identity_key}",
            )
            numeric = [
                array
                for key, array in arrays.items()
                if key not in {"episode_ids", "episode_ids_unique"}
            ]
            require(
                all(np.all(np.isfinite(array)) for array in numeric),
                f"non-finite score array: {identity_key}",
            )
            episode_ids = arrays["episode_ids"].astype(str)
            origin_rows = arrays["origin_rows"]
            stored_window = original["validation"]["per_window"]
            require(
                episode_ids.tolist() == stored_window["episode_ids"]
                and origin_rows.tolist() == stored_window["origin_rows"],
                f"result window identity drift: {identity_key}",
            )
            require(
                np.array_equal(
                    arrays["mahalanobis_distance"],
                    np.asarray(stored_window["mahalanobis_distance"], dtype=np.float64),
                ),
                f"result Mahalanobis drift: {identity_key}",
            )
            require(
                not any(value.startswith(("test_id_", "test_ood_")) for value in episode_ids),
                f"held-out score identity: {identity_key}",
            )
            require(
                arrays["episode_ids_unique"].astype(str).tolist()
                == sorted(expected_validation_ids),
                f"unique validation episode drift: {identity_key}",
            )
            require(
                np.all(arrays["aleatoric_variance"] >= math.exp(-5.0))
                and np.all(arrays["aleatoric_variance"] <= math.exp(2.0)),
                f"aleatoric clamp range drift: {identity_key}",
            )
            require(
                np.all(arrays["first_step_max_sigma"] <= arrays["max_sigma"]),
                f"secondary sigma ordering drift: {identity_key}",
            )
            require(
                np.array_equal(
                    arrays["aleatoric_cdf_sorted"],
                    np.sort(arrays["aleatoric_variance"]),
                )
                and np.array_equal(
                    arrays["mahalanobis_cdf_sorted"],
                    np.sort(arrays["mahalanobis_distance"]),
                ),
                f"sorted CDF reference drift: {identity_key}",
            )
            expected_z_a = right_continuous_cdf(
                arrays["aleatoric_cdf_sorted"], arrays["aleatoric_variance"]
            )
            expected_z_m = right_continuous_cdf(
                arrays["mahalanobis_cdf_sorted"], arrays["mahalanobis_distance"]
            )
            require(
                np.array_equal(arrays["z_a"], expected_z_a)
                and np.array_equal(arrays["z_m"], expected_z_m)
                and np.array_equal(arrays["combined"], np.maximum(expected_z_a, expected_z_m)),
                f"CDF/combined recomputation drift: {identity_key}",
            )
            for signal, window_values in (
                ("aleatoric", arrays["z_a"]),
                ("mahalanobis", arrays["z_m"]),
                ("combined", arrays["combined"]),
            ):
                unique_ids, maxima = episode_maxima(episode_ids.tolist(), window_values)
                require(
                    np.array_equal(unique_ids, arrays["episode_ids_unique"])
                    and np.array_equal(maxima, arrays[f"{signal}_episode_max"]),
                    f"episode maximum drift: {identity_key}:{signal}",
                )
                threshold, false_count = operating_threshold(maxima)
                require(
                    threshold == float(record["thresholds"][signal])
                    and false_count == int(record["validation_false_interventions"][signal]),
                    f"threshold recomputation drift: {identity_key}:{signal}",
                )
            require(
                record["threshold_rule"]
                == "second_largest_of_30_episode_maxima; strict_greater_than",
                f"threshold rule drift: {identity_key}",
            )
            reproduction = record["reproduction"]
            require(
                float(reproduction["rtol"]) == 1e-6
                and float(reproduction["atol"]) == 1e-7
                and float(reproduction["mse_max_abs_difference"]) <= 1e-7
                and float(reproduction["nll_max_abs_difference"]) <= 1e-7
                and float(reproduction["sigma_coverage_max_abs_difference"]) <= 1e-7,
                f"checkpoint replay reproduction drift: {identity_key}",
            )
            reproduction_maxima["mse"] = max(
                reproduction_maxima["mse"],
                float(reproduction["mse_max_abs_difference"]),
            )
            reproduction_maxima["nll"] = max(
                reproduction_maxima["nll"],
                float(reproduction["nll_max_abs_difference"]),
            )
            reproduction_maxima["sigma_coverage"] = max(
                reproduction_maxima["sigma_coverage"],
                float(reproduction["sigma_coverage_max_abs_difference"]),
            )
            if common is None:
                common = {field: arrays[field].copy() for field in COMMON_FIELDS}
                producer_common_hashes = {
                    field: sha256_bytes(singleton_npz_from_score(score_raw, field))
                    for field in COMMON_FIELDS
                }
            else:
                require(
                    all(np.array_equal(common[field], arrays[field]) for field in COMMON_FIELDS),
                    f"common validation array drift: {identity_key}",
                )
            threshold_records[identity_key] = {
                "closed_loop_shortlist": bool(record["closed_loop_shortlist"]),
                "thresholds": record["thresholds"],
                "validation_false_interventions": record["validation_false_interventions"],
                "score_sha256": score_sha256,
            }

        require(
            common is not None and producer_common_hashes is not None,
            "no common score arrays",
        )
        require(
            validation["common_array_sha256"] == producer_common_hashes,
            "manifest common-array hash drift",
        )

    combined_thresholds = {
        float(row["thresholds"]["combined"]) for row in threshold_records.values()
    }
    aleatoric_thresholds = {
        float(row["thresholds"]["aleatoric"]) for row in threshold_records.values()
    }
    return {
        "schema_version": 1,
        "status": "PASS",
        "bundle": {
            "path": str(bundle_path),
            "size_bytes": BUNDLE_SIZE_BYTES,
            "sha256": BUNDLE_SHA256,
            "member_count": 23,
            "score_member_count": 21,
            "manifest_file_sha256": MANIFEST_FILE_SHA256,
            "manifest_self_sha256": MANIFEST_SELF_SHA256,
            "completion_marker": "PASS",
            "zip_crc": "PASS",
            "deterministic_member_metadata": True,
        },
        "frozen_identities": expected_identity_values,
        "validation": {
            "episode_count": 30,
            "window_count": 9459,
            "episode_ids": expected_validation_ids,
            "all_common_arrays_byte_equal": True,
            "common_array_sha256": producer_common_hashes,
            "common_hash_verification": (
                "producer-compressed singleton NPZ reconstructed from the sealed "
                "score member without local recompression"
            ),
            "heldout_excluded": True,
        },
        "records": {
            "heteroscedastic_count": len(threshold_records),
            "closed_loop_threshold_count": sum(
                bool(row["closed_loop_shortlist"]) for row in threshold_records.values()
            ),
            "all_score_hashes": True,
            "all_checkpoint_bindings": True,
            "all_result_bindings": True,
            "all_npz_pickle_free": True,
            "all_cdf_recomputed": True,
            "all_episode_maxima_recomputed": True,
            "all_thresholds_recomputed": True,
            "reproduction_max_abs_difference": reproduction_maxima,
            "thresholds": threshold_records,
        },
        "threshold_observation": {
            "aleatoric_unique_values": sorted(aleatoric_thresholds),
            "combined_unique_values": sorted(combined_thresholds),
            "combined_all_equal_one": combined_thresholds == {1.0},
            "interpretation": (
                "The strict-greater-than frozen rule yields zero combined-score "
                "validation interventions when the threshold is 1.0; this is an "
                "audited consequence of the approved order statistic, not retuning."
            ),
        },
        "validation_code": code_verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=REPOSITORY_ROOT
        / "artifacts"
        / "livifuser_sim_validation_score_freeze_v1_bundle.zip",
    )
    parser.add_argument(
        "--result-archive",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "livifuser_simulation_sweep_v1_results.zip",
    )
    parser.add_argument(
        "--result-audit",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "simulation_sweep_v1_result_audit.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "simulation_sweep_v1.json",
    )
    parser.add_argument(
        "--amendment",
        type=Path,
        default=REPOSITORY_ROOT
        / "docs"
        / "experiments"
        / "PREREGISTRATION_SIM_EVALUATION_EXECUTION_AMENDMENT_2026-08-24.md",
    )
    parser.add_argument(
        "--handoff",
        type=Path,
        default=REPOSITORY_ROOT
        / "artifacts"
        / "simulation"
        / "gpu_handoff_train_val_v1"
        / "handoff_manifest.json",
    )
    parser.add_argument(
        "--code-bundle",
        type=Path,
        default=REPOSITORY_ROOT
        / "artifacts"
        / "cloud"
        / "livifuser_sim_validation_code_8760474f1ccc_r2.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "sim_validation_score_freeze_v1_audit.json",
    )
    args = parser.parse_args()
    inputs = (
        args.bundle,
        args.result_archive,
        args.result_audit,
        args.config,
        args.amendment,
        args.handoff,
        args.code_bundle,
    )
    require(all(path.resolve().is_file() for path in inputs), "audit input is missing")
    report = audit(*(path.resolve() for path in inputs))
    output = args.output.resolve()
    write_json_atomic(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "audit_report": str(output),
                "audit_report_sha256": sha256_file(output),
                "bundle": report["bundle"],
                "records": report["records"]["heteroscedastic_count"],
                "closed_loop_thresholds": report["records"]["closed_loop_threshold_count"],
                "combined_all_equal_one": report["threshold_observation"]["combined_all_equal_one"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
