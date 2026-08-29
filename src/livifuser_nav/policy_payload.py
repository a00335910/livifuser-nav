"""Seal and verify the exact 12-policy closed-loop payload."""

from __future__ import annotations

import copy
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from livifuser_nav.backbone_handoff import (
    ZIP_TIMESTAMP,
    _zip_info,
    json_bytes,
    sha256_bytes,
    sha256_file,
)

RESULT_ARCHIVE_SHA256 = (
    "F5B7D9EAB29DD20CE6710E4B803EAA331A5D7C2E741E9330995A1EAE615B9AC7"
)
SCORE_ARCHIVE_SHA256 = (
    "07116A629E296929D69EDA41E44CB6067CB6C751C735B66FD0A1B736D240751B"
)
SCORE_MANIFEST_FILE_SHA256 = (
    "BFD5A21F150DCFCF12CD988821DE6901A1558ACC0EA183D7F8223940C2C1A729"
)
SCORE_MANIFEST_SELF_SHA256 = (
    "AA90B540579C8285F55422DB41EA549305FA6C755FDF391FA3CD06ADC82127BF"
)
CONFIG_SHA256 = "76680BCE45A67D5D91F42660D5EC25F90450B281838CE311E329477C4E36F09E"
VARIANTS = ("concat", "full", "lidar_only", "rgb_only")
SEEDS = (20260805, 20260806, 20260807)
BUNDLE_ROOT = "livifuser_sim_closed_loop_policy_payload_v1"
BUNDLE_FILENAME = f"{BUNDLE_ROOT}_bundle.zip"
MANIFEST_NAME = "POLICY_PAYLOAD_MANIFEST.json"
COMPLETE_NAME = "POLICY_PAYLOAD_COMPLETE.json"
EXPECTED_MEMBER_COUNT = 29


def canonical_self_hash(value: dict[str, Any], field: str) -> str:
    copied = copy.deepcopy(value)
    copied.pop(field, None)
    payload = json.dumps(copied, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_names(archive: zipfile.ZipFile) -> None:
    names = archive.namelist()
    _require(len(names) == len(set(names)), "archive contains duplicate members")
    _require(archive.testzip() is None, "archive CRC failure")
    for info in archive.infolist():
        parts = Path(info.filename).parts
        _require(
            not info.filename.startswith(("/", "\\"))
            and ".." not in parts,
            f"unsafe archive member: {info.filename}",
        )


def _checkpoint_member(archive: zipfile.ZipFile, variant: str, seed: int) -> str:
    suffix = f"/{variant}/seed_{seed}/checkpoint.pt"
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    _require(len(names) == 1, f"expected one checkpoint member for {variant}/{seed}")
    return names[0]


def _validate_checkpoint(payload: bytes, variant: str, seed: int) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime requirement
        raise RuntimeError("PyTorch is required for checkpoint verification") from exc

    from livifuser_nav.model import LiViFuserPolicy

    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    _require(set(checkpoint) == {"model_state_dict", "variant", "seed", "config_sha256"},
             f"checkpoint metadata keys drifted for {variant}/{seed}")
    _require(checkpoint["variant"] == variant, f"checkpoint variant drifted: {variant}/{seed}")
    _require(int(checkpoint["seed"]) == seed, f"checkpoint seed drifted: {variant}/{seed}")
    _require(checkpoint["config_sha256"] == CONFIG_SHA256,
             f"checkpoint config drifted: {variant}/{seed}")
    model = LiViFuserPolicy(variant=variant)
    expected = model.state_dict()
    observed = checkpoint["model_state_dict"]
    _require(set(observed) == set(expected), f"checkpoint state keys drifted: {variant}/{seed}")
    for name, tensor in expected.items():
        candidate = observed[name]
        _require(candidate.shape == tensor.shape,
                 f"checkpoint shape drifted: {variant}/{seed}/{name}")
        _require(candidate.dtype == tensor.dtype,
                 f"checkpoint dtype drifted: {variant}/{seed}/{name}")
    model.load_state_dict(observed, strict=True)


def _validate_score(payload: bytes, variant: str, seed: int) -> None:
    with np.load(io.BytesIO(payload), allow_pickle=False) as score:
        required = {"aleatoric_cdf_sorted", "mahalanobis_cdf_sorted"}
        _require(required.issubset(score.files), f"score arrays missing for {variant}/{seed}")
        for name in sorted(required):
            values = np.asarray(score[name], dtype=np.float64)
            _require(values.shape == (9459,), f"score CDF shape drifted: {variant}/{seed}/{name}")
            _require(np.all(np.isfinite(values)), f"non-finite score CDF: {variant}/{seed}/{name}")
            _require(np.all(values[1:] >= values[:-1]),
                     f"unsorted score CDF: {variant}/{seed}/{name}")


def _record(name: str, payload: bytes, source: str) -> dict[str, Any]:
    return {
        "name": name,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "source": source,
    }


def _source_payloads(
    result_archive: Path, score_archive: Path, config_path: Path
) -> tuple[dict[str, bytes], list[dict[str, Any]], dict[str, Any]]:
    _require(sha256_file(result_archive) == RESULT_ARCHIVE_SHA256,
             "training result archive SHA-256 drifted")
    _require(sha256_file(score_archive) == SCORE_ARCHIVE_SHA256,
             "validation score archive SHA-256 drifted")
    config_payload = config_path.read_bytes()
    _require(sha256_bytes(config_payload) == CONFIG_SHA256,
             "frozen simulation config SHA-256 drifted")

    members: dict[str, bytes] = {"config/simulation_sweep_v1.json": config_payload}
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(result_archive) as results, zipfile.ZipFile(score_archive) as scores:
        _safe_names(results)
        _safe_names(scores)
        result_manifest = json.loads(
            results.read("livifuser_simulation_sweep_v1/RESULT_BUNDLE_MANIFEST.json")
        )
        _require(result_manifest.get("config_sha256") == CONFIG_SHA256,
                 "result manifest config identity drifted")
        _require(int(result_manifest.get("result_count", -1)) == 24,
                 "result manifest count drifted")

        score_manifest_payload = scores.read("SCORE_FREEZE_MANIFEST.json")
        _require(sha256_bytes(score_manifest_payload) == SCORE_MANIFEST_FILE_SHA256,
                 "score manifest file hash drifted")
        score_manifest = json.loads(score_manifest_payload)
        _require(score_manifest.get("manifest_sha256_excludes_self") == SCORE_MANIFEST_SELF_SHA256,
                 "score manifest declared self-hash drifted")
        _require(canonical_self_hash(score_manifest, "manifest_sha256_excludes_self")
                 == SCORE_MANIFEST_SELF_SHA256, "score manifest self-hash failed")
        completion = json.loads(scores.read("SCORE_FREEZE_COMPLETE.json"))
        _require(completion.get("status") == "COMPLETE"
                 and completion.get("manifest_file_sha256") == SCORE_MANIFEST_FILE_SHA256
                 and completion.get("manifest_sha256_excludes_self")
                 == SCORE_MANIFEST_SELF_SHA256,
                 "score completion marker drifted")
        shortlist = [row for row in score_manifest["records"] if row["closed_loop_shortlist"]]
        _require(len(shortlist) == 12, "score manifest shortlist count drifted")
        indexed = {(row["name"], int(row["seed"])): row for row in shortlist}
        _require(set(indexed) == {(variant, seed) for variant in VARIANTS for seed in SEEDS},
                 "score manifest shortlist identities drifted")

        for variant in VARIANTS:
            for seed in SEEDS:
                score_record = indexed[(variant, seed)]
                _require(score_record["variant"] == variant
                         and score_record["loss"] == "heteroscedastic",
                         f"score record metadata drifted: {variant}/{seed}")
                _require(score_record["thresholds"]["combined"] == 1.0,
                         f"combined threshold drifted: {variant}/{seed}")
                checkpoint_source = _checkpoint_member(results, variant, seed)
                checkpoint = results.read(checkpoint_source)
                _require(sha256_bytes(checkpoint) == score_record["checkpoint_sha256"],
                         f"checkpoint hash drifted: {variant}/{seed}")
                _validate_checkpoint(checkpoint, variant, seed)
                checkpoint_name = f"checkpoints/{variant}_seed_{seed}.pt"
                members[checkpoint_name] = checkpoint

                score_source = score_record["score_file"]
                score_payload = scores.read(score_source)
                _require(len(score_payload) == int(score_record["score_size_bytes"])
                         and sha256_bytes(score_payload) == score_record["score_sha256"],
                         f"score member identity drifted: {variant}/{seed}")
                _validate_score(score_payload, variant, seed)
                score_name = f"scores/{variant}_seed_{seed}.npz"
                members[score_name] = score_payload
                records.append({
                    "variant": variant,
                    "seed": seed,
                    "checkpoint": _record(checkpoint_name, checkpoint, checkpoint_source),
                    "score": _record(score_name, score_payload, score_source),
                    "thresholds": score_record["thresholds"],
                    "threshold_rule": score_record["threshold_rule"],
                    "validation_windows": score_record["validation_windows"],
                    "validation_episodes": score_record["validation_episodes"],
                })

        gaussian_names = {
            "mahalanobis/mean.npy":
                "livifuser_simulation_sweep_v1/worker_0/mahalanobis_mean.npy",
            "mahalanobis/precision.npy":
                "livifuser_simulation_sweep_v1/worker_0/mahalanobis_precision.npy",
        }
        for target, source in gaussian_names.items():
            payload = results.read(source)
            peer = source.replace("worker_0", "worker_1")
            _require(payload == results.read(peer), f"worker Gaussian copies differ: {target}")
            array = np.load(io.BytesIO(payload), allow_pickle=False)
            expected_shape = (384,) if target.endswith("mean.npy") else (384, 384)
            _require(array.shape == expected_shape and np.all(np.isfinite(array)),
                     f"training Gaussian array drifted: {target}")
            members[target] = payload

    expected = 1 + 12 + 12 + 2
    _require(len(members) == expected, "policy payload source member count drifted")
    source_identities = {
        "training_result_archive_sha256": RESULT_ARCHIVE_SHA256,
        "validation_score_archive_sha256": SCORE_ARCHIVE_SHA256,
        "validation_score_manifest_file_sha256": SCORE_MANIFEST_FILE_SHA256,
        "validation_score_manifest_self_sha256": SCORE_MANIFEST_SELF_SHA256,
        "simulation_config_sha256": CONFIG_SHA256,
    }
    return members, records, source_identities


def seal_policy_payload(
    result_archive: str | Path,
    score_archive: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite policy payload: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    members, records, sources = _source_payloads(
        Path(result_archive).resolve(), Path(score_archive).resolve(), Path(config_path).resolve()
    )
    member_records = [_record(name, members[name], "sealed_input") for name in sorted(members)]
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "sealed_closed_loop_policy_payload",
        "scope": {
            "variants": list(VARIANTS),
            "seeds": list(SEEDS),
            "policy_identity_count": 12,
            "training_or_inference_performed": False,
        },
        "sources": sources,
        "records": records,
        "members": member_records,
        "member_count_including_manifest_and_completion": EXPECTED_MEMBER_COUNT,
        "zip_contract": {
            "root": BUNDLE_ROOT,
            "compression": "stored",
            "timestamp": "1980-01-01T00:00:00",
            "unix_mode": "100644",
        },
    }
    manifest["manifest_sha256_excludes_self"] = canonical_self_hash(
        manifest, "manifest_sha256_excludes_self"
    )
    manifest_payload = json_bytes(manifest)
    completion = {
        "schema_version": "1.0.0",
        "status": "complete",
        "manifest_file_sha256": sha256_bytes(manifest_payload),
        "manifest_sha256_excludes_self": manifest["manifest_sha256_excludes_self"],
        "member_count": EXPECTED_MEMBER_COUNT,
        "policy_identity_count": 12,
    }
    with zipfile.ZipFile(output, "x", allowZip64=True) as archive:
        for name in sorted(members):
            archive.writestr(_zip_info(f"{BUNDLE_ROOT}/{name}"), members[name])
        archive.writestr(_zip_info(f"{BUNDLE_ROOT}/{MANIFEST_NAME}"), json_bytes(manifest))
        archive.writestr(_zip_info(f"{BUNDLE_ROOT}/{COMPLETE_NAME}"), json_bytes(completion))
    return verify_policy_payload(output)


def verify_policy_payload(bundle_path: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    _require(bundle.is_file(), f"policy payload does not exist: {bundle}")
    with zipfile.ZipFile(bundle) as archive:
        _safe_names(archive)
        infos = archive.infolist()
        _require(len(infos) == EXPECTED_MEMBER_COUNT, "policy payload member count drifted")
        for info in infos:
            _require(info.date_time == ZIP_TIMESTAMP
                     and info.compress_type == zipfile.ZIP_STORED,
                     f"policy payload ZIP metadata drifted: {info.filename}")
            _require((info.external_attr >> 16) == 0o100644,
                     f"policy payload member mode drifted: {info.filename}")
        manifest_name = f"{BUNDLE_ROOT}/{MANIFEST_NAME}"
        completion_name = f"{BUNDLE_ROOT}/{COMPLETE_NAME}"
        manifest_payload = archive.read(manifest_name)
        manifest = json.loads(manifest_payload)
        declared = manifest.get("manifest_sha256_excludes_self")
        _require(declared == canonical_self_hash(manifest, "manifest_sha256_excludes_self"),
                 "policy payload manifest self-hash mismatch")
        _require(manifest.get("sources") == {
            "simulation_config_sha256": CONFIG_SHA256,
            "training_result_archive_sha256": RESULT_ARCHIVE_SHA256,
            "validation_score_archive_sha256": SCORE_ARCHIVE_SHA256,
            "validation_score_manifest_file_sha256": SCORE_MANIFEST_FILE_SHA256,
            "validation_score_manifest_self_sha256": SCORE_MANIFEST_SELF_SHA256,
        }, "policy payload source identities drifted")
        records = manifest.get("records", [])
        _require(len(records) == 12, "policy payload record count drifted")
        _require({(row["variant"], int(row["seed"])) for row in records}
                 == {(variant, seed) for variant in VARIANTS for seed in SEEDS},
                 "policy payload record identities drifted")
        listed = manifest.get("members", [])
        expected_names = {manifest_name, completion_name}
        for row in listed:
            name = f"{BUNDLE_ROOT}/{row['name']}"
            expected_names.add(name)
            payload = archive.read(name)
            _require(len(payload) == int(row["size_bytes"])
                     and sha256_bytes(payload) == row["sha256"],
                     f"policy payload nested identity mismatch: {row['name']}")
        _require(set(archive.namelist()) == expected_names,
                 "policy payload exact member set drifted")
        for row in records:
            variant, seed = row["variant"], int(row["seed"])
            checkpoint = archive.read(f"{BUNDLE_ROOT}/{row['checkpoint']['name']}")
            score = archive.read(f"{BUNDLE_ROOT}/{row['score']['name']}")
            _validate_checkpoint(checkpoint, variant, seed)
            _validate_score(score, variant, seed)
            _require(row["thresholds"]["combined"] == 1.0,
                     f"sealed combined threshold drifted: {variant}/{seed}")
        completion = json.loads(archive.read(completion_name))
        _require(completion == {
            "schema_version": "1.0.0",
            "status": "complete",
            "manifest_file_sha256": sha256_bytes(manifest_payload),
            "manifest_sha256_excludes_self": declared,
            "member_count": EXPECTED_MEMBER_COUNT,
            "policy_identity_count": 12,
        }, "policy payload completion marker drifted")
    return {
        "schema_version": "1.0.0",
        "status": "verified",
        "bundle_path": str(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
        "bundle_sha256": sha256_file(bundle),
        "member_count": EXPECTED_MEMBER_COUNT,
        "policy_identity_count": 12,
        "manifest_file_sha256": sha256_bytes(manifest_payload),
        "manifest_sha256_excludes_self": declared,
    }
