#!/usr/bin/env python3
"""Replay frozen checkpoints on validation-ID and derive uncertainty scores."""

from __future__ import annotations

import argparse
import io
import json
import sys
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
    ROOT,
    sha256_bytes,
    sha256_file,
)
from prepare_sim_training_data import (  # noqa: E402
    BACKBONE_CONTRACT_SHA256,
    CACHE_MANIFEST_SHA256,
    CACHE_SELF_SHA256,
    HANDOFF_SELF_SHA256,
)
from run_baseline_sweep import (  # noqa: E402
    common_cache_identity,
    evaluate_model,
    load_split,
    resolve_device,
)

from livifuser_nav.evaluation import (  # noqa: E402
    LOG_VARIANCE_CLAMP,
    sigma_coverage,
    window_nll,
    window_normalized_mse,
)
from livifuser_nav.model import LiViFuserPolicy  # noqa: E402

RESULT_ARCHIVE_SHA256 = "F5B7D9EAB29DD20CE6710E4B803EAA331A5D7C2E741E9330995A1EAE615B9AC7"
SEEDS = (20260805, 20260806, 20260807)
HETEROSCEDASTIC_PARTITIONS = {
    worker: tuple(name for name in names if name != "full_mean_only")
    for worker, names in PARTITIONS.items()
}
CLOSED_LOOP_NAMES = {"full", "lidar_only", "concat", "rgb_only"}
REPRODUCTION_RTOL = 1e-6
REPRODUCTION_ATOL = 1e-7
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, indent=2) + chr(10)
    if path.exists():
        require(path.read_text(encoding="utf-8") == raw, f"existing JSON drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(raw, encoding="utf-8", newline=chr(10))
    temporary.replace(path)


def npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, npy_bytes(np.asarray(arrays[name])))
    return output.getvalue()


def write_bytes_verified(path: Path, payload: bytes) -> None:
    if path.exists():
        require(path.read_bytes() == payload, f"existing score artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def right_continuous_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_reference = np.sort(np.asarray(reference, dtype=np.float64))
    require(sorted_reference.ndim == 1 and sorted_reference.size > 0, "empty CDF")
    return (
        np.searchsorted(sorted_reference, np.asarray(values), side="right") / sorted_reference.size
    )


def episode_maxima(episode_ids: list[str], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value_array = np.asarray(values, dtype=np.float64)
    require(len(episode_ids) == value_array.size, "episode/value length mismatch")
    identities = np.asarray(sorted(set(episode_ids)), dtype=np.str_)
    maxima = np.asarray(
        [
            value_array[
                np.asarray([episode == identity for episode in episode_ids], dtype=bool)
            ].max()
            for identity in identities
        ],
        dtype=np.float64,
    )
    require(identities.size == 30, "expected 30 validation episodes")
    return identities, maxima


def operating_threshold(maxima: np.ndarray) -> tuple[float, int]:
    values = np.sort(np.asarray(maxima, dtype=np.float64))
    require(values.shape == (30,), "threshold requires 30 episode maxima")
    threshold = float(values[-2])
    false_interventions = int(np.count_nonzero(values > threshold))
    require(false_interventions <= 1, "validation false-intervention cap exceeded")
    return threshold, false_interventions


def validate_plan(plan: dict[str, Any]) -> None:
    require(
        plan.get("purpose") == "validation_score_freeze_only",
        "data plan purpose drift",
    )
    require(
        (
            plan.get("source_handoff_self_sha256"),
            plan.get("cache_manifest_sha256"),
            plan.get("cache_manifest_self_sha256"),
            plan.get("backbone_contract_sha256"),
        )
        == (
            HANDOFF_SELF_SHA256,
            CACHE_MANIFEST_SHA256,
            CACHE_SELF_SHA256,
            BACKBONE_CONTRACT_SHA256,
        ),
        "validation source identity drift",
    )
    require(plan.get("heldout_attached") is False, "data plan lacks held-out exclusion")
    require(
        plan.get("excluded_splits") == ["test_id", "test_ood"],
        "held-out split guard drift",
    )
    validation = plan["validation"]
    require(
        (
            int(validation["episode_count"]),
            int(validation["accepted_samples"]),
            int(validation["windows_k8_h8"]),
        )
        == (30, 13_125, 9_459),
        "validation plan count drift",
    )
    identities = [str(value) for value in validation["episode_ids"]]
    require(len(identities) == len(set(identities)) == 30, "validation identity drift")
    require(
        not any(value.startswith(("test_id_", "test_ood_")) for value in identities),
        "held-out episode entered validation plan",
    )
    require(
        len(validation["exports"]) == len(validation["caches"]) == 30,
        "validation path count drift",
    )


class ResultSource:
    """Read immutable result members from either the original ZIP or Kaggle expansion."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.archive = zipfile.ZipFile(self.path) if self.path.is_file() else None

    def read(self, member: str) -> bytes:
        require(member.startswith(ROOT), f"result member root drift: {member}")
        if self.archive is not None:
            return self.archive.read(member)
        relative = member.removeprefix(ROOT)
        target = self.path.joinpath(*relative.split("/"))
        require(target.is_file(), f"expanded result member missing: {member}")
        return target.read_bytes()

    def __enter__(self) -> ResultSource:
        return self

    def __exit__(self, *unused: object) -> None:
        if self.archive is not None:
            self.archive.close()


def validate_result_source(path: Path, audit_report: dict[str, Any]) -> dict[str, Any]:
    source = path.resolve()
    require(
        audit_report["archive"]["sha256"] == RESULT_ARCHIVE_SHA256,
        "audit report archive identity drift",
    )
    if source.is_file():
        require(
            sha256_file(source) == RESULT_ARCHIVE_SHA256,
            "result archive hash drift",
        )
        return {
            "mode": "original_zip",
            "path": str(source),
            "frozen_archive_sha256": RESULT_ARCHIVE_SHA256,
        }
    require(source.is_dir(), f"result source is missing: {source}")
    require(source.name == ROOT.rstrip("/"), "expanded result root name drift")
    expected = {str(name): str(digest) for name, digest in audit_report["member_sha256"].items()}
    observed = {
        ROOT + file.relative_to(source).as_posix() for file in source.rglob("*") if file.is_file()
    }
    require(observed == set(expected), "expanded result exact file set drift")
    for member, expected_sha256 in sorted(expected.items()):
        relative = member.removeprefix(ROOT)
        target = source.joinpath(*relative.split("/"))
        require(
            sha256_file(target) == expected_sha256,
            f"expanded result member hash drift: {member}",
        )
    return {
        "mode": "kaggle_expanded_directory",
        "path": str(source),
        "file_count": len(observed),
        "all_member_sha256_verified": True,
        "frozen_archive_sha256": RESULT_ARCHIVE_SHA256,
    }


def result_member(name: str, seed: int) -> str:
    workers = [worker for worker, names in PARTITIONS.items() if name in names]
    require(len(workers) == 1, f"run partition lookup failed: {name}")
    return ROOT + f"worker_{workers[0]}/{name}/seed_{seed}/result.json"


def checkpoint_member(name: str, seed: int) -> str:
    return result_member(name, seed).removesuffix("result.json") + "checkpoint.pt"


def max_nested_difference(left: Any, right: Any) -> float:
    if isinstance(left, dict):
        require(set(left) == set(right), "nested metric key drift")
        return max(max_nested_difference(left[key], right[key]) for key in left)
    return abs(float(left) - float(right))


def replay_one(
    name: str,
    seed: int,
    config: dict[str, Any],
    dataset: Any,
    caches: list[Any],
    tokens: list[Any],
    device: torch.device,
    archive: ResultSource,
    audit_report: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    result_payload = archive.read(result_member(name, seed))
    stored = json.loads(result_payload)
    run_config = stored["run"]
    require(run_config["name"] == name, f"result name drift: {name}:{seed}")
    require(
        run_config["loss"] == "heteroscedastic",
        f"uncertainty replay requires heteroscedastic head: {name}:{seed}",
    )
    checkpoint_payload = archive.read(checkpoint_member(name, seed))
    checkpoint_sha256 = sha256_bytes(checkpoint_payload)
    identity_key = f"{name}:{seed}"
    require(
        checkpoint_sha256 == audit_report["checkpoint_sha256"][identity_key],
        f"checkpoint audit hash drift: {identity_key}",
    )
    require(
        sha256_bytes(result_payload) == audit_report["result_sha256"][identity_key],
        f"result audit hash drift: {identity_key}",
    )
    checkpoint = torch.load(
        io.BytesIO(checkpoint_payload),
        map_location="cpu",
        weights_only=True,
    )
    require(
        checkpoint["variant"] == run_config["variant"]
        and checkpoint["seed"] == seed
        and checkpoint["config_sha256"] == CONFIG_SHA256,
        f"checkpoint identity drift: {identity_key}",
    )
    model = LiViFuserPolicy(variant=run_config["variant"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    evaluation = evaluate_model(model, dataset, caches, tokens, config, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    episode_ids = [str(value) for value in evaluation["episode_ids"]]
    origin_rows = [int(value) for value in evaluation["origin_rows"]]
    stored_window = stored["validation"]["per_window"]
    require(
        episode_ids == stored_window["episode_ids"] and origin_rows == stored_window["origin_rows"],
        f"validation window identity reproduction failed: {identity_key}",
    )
    mean = np.asarray(evaluation["mean"], dtype=np.float64)
    log_variance = np.asarray(evaluation["log_variance"], dtype=np.float64)
    target = np.asarray(evaluation["target"], dtype=np.float64)
    mse = window_normalized_mse(mean, target)
    nll = window_nll(mean, log_variance, target)
    stored_mse = np.asarray(stored_window["normalized_mse"], dtype=np.float64)
    stored_nll = np.asarray(stored_window["nll"], dtype=np.float64)
    require(
        np.allclose(
            mse,
            stored_mse,
            rtol=REPRODUCTION_RTOL,
            atol=REPRODUCTION_ATOL,
        ),
        f"MSE reproduction failed: {identity_key}",
    )
    require(
        np.allclose(
            nll,
            stored_nll,
            rtol=REPRODUCTION_RTOL,
            atol=REPRODUCTION_ATOL,
        ),
        f"NLL reproduction failed: {identity_key}",
    )
    recomputed_coverage = sigma_coverage(mean, log_variance, target)
    coverage_difference = max_nested_difference(
        recomputed_coverage, stored["validation"]["sigma_coverage"]
    )
    require(
        coverage_difference <= REPRODUCTION_ATOL,
        f"sigma-coverage reproduction failed: {identity_key}",
    )

    clamped = np.clip(
        log_variance,
        LOG_VARIANCE_CLAMP[0],
        LOG_VARIANCE_CLAMP[1],
    )
    variance = np.exp(clamped)
    sigma = np.exp(0.5 * clamped)
    aleatoric = variance.mean(axis=(1, 2))
    max_sigma = sigma.max(axis=(1, 2))
    first_step_max_sigma = sigma[:, 0, :].max(axis=1)
    mahalanobis = np.asarray(
        stored_window["mahalanobis_distance"],
        dtype=np.float64,
    )
    z_a = right_continuous_cdf(aleatoric, aleatoric)
    z_m = right_continuous_cdf(mahalanobis, mahalanobis)
    combined = np.maximum(z_a, z_m)
    unique_episodes, aleatoric_episode_max = episode_maxima(episode_ids, z_a)
    mahalanobis_episodes, mahalanobis_episode_max = episode_maxima(episode_ids, z_m)
    combined_episodes, combined_episode_max = episode_maxima(episode_ids, combined)
    require(
        np.array_equal(unique_episodes, mahalanobis_episodes)
        and np.array_equal(unique_episodes, combined_episodes),
        f"episode maximum identities drift: {identity_key}",
    )
    thresholds: dict[str, float] = {}
    false_interventions: dict[str, int] = {}
    for score_name, maxima in (
        ("aleatoric", aleatoric_episode_max),
        ("mahalanobis", mahalanobis_episode_max),
        ("combined", combined_episode_max),
    ):
        threshold, count = operating_threshold(maxima)
        thresholds[score_name] = threshold
        false_interventions[score_name] = count

    arrays = {
        "episode_ids": np.asarray(episode_ids, dtype=np.str_),
        "origin_rows": np.asarray(origin_rows, dtype=np.int64),
        "aleatoric_variance": aleatoric,
        "max_sigma": max_sigma,
        "first_step_max_sigma": first_step_max_sigma,
        "aleatoric_cdf_sorted": np.sort(aleatoric),
        "mahalanobis_distance": mahalanobis,
        "mahalanobis_cdf_sorted": np.sort(mahalanobis),
        "z_a": z_a,
        "z_m": z_m,
        "combined": combined,
        "episode_ids_unique": unique_episodes,
        "aleatoric_episode_max": aleatoric_episode_max,
        "mahalanobis_episode_max": mahalanobis_episode_max,
        "combined_episode_max": combined_episode_max,
    }
    score_payload = deterministic_npz(arrays)
    relative = Path("scores") / f"{name}_seed_{seed}.npz"
    score_path = output_root / relative
    write_bytes_verified(score_path, score_payload)
    return {
        "name": name,
        "variant": run_config["variant"],
        "loss": run_config["loss"],
        "seed": seed,
        "closed_loop_shortlist": name in CLOSED_LOOP_NAMES,
        "checkpoint_sha256": checkpoint_sha256,
        "result_sha256": sha256_bytes(result_payload),
        "score_file": relative.as_posix(),
        "score_size_bytes": len(score_payload),
        "score_sha256": sha256_bytes(score_payload),
        "validation_windows": len(episode_ids),
        "validation_episodes": int(unique_episodes.size),
        "threshold_rule": "second_largest_of_30_episode_maxima; strict_greater_than",
        "thresholds": thresholds,
        "validation_false_interventions": false_interventions,
        "reproduction": {
            "rtol": REPRODUCTION_RTOL,
            "atol": REPRODUCTION_ATOL,
            "mse_max_abs_difference": float(np.max(np.abs(mse - stored_mse))),
            "nll_max_abs_difference": float(np.max(np.abs(nll - stored_nll))),
            "sigma_coverage_max_abs_difference": coverage_difference,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-plan", type=Path, required=True)
    parser.add_argument("--results-archive", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-name", action="append", required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    require(sha256_file(config_path) == CONFIG_SHA256, "frozen config hash drift")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(args.data_plan.resolve().read_text(encoding="utf-8"))
    validate_plan(plan)
    audit_report = json.loads(args.audit_report.resolve().read_text(encoding="utf-8"))
    require(
        audit_report["status"] == "PASS"
        and audit_report["archive"]["sha256"] == RESULT_ARCHIVE_SHA256
        and audit_report["frozen_provenance"]["config_sha256"] == CONFIG_SHA256,
        "result audit report drift",
    )
    result_source_path = args.results_archive.resolve()
    result_source = validate_result_source(result_source_path, audit_report)
    names = [str(value) for value in args.run_name]
    allowed = {name for partition in HETEROSCEDASTIC_PARTITIONS.values() for name in partition}
    require(
        len(names) == len(set(names)) and set(names).issubset(allowed),
        "replay run-name selection drift",
    )
    device = resolve_device(args.device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(int(config["torch_threads"]))
    validation = plan["validation"]
    dataset, caches, tokens = load_split(
        [Path(value) for value in validation["exports"]],
        [Path(value) for value in validation["caches"]],
        config,
        "validation",
    )
    require(len(dataset.windows) == 9459, "loaded validation window count drift")
    require(common_cache_identity(caches) == config["backbone"], "backbone drift")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with ResultSource(result_source_path) as archive:
        for name in names:
            for seed in SEEDS:
                print(f"replaying {name} seed {seed} ...", flush=True)
                record = replay_one(
                    name,
                    seed,
                    config,
                    dataset,
                    caches,
                    tokens,
                    device,
                    archive,
                    audit_report,
                    output,
                )
                records.append(record)
                print(
                    f"  {name} seed {seed}: "
                    f"aleatoric threshold {record['thresholds']['aleatoric']:.6f}, "
                    f"combined threshold {record['thresholds']['combined']:.6f}",
                    flush=True,
                )
    summary = {
        "schema_version": 1,
        "config_sha256": CONFIG_SHA256,
        "result_archive_sha256": RESULT_ARCHIVE_SHA256,
        "result_source": result_source,
        "data_plan_sha256": sha256_file(args.data_plan.resolve()),
        "device": str(device),
        "run_names": names,
        "record_count": len(records),
        "records": records,
    }
    write_json_atomic(output / "worker_score_summary.json", summary)
    print(json.dumps({"output": str(output), "records": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
