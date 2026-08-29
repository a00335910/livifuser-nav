#!/usr/bin/env python3
"""Run one immutable partition of the approved simulation held-out evaluation."""

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
from prepare_sim_heldout_data import (  # noqa: E402
    BACKBONE_CONTRACT_SHA256,
    CACHE_BUNDLE_SHA256,
    CACHE_MANIFEST_SHA256,
    CACHE_SELF_SHA256,
    HANDOFF_SELF_SHA256,
)
from replay_sim_validation_scores import (  # noqa: E402
    CLOSED_LOOP_NAMES,
    SEEDS,
    ResultSource,
    checkpoint_member,
    deterministic_npz,
    result_member,
    validate_result_source,
)
from run_baseline_sweep import (  # noqa: E402
    common_cache_identity,
    evaluate_model,
    load_split,
    resolve_device,
)

from livifuser_nav.evaluation import (  # noqa: E402
    LOG_VARIANCE_CLAMP,
    mahalanobis_distances,
    normalized_error,
    window_nll,
    window_normalized_mse,
)
from livifuser_nav.heldout_evaluation import right_continuous_cdf  # noqa: E402
from livifuser_nav.model import LiViFuserPolicy  # noqa: E402

SCORE_BUNDLE_SHA256 = "07116A629E296929D69EDA41E44CB6067CB6C751C735B66FD0A1B736D240751B"
SCORE_MANIFEST_FILE_SHA256 = "BFD5A21F150DCFCF12CD988821DE6901A1558ACC0EA183D7F8223940C2C1A729"
SCORE_MANIFEST_SELF_SHA256 = "AA90B540579C8285F55422DB41EA549305FA6C755FDF391FA3CD06ADC82127BF"
RESULT_AUDIT_SHA256 = "4D1CEA8F2D61EF76E1A48770FB6228F14683DAF6943C4932C06FCE0FB46611B3"
SCORE_AUDIT_SHA256 = "A9071ABF41F25B0FA68209B2AFB94242F33B83E4330DF6B9B563A5FA6ADA3E97"
EXPECTED_WINDOWS = 34_503
EXPECTED_EPISODES = 110
TRAINING_MEAN = np.asarray((0.047447296062892504, -0.005205255475722954), dtype=np.float64)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_npz_bytes(payload: bytes, label: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        require(
            all(archive[name].dtype.kind != "O" for name in archive.files), f"object array: {label}"
        )
        return {name: archive[name] for name in archive.files}


class ScoreSource:
    """Read and fully verify the score freeze from its ZIP or Kaggle expansion."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.archive = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.archive is not None:
            require(sha256_file(self.path) == SCORE_BUNDLE_SHA256, "score ZIP hash drift")
            infos = [info for info in self.archive.infolist() if not info.is_dir()]
            names = [info.filename for info in infos]
            require(len(names) == len(set(names)) == 23, "score ZIP member set drift")
            require(self.archive.testzip() is None, "score ZIP CRC failure")
        else:
            require(self.path.is_dir(), f"score source missing: {self.path}")
            names = [
                file.relative_to(self.path).as_posix()
                for file in self.path.rglob("*")
                if file.is_file()
            ]
            require(len(names) == len(set(names)) == 23, "expanded score member set drift")
        self.names = set(names)
        manifest_raw = self.read("SCORE_FREEZE_MANIFEST.json")
        require(
            sha256_bytes(manifest_raw) == SCORE_MANIFEST_FILE_SHA256, "score manifest hash drift"
        )
        self.manifest = json.loads(manifest_raw)
        require(
            self.manifest["manifest_sha256_excludes_self"] == SCORE_MANIFEST_SELF_SHA256,
            "score manifest self-hash identity drift",
        )
        member_rows = {row["name"]: row for row in self.manifest["members"]}
        expected = {*member_rows, "SCORE_FREEZE_MANIFEST.json", "SCORE_FREEZE_COMPLETE.json"}
        require(self.names == expected and len(member_rows) == 21, "score exact set drift")
        for name, record in member_rows.items():
            payload = self.read(name)
            require(
                len(payload) == int(record["size_bytes"])
                and sha256_bytes(payload) == record["sha256"],
                f"score member hash drift: {name}",
            )
        completion = json.loads(self.read("SCORE_FREEZE_COMPLETE.json"))
        require(
            completion["status"] == "COMPLETE"
            and completion["manifest_file_sha256"] == SCORE_MANIFEST_FILE_SHA256
            and completion["manifest_sha256_excludes_self"] == SCORE_MANIFEST_SELF_SHA256
            and completion["exact_bundle_member_count"] == 23,
            "score completion marker drift",
        )
        self.records = {
            (str(row["name"]), int(row["seed"])): row for row in self.manifest["records"]
        }
        require(len(self.records) == 21, "score record identity drift")

    def read(self, name: str) -> bytes:
        if self.archive is not None:
            return self.archive.read(name)
        target = self.path.joinpath(*name.split("/"))
        require(target.is_file(), f"expanded score member missing: {name}")
        return target.read_bytes()

    def score_arrays(self, name: str, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        record = self.records[(name, seed)]
        return load_npz_bytes(self.read(record["score_file"]), f"{name}:{seed}"), record

    def __enter__(self) -> ScoreSource:
        return self

    def __exit__(self, *unused: object) -> None:
        if self.archive is not None:
            self.archive.close()


def validate_plan(plan: dict[str, Any]) -> None:
    require(
        plan.get("purpose") == "approved_one_time_simulation_heldout_evaluation",
        "plan purpose drift",
    )
    require(
        (
            plan.get("source_handoff_self_sha256"),
            plan.get("cache_transport_sha256"),
            plan.get("cache_manifest_sha256"),
            plan.get("cache_manifest_self_sha256"),
            plan.get("backbone_contract_sha256"),
        )
        == (
            HANDOFF_SELF_SHA256,
            CACHE_BUNDLE_SHA256,
            CACHE_MANIFEST_SHA256,
            CACHE_SELF_SHA256,
            BACKBONE_CONTRACT_SHA256,
        ),
        "held-out plan frozen identity drift",
    )
    require(plan.get("heldout_attached") is True, "held-out plan is not attached")
    require(plan.get("allowed_splits") == ["test_id", "test_ood"], "held-out split contract drift")
    require(
        (int(plan["episode_count"]), int(plan["accepted_samples"]), int(plan["windows_k8_h8"]))
        == (EXPECTED_EPISODES, 47_326, EXPECTED_WINDOWS),
        "held-out plan count drift",
    )
    episodes = plan["episodes"]
    require(len(episodes) == 110, "held-out episode record count drift")
    require(
        [int(row["ordinal"]) for row in episodes] == list(range(150, 260)),
        "held-out plan ordering drift",
    )
    require(
        {row["condition"] for row in episodes} == {"C0", "C1", "C3b", "C4"},
        "held-out condition labels drift",
    )


def load_gaussian(
    results: ResultSource,
    result_audit: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    names = (
        ROOT + "worker_0/mahalanobis_mean.npy",
        ROOT + "worker_0/mahalanobis_precision.npy",
    )
    payloads = [results.read(name) for name in names]
    require(
        sha256_bytes(payloads[0]) == result_audit["mahalanobis"]["mean_sha256"]
        and sha256_bytes(payloads[1]) == result_audit["mahalanobis"]["precision_sha256"],
        "Gaussian hash drift",
    )
    mean = np.load(io.BytesIO(payloads[0]), allow_pickle=False)
    precision = np.load(io.BytesIO(payloads[1]), allow_pickle=False)
    require(mean.shape == (384,) and precision.shape == (384, 384), "Gaussian shape drift")
    return mean, precision


def build_common(
    plan: dict[str, Any],
    dataset: Any,
    caches: list[Any],
    gaussian_mean: np.ndarray,
    gaussian_precision: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    metadata = {str(row["episode_id"]): row for row in plan["episodes"]}
    episode_ids = []
    origin_rows = []
    targets = []
    pooled = []
    repeat = []
    splits = []
    worlds = []
    conditions = []
    episode_indices = []
    observation_seeds = []
    for ref in dataset.windows:
        run = dataset.runs[ref.run_index]
        identity = run.run_id
        row = metadata[identity]
        require(
            ref.origin_row > 0 and ref.context_rows[-2] == ref.origin_row - 1,
            "repeat-last alignment drift",
        )
        episode_ids.append(identity)
        origin_rows.append(ref.origin_row)
        targets.append(dataset.targets(ref).astype(np.float32))
        pooled.append(
            np.asarray(caches[ref.run_index].pooled_features[ref.origin_row], dtype=np.float64)
        )
        previous = np.asarray(run.vectors["action"][ref.origin_row - 1], dtype=np.float64)
        repeat.append(np.repeat(previous[None, :], 8, axis=0))
        splits.append(str(row["split"]))
        worlds.append(str(row["world_name"]))
        conditions.append(str(row["condition"]))
        episode_indices.append(int(row["episode_index"]))
        observation_seeds.append(int(row["observation_seed"]))
    require(len(episode_ids) == EXPECTED_WINDOWS, "loaded held-out window count drift")
    target = np.asarray(targets, dtype=np.float32)
    repeat_mean = np.asarray(repeat, dtype=np.float64)
    constant_mean = np.broadcast_to(TRAINING_MEAN, target.shape).copy()
    mahalanobis = mahalanobis_distances(
        np.asarray(pooled, dtype=np.float64), gaussian_mean, gaussian_precision
    )
    common = {
        "episode_ids": np.asarray(episode_ids, dtype=np.str_),
        "splits": np.asarray(splits, dtype=np.str_),
        "worlds": np.asarray(worlds, dtype=np.str_),
        "conditions": np.asarray(conditions, dtype=np.str_),
        "episode_indices": np.asarray(episode_indices, dtype=np.int64),
        "observation_seeds": np.asarray(observation_seeds, dtype=np.int64),
        "origin_rows": np.asarray(origin_rows, dtype=np.int64),
        "target": target,
        "mahalanobis_distance": mahalanobis,
    }
    repeat_squared = np.square(normalized_error(repeat_mean, target))
    constant_squared = np.square(normalized_error(constant_mean, target))
    trivial = {
        "repeat_last_mean": repeat_mean,
        "repeat_last_normalized_mse": repeat_squared.mean(axis=(1, 2)),
        "repeat_last_per_horizon_squared_error": repeat_squared.mean(axis=2),
        "constant_training_mean": constant_mean,
        "constant_training_mean_normalized_mse": constant_squared.mean(axis=(1, 2)),
        "constant_training_mean_per_horizon_squared_error": constant_squared.mean(axis=2),
    }
    return common, trivial


def completed_record(
    output_root: Path,
    name: str,
    seed: int,
    checkpoint_sha256: str,
) -> dict[str, Any] | None:
    record_path = output_root / "records" / f"{name}_seed_{seed}.json"
    if not record_path.exists():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    require(
        record["name"] == name
        and int(record["seed"]) == seed
        and record["checkpoint_sha256"] == checkpoint_sha256,
        f"completed record identity drift: {name}:{seed}",
    )
    prediction = output_root / record["prediction_file"]
    require(
        prediction.is_file()
        and prediction.stat().st_size == int(record["prediction_size_bytes"])
        and sha256_file(prediction) == record["prediction_sha256"],
        f"completed prediction integrity drift: {name}:{seed}",
    )
    return record


def write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        require(path.read_bytes() == payload, f"immutable output drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    require(not temporary.exists(), f"refusing stale partial output: {temporary}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def evaluate_one(
    name: str,
    seed: int,
    config: dict[str, Any],
    dataset: Any,
    caches: list[Any],
    tokens: list[Any],
    common: dict[str, np.ndarray],
    device: torch.device,
    results: ResultSource,
    result_audit: dict[str, Any],
    scores: ScoreSource,
    output_root: Path,
) -> dict[str, Any]:
    identity = f"{name}:{seed}"
    result_payload = results.read(result_member(name, seed))
    checkpoint_payload = results.read(checkpoint_member(name, seed))
    checkpoint_sha256 = sha256_bytes(checkpoint_payload)
    require(
        checkpoint_sha256 == result_audit["checkpoint_sha256"][identity]
        and sha256_bytes(result_payload) == result_audit["result_sha256"][identity],
        f"result/checkpoint audit binding drift: {identity}",
    )
    existing = completed_record(output_root, name, seed, checkpoint_sha256)
    if existing is not None:
        print(f"skipping immutable completed {name} seed {seed}", flush=True)
        return existing
    stored = json.loads(result_payload)
    run = stored["run"]
    checkpoint = torch.load(io.BytesIO(checkpoint_payload), map_location="cpu", weights_only=True)
    require(
        run["name"] == name
        and int(stored["seed"]) == seed
        and checkpoint["variant"] == run["variant"]
        and int(checkpoint["seed"]) == seed
        and checkpoint["config_sha256"] == CONFIG_SHA256,
        f"checkpoint semantic identity drift: {identity}",
    )
    model = LiViFuserPolicy(variant=run["variant"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    evaluation = evaluate_model(model, dataset, caches, tokens, config, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    require(
        np.array_equal(np.asarray(evaluation["episode_ids"], dtype=np.str_), common["episode_ids"])
        and np.array_equal(
            np.asarray(evaluation["origin_rows"], dtype=np.int64), common["origin_rows"]
        )
        and np.array_equal(np.asarray(evaluation["target"], dtype=np.float32), common["target"]),
        f"held-out target alignment drift: {identity}",
    )
    mean = np.asarray(evaluation["mean"], dtype=np.float32)
    log_variance = np.asarray(evaluation["log_variance"], dtype=np.float32)
    target = common["target"]
    squared = np.square(normalized_error(mean, target))
    arrays: dict[str, np.ndarray] = {
        "mean": mean,
        "log_variance": log_variance,
        "normalized_mse": window_normalized_mse(mean, target),
        "per_horizon_squared_error": squared.mean(axis=2),
    }
    if run["loss"] == "heteroscedastic":
        score_arrays, score_record = scores.score_arrays(name, seed)
        clamped = np.clip(log_variance.astype(np.float64), *LOG_VARIANCE_CLAMP)
        sigma = np.exp(0.5 * clamped)
        aleatoric = np.exp(clamped).mean(axis=(1, 2))
        z_a = right_continuous_cdf(score_arrays["aleatoric_cdf_sorted"], aleatoric)
        z_m = right_continuous_cdf(
            score_arrays["mahalanobis_cdf_sorted"], common["mahalanobis_distance"]
        )
        arrays.update(
            {
                "nll": window_nll(mean, log_variance, target),
                "aleatoric_variance": aleatoric,
                "max_sigma": sigma.max(axis=(1, 2)),
                "first_step_max_sigma": sigma[:, 0, :].max(axis=1),
                "z_a": z_a,
                "z_m": z_m,
                "combined": np.maximum(z_a, z_m),
            }
        )
        score_sha256 = score_record["score_sha256"]
        thresholds = score_record["thresholds"] if name in CLOSED_LOOP_NAMES else None
    else:
        require(run["loss"] == "mean_only" and name == "full_mean_only", "loss contract drift")
        score_sha256 = None
        thresholds = None
    require(
        all(np.all(np.isfinite(value)) for value in arrays.values()),
        f"non-finite prediction: {identity}",
    )
    payload = deterministic_npz(arrays)
    relative = Path("predictions") / f"{name}_seed_{seed}.npz"
    write_immutable(output_root / relative, payload)
    record = {
        "name": name,
        "variant": run["variant"],
        "loss": run["loss"],
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
        "result_sha256": sha256_bytes(result_payload),
        "validation_score_sha256": score_sha256,
        "closed_loop_shortlist": name in CLOSED_LOOP_NAMES,
        "thresholds": thresholds,
        "prediction_file": relative.as_posix(),
        "prediction_size_bytes": len(payload),
        "prediction_sha256": sha256_bytes(payload),
        "window_count": EXPECTED_WINDOWS,
    }
    record_payload = (json.dumps(record, indent=2) + chr(10)).encode()
    write_immutable(output_root / "records" / f"{name}_seed_{seed}.json", record_payload)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-plan", type=Path, required=True)
    parser.add_argument("--results-source", type=Path, required=True)
    parser.add_argument("--result-audit", type=Path, required=True)
    parser.add_argument("--score-source", type=Path, required=True)
    parser.add_argument("--score-audit", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-name", action="append", required=True)
    args = parser.parse_args()
    require(sha256_file(args.config.resolve()) == CONFIG_SHA256, "config hash drift")
    require(
        sha256_file(args.result_audit.resolve()) == RESULT_AUDIT_SHA256, "result audit hash drift"
    )
    require(sha256_file(args.score_audit.resolve()) == SCORE_AUDIT_SHA256, "score audit hash drift")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    plan = json.loads(args.data_plan.read_text(encoding="utf-8"))
    validate_plan(plan)
    result_audit = json.loads(args.result_audit.read_text(encoding="utf-8"))
    score_audit = json.loads(args.score_audit.read_text(encoding="utf-8"))
    require(result_audit["status"] == score_audit["status"] == "PASS", "input audit is not PASS")
    result_source = args.results_source.resolve()
    validate_result_source(result_source, result_audit)
    names = [str(value) for value in args.run_name]
    allowed = {name for partition in PARTITIONS.values() for name in partition}
    require(len(names) == len(set(names)) and set(names).issubset(allowed), "run partition drift")
    device = resolve_device(args.device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(int(config["torch_threads"]))
    paths = plan["episodes"]
    dataset, caches, tokens = load_split(
        [Path(row["export"]) for row in paths],
        [Path(row["cache"]) for row in paths],
        config,
        "heldout",
    )
    require(len(dataset.windows) == EXPECTED_WINDOWS, "loaded held-out windows drift")
    require(common_cache_identity(caches) == config["backbone"], "held-out backbone drift")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with ResultSource(result_source) as results, ScoreSource(args.score_source) as scores:
        gaussian_mean, gaussian_precision = load_gaussian(results, result_audit)
        common, trivial = build_common(plan, dataset, caches, gaussian_mean, gaussian_precision)
        common_payload = deterministic_npz(common)
        trivial_payload = deterministic_npz(trivial)
        write_immutable(output / "worker_common.npz", common_payload)
        write_immutable(output / "worker_trivial_baselines.npz", trivial_payload)
        records = []
        for name in names:
            for seed in SEEDS:
                print(f"evaluating {name} seed {seed} ...", flush=True)
                records.append(
                    evaluate_one(
                        name,
                        seed,
                        config,
                        dataset,
                        caches,
                        tokens,
                        common,
                        device,
                        results,
                        result_audit,
                        scores,
                        output,
                    )
                )
                print(f"completed {name} seed {seed}", flush=True)
    summary = {
        "schema_version": 1,
        "run_names": names,
        "record_count": len(records),
        "common_sha256": sha256_bytes(common_payload),
        "trivial_sha256": sha256_bytes(trivial_payload),
        "records": records,
    }
    payload = (json.dumps(summary, indent=2) + chr(10)).encode()
    write_immutable(output / "worker_summary.json", payload)
    print(json.dumps({"output": str(output), "records": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
