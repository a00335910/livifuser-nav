#!/usr/bin/env python3
"""Package the frozen held-out evaluator as a deterministic Kaggle dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SHA256 = "76680BCE45A67D5D91F42660D5EC25F90450B281838CE311E329477C4E36F09E"
AMENDMENT_SHA256 = "2CD7ADE1AC43FBC74975D9987E6C6052F5146B9FAD4CB97B1D46BEC996F4EE55"
REPAIR_SHA256 = "EB19516B2D84D4830A7A34B7EDB56DFBACE7E8C8E17866AEB9605B9929AC9357"
RESULT_AUDIT_SHA256 = "4D1CEA8F2D61EF76E1A48770FB6228F14683DAF6943C4932C06FCE0FB46611B3"
SCORE_AUDIT_SHA256 = "A9071ABF41F25B0FA68209B2AFB94242F33B83E4330DF6B9B563A5FA6ADA3E97"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
REQUIRED = (
    Path("artifacts/sim_validation_score_freeze_v1_audit.json"),
    Path("artifacts/simulation_sweep_v1_result_audit.json"),
    Path("config/simulation_sweep_v1.json"),
    Path(
        "docs/experiments/PREREGISTRATION_SIM_HELDOUT_EVALUATION_EXECUTION_AMENDMENT_2026-08-24.md"
    ),
    Path("docs/experiments/PREREGISTRATION_SIM_HELDOUT_EVALUATION_EXECUTION_REPAIR_2026-08-24.md"),
    Path("scripts/audit_simulation_sweep_results.py"),
    Path("scripts/evaluate_sim_heldout.py"),
    Path("scripts/package_sim_heldout_evaluation_code.py"),
    Path("scripts/prepare_sim_heldout_data.py"),
    Path("scripts/prepare_sim_training_data.py"),
    Path("scripts/replay_sim_validation_scores.py"),
    Path("scripts/run_baseline_sweep.py"),
    Path("scripts/run_sim_heldout_evaluation_kaggle.py"),
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def selected_files() -> list[Path]:
    files = set(REQUIRED)
    files.update(
        path.relative_to(REPOSITORY_ROOT)
        for path in (REPOSITORY_ROOT / "src" / "livifuser_nav").glob("*.py")
    )
    return sorted(files, key=lambda path: path.as_posix())


def deterministic_zip(members: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, members[name])
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fixed = {
        Path("config/simulation_sweep_v1.json"): CONFIG_SHA256,
        REQUIRED[3]: AMENDMENT_SHA256,
        REQUIRED[4]: REPAIR_SHA256,
        REQUIRED[1]: RESULT_AUDIT_SHA256,
        REQUIRED[0]: SCORE_AUDIT_SHA256,
    }
    for relative, expected in fixed.items():
        if sha256_file(REPOSITORY_ROOT / relative) != expected:
            raise RuntimeError(f"frozen file hash mismatch: {relative}")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output = (
        args.output
        or REPOSITORY_ROOT
        / "artifacts"
        / "cloud"
        / f"livifuser_sim_heldout_eval_code_{AMENDMENT_SHA256[:12].lower()}_r3.zip"
    ).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    entries = []
    members: dict[str, bytes] = {}
    for relative in selected_files():
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        entries.append(
            {
                "path": relative.as_posix(),
                "size_bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
        members[(Path("LiViFuser") / relative).as_posix()] = payload
    manifest = {
        "schema_version": 1,
        "purpose": "Approved one-time simulation held-out offline evaluation",
        "git_revision": revision,
        "source_state": ("exact per-file hash manifest; unrelated worktree files excluded"),
        "frozen_config_sha256": CONFIG_SHA256,
        "frozen_amendment_sha256": AMENDMENT_SHA256,
        "execution_repair_sha256": REPAIR_SHA256,
        "result_audit_sha256": RESULT_AUDIT_SHA256,
        "score_freeze_audit_sha256": SCORE_AUDIT_SHA256,
        "heldout_feature_or_outcome_included": False,
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "files": entries,
    }
    manifest_payload = (json.dumps(manifest, indent=2) + chr(10)).encode()
    members["LiViFuser/cloud_bundle_manifest.json"] = manifest_payload
    bundle_payload = deterministic_zip(members)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bundle_payload)
    report = {
        "bundle": str(output),
        "bundle_size_bytes": len(bundle_payload),
        "bundle_sha256": sha256_bytes(bundle_payload),
        "manifest_sha256": sha256_bytes(manifest_payload),
        "frozen_config_sha256": CONFIG_SHA256,
        "frozen_amendment_sha256": AMENDMENT_SHA256,
        "execution_repair_sha256": REPAIR_SHA256,
        "file_count": len(entries),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2) + chr(10),
        encoding="utf-8",
        newline=chr(10),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
