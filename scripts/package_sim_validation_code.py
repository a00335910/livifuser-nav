#!/usr/bin/env python3
"""Package the frozen validation score-replay code for a Kaggle dataset."""

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
AMENDMENT_SHA256 = "8760474F1CCC6269BD23A28489DD01076891ECBF9E66A6F39BBF8E2838F6DCD7"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
REQUIRED = (
    Path("artifacts/simulation_sweep_v1_result_audit.json"),
    Path("config/simulation_sweep_v1.json"),
    Path("docs/experiments/PREREGISTRATION_SIM_EVALUATION_EXECUTION_AMENDMENT_2026-08-24.md"),
    Path("scripts/audit_simulation_sweep_results.py"),
    Path("scripts/prepare_sim_training_data.py"),
    Path("scripts/replay_sim_validation_scores.py"),
    Path("scripts/run_baseline_sweep.py"),
    Path("scripts/run_sim_validation_score_freeze_kaggle.py"),
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
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
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
    config = REPOSITORY_ROOT / "config" / "simulation_sweep_v1.json"
    amendment = REPOSITORY_ROOT / REQUIRED[2]
    if sha256_file(config) != CONFIG_SHA256:
        raise RuntimeError("frozen simulation config hash mismatch")
    if sha256_file(amendment) != AMENDMENT_SHA256:
        raise RuntimeError("frozen evaluation amendment hash mismatch")
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
        / f"livifuser_sim_validation_code_{AMENDMENT_SHA256[:12].lower()}.zip"
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
        "purpose": "Frozen validation-only simulation uncertainty score replay",
        "git_revision": revision,
        "source_state": "exact per-file hash manifest; unrelated worktree files excluded",
        "frozen_config_sha256": CONFIG_SHA256,
        "frozen_amendment_sha256": AMENDMENT_SHA256,
        "heldout_code_or_identity_included": False,
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
