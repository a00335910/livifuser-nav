#!/usr/bin/env python3
"""Package the exact frozen simulation training code for a Kaggle dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SHA256 = "76680BCE45A67D5D91F42660D5EC25F90450B281838CE311E329477C4E36F09E"
REQUIRED = (
    Path("config/simulation_sweep_v1.json"),
    Path("scripts/prepare_sim_training_data.py"),
    Path("scripts/run_baseline_sweep.py"),
    Path("scripts/run_simulation_sweep_kaggle.py"),
)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = REPOSITORY_ROOT / "config" / "simulation_sweep_v1.json"
    if sha256_file(config) != CONFIG_SHA256:
        raise RuntimeError("frozen simulation config hash mismatch")
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
        / f"livifuser_sim_training_code_{CONFIG_SHA256[:12].lower()}.zip"
    ).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for relative in selected_files():
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "purpose": "Frozen LiViFuser confirmatory-v3 simulation training code",
        "git_revision": revision,
        "source_state": "exact per-file hash manifest; unrelated worktree files excluded",
        "frozen_config_sha256": CONFIG_SHA256,
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "files": entries,
    }
    manifest_payload = json.dumps(manifest, indent=2) + "\n"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for entry in entries:
            relative = Path(entry["path"])
            archive.write(REPOSITORY_ROOT / relative, (Path("LiViFuser") / relative).as_posix())
        archive.writestr("LiViFuser/cloud_bundle_manifest.json", manifest_payload)
    report = {
        "bundle": str(output),
        "bundle_size_bytes": output.stat().st_size,
        "bundle_sha256": sha256_file(output),
        "manifest_sha256": hashlib.sha256(manifest_payload.encode()).hexdigest().upper(),
        "frozen_config_sha256": CONFIG_SHA256,
        "file_count": len(entries),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
