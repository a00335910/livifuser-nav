#!/usr/bin/env python3
"""Create the minimal hash-manifested source/export/cache Kaggle upload."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from livifuser_nav.cloud_bundle import sha256_file  # noqa: E402

DATA_ROOTS = (
    Path("artifacts/export/protocol_clean_30"),
    Path("artifacts/features/protocol_clean_30"),
)


def tracked_files() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return [Path(item.decode("utf-8")) for item in output.split(b"\0") if item]


def require_clean_revision() -> str:
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("refusing to bundle a dirty worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def selected_files() -> list[Path]:
    selected = set(tracked_files())
    for root in DATA_ROOTS:
        absolute = REPOSITORY_ROOT / root
        if not absolute.is_dir():
            raise FileNotFoundError(absolute)
        selected.update(
            path.relative_to(REPOSITORY_ROOT)
            for path in absolute.rglob("*")
            if path.is_file()
        )
    return sorted(selected, key=lambda path: path.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    revision = require_clean_revision()
    output = (
        args.output
        if args.output is not None
        else REPOSITORY_ROOT
        / "artifacts"
        / "cloud"
        / f"livifuser_kaggle_t4x2_{revision[:7]}.zip"
    ).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    files = selected_files()
    entries = []
    for relative in files:
        absolute = REPOSITORY_ROOT / relative
        entries.append(
            {
                "path": relative.as_posix(),
                "size_bytes": absolute.stat().st_size,
                "sha256": sha256_file(absolute),
            }
        )
    manifest = {
        "schema_version": 1,
        "purpose": "LiViFuser pilot5 Kaggle T4x2 training input",
        "git_revision": revision,
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "included_data_roots": [path.as_posix() for path in DATA_ROOTS],
        "exclusions": "raw bags, images outside exports, DINO ONNX, prior experiments",
        "files": entries,
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        for relative in files:
            archive.write(
                REPOSITORY_ROOT / relative,
                (Path("LiViFuser") / relative).as_posix(),
            )
        archive.writestr("LiViFuser/cloud_bundle_manifest.json", manifest_text)

    report = {
        "bundle": str(output),
        "bundle_size_bytes": output.stat().st_size,
        "bundle_sha256": sha256_file(output),
        "embedded_manifest_sha256": __import__("hashlib").sha256(
            manifest_text.encode("utf-8")
        ).hexdigest(),
        **{key: manifest[key] for key in ("git_revision", "file_count", "total_bytes")},
    }
    report_path = output.with_suffix(".manifest.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
