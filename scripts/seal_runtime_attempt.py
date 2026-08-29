#!/usr/bin/env python3
"""Seal a completed development attempt without modifying its evidence files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MANIFEST = "ATTEMPT_MANIFEST.json"
COMPLETE = "ATTEMPT_COMPLETE.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=Path)
    args = parser.parse_args()
    root = args.attempt.resolve()
    if not root.is_dir():
        raise ValueError(f"attempt is not a directory: {root}")
    manifest_path, complete_path = root / MANIFEST, root / COMPLETE
    if manifest_path.exists() or complete_path.exists():
        raise FileExistsError("attempt is already sealed or partially sealed")
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append(
            {
                "name": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "1.0.0",
        "status": "sealed_development_attempt",
        "confirmatory": False,
        "members": records,
        "member_count": len(records),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    completion = {
        "schema_version": "1.0.0",
        "status": "complete_development_attempt",
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest().upper(),
        "member_count": len(records),
    }
    complete_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
