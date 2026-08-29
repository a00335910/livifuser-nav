#!/usr/bin/env python3
"""Independently audit the sealed held-out-evaluation Kaggle code bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_SHA256 = "D9A8D22DEEF5663D704C661720724166E1F6C9AE9DA8FE4E1B0EBE6F9E451E13"
BUNDLE_SIZE_BYTES = 142_326
MANIFEST_SHA256 = "9E0A6F5176F290F46AC732575459053F5A7E95A8E8A2F53E67F6281B03517F74"
AMENDMENT_SHA256 = "2CD7ADE1AC43FBC74975D9987E6C6052F5146B9FAD4CB97B1D46BEC996F4EE55"
REPAIR_SHA256 = "EB19516B2D84D4830A7A34B7EDB56DFBACE7E8C8E17866AEB9605B9929AC9357"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def safe_member(name: str) -> bool:
    member = PurePosixPath(name)
    return not member.is_absolute() and ".." not in member.parts


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


def audit(bundle_path: Path) -> dict[str, Any]:
    require(bundle_path.stat().st_size == BUNDLE_SIZE_BYTES, "bundle size drift")
    require(sha256_file(bundle_path) == BUNDLE_SHA256, "bundle hash drift")
    with zipfile.ZipFile(bundle_path) as bundle:
        infos = [info for info in bundle.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)) == 38, "bundle exact member count drift")
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
        manifest_name = "LiViFuser/cloud_bundle_manifest.json"
        manifest_raw = bundle.read(manifest_name)
        require(sha256_bytes(manifest_raw) == MANIFEST_SHA256, "manifest hash drift")
        manifest = json.loads(manifest_raw)
        require(
            manifest["frozen_amendment_sha256"] == AMENDMENT_SHA256
            and manifest["execution_repair_sha256"] == REPAIR_SHA256
            and manifest["heldout_feature_or_outcome_included"] is False
            and int(manifest["file_count"]) == 37,
            "manifest semantic contract drift",
        )
        rows = {row["path"]: row for row in manifest["files"]}
        expected = {(Path("LiViFuser") / relative).as_posix() for relative in rows} | {
            manifest_name
        }
        require(set(names) == expected, "manifest/member exact set drift")
        compiled = 0
        for relative, record in rows.items():
            name = (Path("LiViFuser") / relative).as_posix()
            payload = bundle.read(name)
            require(
                len(payload) == int(record["size_bytes"])
                and sha256_bytes(payload) == record["sha256"],
                f"member hash drift: {relative}",
            )
            if relative.endswith(".py"):
                compile(payload, relative, "exec")
                compiled += 1
    return {
        "schema_version": 1,
        "status": "PASS",
        "bundle": {
            "path": str(bundle_path),
            "size_bytes": BUNDLE_SIZE_BYTES,
            "sha256": BUNDLE_SHA256,
            "member_count": 38,
            "manifest_sha256": MANIFEST_SHA256,
            "zip_crc": "PASS",
            "deterministic_metadata": True,
        },
        "files": {
            "manifested": 37,
            "all_hashes_verified": True,
            "python_files_compiled": compiled,
            "heldout_feature_or_outcome_included": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=REPOSITORY_ROOT
        / "artifacts"
        / "cloud"
        / "livifuser_sim_heldout_eval_code_2cd7ade1ac43_r3.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "sim_heldout_evaluation_code_r3_audit.json",
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
                "python_files_compiled": report["files"]["python_files_compiled"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
