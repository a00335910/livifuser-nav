"""Provenance capture for reproducible exports.

`docs/RESEARCH_RECORD.md` requires every material artifact to cite the exact
code revision and input hashes that produced it. The repository currently has no
commits, so `git rev-parse` cannot supply a revision. Rather than omit code
identity, :func:`code_identity` hashes the source files themselves and records
the git state honestly, so a dataset produced today can still be tied to the
exact logic that built it.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Uppercase SHA-256 of one file, matching the checksums already recorded."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def flush_and_close_memmaps(*arrays: Any) -> None:
    """Make writable NumPy mappings final before provenance hashing.

    ``memmap.flush()`` writes dirty pages but leaves the mapping open. On WSL
    DrvFS, hashing in that state can record bytes that differ from the file
    observed after process teardown. Closing the backing ``mmap`` establishes
    the exact file image that downstream audits and copies will read. ``Any``
    keeps this provenance module independent of NumPy at import time.
    """

    first_error: BaseException | None = None
    for array in arrays:
        try:
            array.flush()
        except BaseException as error:  # still close every mapping before failing
            if first_error is None:
                first_error = error
        mapping = getattr(array, "_mmap", None)
        if mapping is None:
            if first_error is None:
                first_error = RuntimeError("NumPy memmap has no backing mmap to close")
            continue
        try:
            mapping.close()
        except BaseException as error:  # preserve the first failure after cleanup
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def hash_paths(paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    """Hash a set of named inputs, recording absence rather than skipping it."""

    resolved: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        if path.is_file():
            resolved[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        else:
            resolved[name] = {"path": str(path), "sha256": None, "missing": True}
    return resolved


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


@dataclass(frozen=True, slots=True)
class CodeIdentity:
    """What produced an artifact, from git if possible and from hashes always."""

    git_revision: str | None
    git_state: str
    dirty: bool | None
    source_tree_sha256: str
    source_files: dict[str, str] = field(default_factory=dict)

    def as_manifest(self) -> dict[str, object]:
        return {
            "git_revision": self.git_revision,
            "git_state": self.git_state,
            "git_worktree_dirty": self.dirty,
            "source_tree_sha256": self.source_tree_sha256,
            "source_files": self.source_files,
        }


def code_identity(root: Path, sources: list[Path]) -> CodeIdentity:
    """Identify the code that produced an export.

    ``source_tree_sha256`` folds the individual file hashes into one value in a
    stable, path-sorted order, so it is a usable dataset-lineage key even while
    the repository has no commits to cite.
    """

    revision = _git(root, "rev-parse", "HEAD")
    if revision is None:
        git_state = "no_commits_or_not_a_repository"
        dirty: bool | None = None
    else:
        status = _git(root, "status", "--porcelain")
        dirty = bool(status)
        git_state = "dirty" if dirty else "clean"

    file_hashes: dict[str, str] = {}
    for source in sorted(sources, key=lambda path: path.as_posix()):
        if source.is_file():
            try:
                key = source.relative_to(root).as_posix()
            except ValueError:
                key = source.as_posix()
            file_hashes[key] = sha256_file(source)

    combined = "\n".join(f"{key}:{value}" for key, value in sorted(file_hashes.items()))
    return CodeIdentity(
        git_revision=revision,
        git_state=git_state,
        dirty=dirty,
        source_tree_sha256=sha256_bytes(combined.encode("utf-8")),
        source_files=file_hashes,
    )


def environment_identity() -> dict[str, object]:
    """Host and interpreter facts needed to reproduce a run."""

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
