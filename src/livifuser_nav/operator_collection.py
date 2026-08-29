"""Validated operator-owned episode plans and offload manifests.

This module is deliberately ROS-free so collection metadata and destructive
offload decisions can be tested on Windows before the operator uses the WSL
launcher against the robot.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,79}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")
ALLOWED_SPLITS = frozenset({"train", "validation", "test", "development"})
PLAN_COLUMNS = (
    "sequence",
    "episode_id",
    "split",
    "environment_id",
    "route_id",
    "layout_id",
    "duration_s",
    "forward_m",
    "left_m",
    "obstacles",
    "lighting",
    "route_notes",
    "confirmed",
)


def _required_text(row: dict[str, str], name: str) -> str:
    value = row.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _number(row: dict[str, str], name: str) -> float:
    text = _required_text(row, name)
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _confirmed(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0", ""}:
        return False
    raise ValueError("confirmed must be true/false, yes/no, or 1/0")


@dataclass(frozen=True, slots=True)
class CollectionEpisode:
    sequence: int
    episode_id: str
    split: str
    environment_id: str
    route_id: str
    layout_id: str
    duration_s: float
    forward_m: float
    left_m: float
    obstacles: str
    lighting: str
    route_notes: str
    confirmed: bool

    @classmethod
    def from_csv_row(cls, row: dict[str, str], *, row_number: int) -> CollectionEpisode:
        missing = [name for name in PLAN_COLUMNS if name not in row]
        if missing:
            raise ValueError(f"row {row_number}: missing columns: {', '.join(missing)}")
        try:
            sequence = int(_required_text(row, "sequence"))
        except ValueError as error:
            raise ValueError(f"row {row_number}: sequence must be an integer") from error
        episode = cls(
            sequence=sequence,
            episode_id=_required_text(row, "episode_id"),
            split=_required_text(row, "split"),
            environment_id=_required_text(row, "environment_id"),
            route_id=_required_text(row, "route_id"),
            layout_id=_required_text(row, "layout_id"),
            duration_s=_number(row, "duration_s"),
            forward_m=_number(row, "forward_m"),
            left_m=_number(row, "left_m"),
            obstacles=_required_text(row, "obstacles"),
            lighting=_required_text(row, "lighting"),
            route_notes=_required_text(row, "route_notes"),
            confirmed=_confirmed(row.get("confirmed", "")),
        )
        try:
            episode.validate()
        except ValueError as error:
            raise ValueError(f"row {row_number}: {error}") from error
        return episode

    def validate(self) -> None:
        if self.sequence <= 0:
            raise ValueError("sequence must be positive")
        for name in ("episode_id", "environment_id", "route_id", "layout_id"):
            value = str(getattr(self, name))
            if not IDENTIFIER_PATTERN.fullmatch(value):
                raise ValueError(f"{name} must match [a-z0-9][a-z0-9_-]{{2,79}}")
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}")
        if not 5.0 <= self.duration_s <= 300.0:
            raise ValueError("duration_s must be in [5, 300]")
        if not 0.25 <= self.forward_m <= 10.0:
            raise ValueError("forward_m must be in [0.25, 10]")
        if not -5.0 <= self.left_m <= 5.0:
            raise ValueError("left_m must be in [-5, 5]")
        for name in ("obstacles", "lighting", "route_notes"):
            value = str(getattr(self, name))
            if not value.strip() or len(value) > 500:
                raise ValueError(f"{name} must contain 1-500 characters")

    def require_confirmed(self) -> None:
        if not self.confirmed:
            raise ValueError(
                f"{self.episode_id} is not confirmed; bind its physical details and "
                "set confirmed=true before recording"
            )

    def operator_record(self, *, revision: str, authorized_wall_time: str) -> dict[str, Any]:
        if not REVISION_PATTERN.fullmatch(revision):
            raise ValueError("revision must be a 7-40 character lowercase Git hash")
        self.require_confirmed()
        document = asdict(self)
        document.update(
            {
                "schema_version": "1.0.0",
                "acquisition_code_revision": revision,
                "authorization": {
                    "kind": "local_operator_exact_episode_confirmation",
                    "episode_id": self.episode_id,
                    "confirmed_wall_time": authorized_wall_time,
                },
            }
        )
        return document


@dataclass(frozen=True, slots=True)
class CollectionPlan:
    source: Path
    episodes: tuple[CollectionEpisode, ...]

    def pending(self, completed_ids: set[str]) -> tuple[CollectionEpisode, ...]:
        return tuple(
            episode for episode in self.episodes if episode.episode_id not in completed_ids
        )


def load_collection_plan(path: Path, *, expected_count: int | None = 30) -> CollectionPlan:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("collection plan has no CSV header")
        missing = [name for name in PLAN_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"collection plan missing columns: {', '.join(missing)}")
        episodes = tuple(
            CollectionEpisode.from_csv_row(row, row_number=index)
            for index, row in enumerate(reader, start=2)
        )
    if expected_count is not None and len(episodes) != expected_count:
        raise ValueError(
            f"collection plan must contain {expected_count} episodes, got {len(episodes)}"
        )
    sequences = [episode.sequence for episode in episodes]
    if sequences != list(range(1, len(episodes) + 1)):
        raise ValueError("sequence must be contiguous and ordered from 1")
    ids = [episode.episode_id for episode in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("episode_id values must be unique")
    environments_by_split: dict[str, set[str]] = {}
    for episode in episodes:
        environments_by_split.setdefault(episode.split, set()).add(episode.environment_id)
    split_names = sorted(environments_by_split)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            overlap = environments_by_split[first] & environments_by_split[second]
            if overlap:
                raise ValueError(
                    f"environment leakage between {first} and {second}: {sorted(overlap)}"
                )
    return CollectionPlan(path.resolve(), episodes)


def remote_episode_paths(remote_root: str, episode_id: str) -> tuple[PurePosixPath, ...]:
    if not IDENTIFIER_PATTERN.fullmatch(episode_id):
        raise ValueError("invalid episode_id")
    root = PurePosixPath(remote_root)
    if not root.is_absolute() or str(root) == "/":
        raise ValueError("remote_root must be an absolute, non-root POSIX path")
    return (
        root / episode_id,
        root / f"{episode_id}.episode.json",
        root / f"{episode_id}.operator.json",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
