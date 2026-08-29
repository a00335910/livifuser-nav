from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from livifuser_nav.operator_collection import (
    PLAN_COLUMNS,
    CollectionEpisode,
    load_collection_plan,
    remote_episode_paths,
)


def valid_row(sequence: int = 1, **overrides: str) -> dict[str, str]:
    row = {
        "sequence": str(sequence),
        "episode_id": f"train_room_a_route01_layout01_{sequence:03d}",
        "split": "train",
        "environment_id": "room_a",
        "route_id": "route01",
        "layout_id": f"layout{sequence:02d}",
        "duration_s": "60",
        "forward_m": "3.0",
        "left_m": "0.0",
        "obstacles": "one cardboard box at measured mark",
        "lighting": "ceiling lights on; blinds closed",
        "route_notes": "marked start and goal; dry clear floor",
        "confirmed": "true",
    }
    row.update(overrides)
    return row


class CollectionEpisodeTests(unittest.TestCase):
    def test_valid_episode_produces_exact_authorization_record(self) -> None:
        episode = CollectionEpisode.from_csv_row(valid_row(), row_number=2)
        record = episode.operator_record(
            revision="4ab54c1", authorized_wall_time="2026-08-04T12:00:00+01:00"
        )
        self.assertEqual(record["episode_id"], episode.episode_id)
        self.assertEqual(record["duration_s"], 60.0)
        self.assertEqual(
            record["authorization"]["kind"],
            "local_operator_exact_episode_confirmation",
        )

    def test_unconfirmed_or_invalid_physical_plan_cannot_arm(self) -> None:
        episode = CollectionEpisode.from_csv_row(valid_row(confirmed="false"), row_number=2)
        with self.assertRaisesRegex(ValueError, "not confirmed"):
            episode.require_confirmed()
        for override in (
            {"duration_s": "301"},
            {"forward_m": "0"},
            {"obstacles": ""},
            {"episode_id": "UPPER"},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                CollectionEpisode.from_csv_row(valid_row(**override), row_number=2)


class CollectionPlanTests(unittest.TestCase):
    def write_plan(self, rows: list[dict[str, str]]) -> Path:
        path = Path(self.temporary.name) / "plan.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PLAN_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_plan_requires_order_unique_ids_and_environment_level_splits(self) -> None:
        rows = [
            valid_row(1),
            valid_row(
                2,
                episode_id="validation_corridor_a_route01_layout01_001",
                split="validation",
                environment_id="corridor_a",
            ),
        ]
        plan = load_collection_plan(self.write_plan(rows), expected_count=2)
        self.assertEqual(len(plan.episodes), 2)

        leaked = rows.copy()
        leaked[1] = valid_row(
            2,
            episode_id="validation_room_a_route01_layout01_001",
            split="validation",
            environment_id="room_a",
        )
        with self.assertRaisesRegex(ValueError, "environment leakage"):
            load_collection_plan(self.write_plan(leaked), expected_count=2)

    def test_expected_count_duplicate_and_sequence_gaps_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must contain 2"):
            load_collection_plan(self.write_plan([valid_row(1)]), expected_count=2)
        duplicate = [valid_row(1), valid_row(2, episode_id=valid_row(1)["episode_id"])]
        with self.assertRaisesRegex(ValueError, "unique"):
            load_collection_plan(self.write_plan(duplicate), expected_count=2)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            load_collection_plan(self.write_plan([valid_row(2)]), expected_count=1)

    def test_pending_preserves_plan_order(self) -> None:
        plan = load_collection_plan(self.write_plan([valid_row(1), valid_row(2)]), expected_count=2)
        self.assertEqual(plan.pending({plan.episodes[0].episode_id}), (plan.episodes[1],))


class RemotePathTests(unittest.TestCase):
    def test_paths_are_exactly_bounded_below_remote_root(self) -> None:
        paths = remote_episode_paths(
            "/home/operator/livifuser_bags", "train_room_a_route01_layout01_001"
        )
        self.assertEqual(paths[0].name, "train_room_a_route01_layout01_001")
        self.assertEqual(paths[1].name, "train_room_a_route01_layout01_001.episode.json")
        self.assertEqual(paths[2].name, "train_room_a_route01_layout01_001.operator.json")
        for root in ("/", "relative"):
            with self.subTest(root=root), self.assertRaises(ValueError):
                remote_episode_paths(root, "train_room_a_001")


if __name__ == "__main__":
    unittest.main()
