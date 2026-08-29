"""Unit tests for sweep batch assembly.

Uses duck-typed stand-ins for the export run and feature cache so the stacking
order, per-window identity, and shape contracts are pinned without touching
real export artifacts. `tokenize_lidar` itself is covered by
`test_learning_data.py`.
"""

from __future__ import annotations

import unittest

import numpy as np

from livifuser_nav.batching import RunTokens, window_arrays
from livifuser_nav.learning_data import WindowRef

ROWS = 12
SECTORS = 5
CONTEXT = 3
HORIZON = 2


class FakeRun:
    def __init__(self, run_id: str, offset: float) -> None:
        self.run_id = run_id
        self.count = ROWS
        rows = np.arange(ROWS, dtype=np.float32)
        self.vectors = {
            "goal": np.stack([rows + offset, rows, rows], axis=1),
            "robot_state": np.stack([rows + offset, -rows], axis=1),
            "action": np.stack([rows + offset, rows * 0.1], axis=1),
        }


class FakeDataset:
    def __init__(self, runs: tuple[FakeRun, ...]) -> None:
        self.runs = runs

    def targets(self, ref: WindowRef) -> np.ndarray:
        return np.asarray(
            self.runs[ref.run_index].vectors["action"][list(ref.action_rows)]
        )


class FakeCache:
    def __init__(self, offset: float) -> None:
        rows = np.arange(ROWS, dtype=np.float32)
        self.patch_tokens = (rows + offset)[:, None, None] * np.ones(
            (ROWS, 49, 384), dtype=np.float16
        )
        self.pooled_features = np.stack([rows + offset] * 384, axis=1)


def fake_tokens(offset: float) -> RunTokens:
    rows = np.arange(ROWS, dtype=np.float32)
    features = (rows + offset)[:, None, None] * np.ones(
        (ROWS, SECTORS, 4), dtype=np.float32
    )
    visual_mask = np.zeros((ROWS, SECTORS, 49), dtype=bool)
    visual_mask[:, :, 0] = True
    in_fov = np.ones((ROWS, SECTORS), dtype=bool)
    return RunTokens(features, visual_mask, in_fov)


def make_ref(run_index: int, origin: int) -> WindowRef:
    context = tuple(range(origin - CONTEXT + 1, origin + 1))
    actions = tuple(range(origin, origin + HORIZON))
    return WindowRef(run_index, context, actions)


class WindowArraysTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = FakeDataset((FakeRun("run_a", 0.0), FakeRun("run_b", 100.0)))
        self.caches = [FakeCache(0.0), FakeCache(100.0)]
        self.tokens = [fake_tokens(0.0), fake_tokens(100.0)]

    def test_shapes_and_identity_follow_the_refs(self) -> None:
        refs = [make_ref(0, 4), make_ref(1, 6)]
        arrays = window_arrays(self.dataset, self.caches, self.tokens, refs)
        self.assertEqual(arrays["visual_tokens"].shape, (2, CONTEXT, 49, 384))
        self.assertEqual(arrays["lidar_features"].shape, (2, CONTEXT, SECTORS, 4))
        self.assertEqual(arrays["visual_mask"].shape, (2, CONTEXT, SECTORS, 49))
        self.assertEqual(arrays["in_fov"].shape, (2, CONTEXT, SECTORS))
        self.assertEqual(arrays["goal"].shape, (2, CONTEXT, 3))
        self.assertEqual(arrays["robot_state"].shape, (2, CONTEXT, 2))
        self.assertEqual(arrays["target"].shape, (2, HORIZON, 2))
        self.assertEqual(arrays["origin_pooled_features"].shape, (2, 384))
        self.assertEqual(arrays["episode_ids"], ["run_a", "run_b"])
        self.assertEqual(arrays["origin_rows"], [4, 6])

    def test_rows_come_from_the_right_run(self) -> None:
        refs = [make_ref(1, 5)]
        arrays = window_arrays(self.dataset, self.caches, self.tokens, refs)
        # Run B values are offset by 100, and the context ends at the origin row.
        self.assertEqual(float(arrays["goal"][0, -1, 0]), 105.0)
        self.assertEqual(float(arrays["lidar_features"][0, -1, 0, 0]), 105.0)
        self.assertEqual(float(arrays["origin_pooled_features"][0, 0]), 105.0)
        self.assertEqual(float(arrays["target"][0, 0, 0]), 105.0)

    def test_visual_tokens_are_float32(self) -> None:
        arrays = window_arrays(
            self.dataset, self.caches, self.tokens, [make_ref(0, 3)]
        )
        self.assertEqual(arrays["visual_tokens"].dtype, np.float32)

    def test_cache_count_mismatch_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "per run"):
            window_arrays(self.dataset, self.caches[:1], self.tokens, [make_ref(0, 3)])


if __name__ == "__main__":
    unittest.main()
