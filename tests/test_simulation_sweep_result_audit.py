"""Unit tests for the immutable simulation-result bundle auditor."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_auditor():
    path = REPOSITORY_ROOT / "scripts" / "audit_simulation_sweep_results.py"
    spec = importlib.util.spec_from_file_location("audit_simulation_sweep_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_auditor()


class MemberContractTests(unittest.TestCase):
    def test_expected_bundle_has_exactly_99_members(self) -> None:
        runs = {
            name: {"name": name} for partition in AUDIT.PARTITIONS.values() for name in partition
        }
        files, directories = AUDIT.expected_members(runs, [20260805, 20260806, 20260807])
        self.assertEqual(len(files), 64)
        self.assertEqual(len(directories), 35)
        self.assertEqual(len(files | directories), 99)

    def test_variant_summary_preserves_every_seed(self) -> None:
        rows = [
            {
                "name": "full",
                "variant": "full",
                "loss": "heteroscedastic",
                "seed": seed,
                "parameter_count": 10,
                "training_seconds": seconds,
                "macro_episode_normalized_mse": mse,
                "macro_episode_nll": nll,
            }
            for seed, seconds, mse, nll in (
                (20260807, 3.0, 0.3, 0.6),
                (20260805, 1.0, 0.1, 0.2),
                (20260806, 2.0, 0.2, 0.4),
            )
        ]
        summary = AUDIT.summarize_variant(rows)
        self.assertEqual(
            [row["seed"] for row in summary["per_seed"]],
            [20260805, 20260806, 20260807],
        )
        self.assertAlmostEqual(summary["mse_mean"], 0.2)
        self.assertAlmostEqual(summary["nll_mean"], 0.4)
        self.assertEqual(summary["training_seconds_total"], 6.0)
        self.assertTrue(summary["nll_interpretable"])


if __name__ == "__main__":
    unittest.main()
