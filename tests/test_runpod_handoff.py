from __future__ import annotations

import unittest

from livifuser_nav.runpod_handoff import (
    BUNDLE_ROOT,
    PROTECTED_CACHE_NAMES,
    verify_runpod_handoff,
)


class RunPodHandoffTest(unittest.TestCase):
    def test_missing_bundle_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            verify_runpod_handoff("definitely-missing-runpod-input.zip")

    def test_protected_cache_names_are_explicit(self) -> None:
        self.assertEqual(len(PROTECTED_CACHE_NAMES), 2)
        self.assertTrue(all("cache" in name for name in PROTECTED_CACHE_NAMES))

    def test_bundle_root_is_versioned(self) -> None:
        self.assertEqual(BUNDLE_ROOT, "livifuser_runpod_input_v1")


if __name__ == "__main__":
    unittest.main()
