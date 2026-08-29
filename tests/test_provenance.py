from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from livifuser_nav.provenance import flush_and_close_memmaps, sha256_file  # noqa: E402


class DrvFSPostCloseHashTests(unittest.TestCase):
    def test_repository_backed_memmaps_are_closed_before_hashing(self) -> None:
        """Use /mnt/d when run in the frozen WSL checkout, not Linux /tmp."""

        with tempfile.TemporaryDirectory(prefix=".drvfs-memmap-", dir=REPO_ROOT) as name:
            root = Path(name)
            rgb_path = root / "rgb.npy"
            scan_path = root / "scan.npy"
            rgb = np.lib.format.open_memmap(
                rgb_path,
                mode="w+",
                dtype=np.uint8,
                shape=(12, 240, 320, 3),
            )
            scan = np.lib.format.open_memmap(
                scan_path,
                mode="w+",
                dtype=np.float32,
                shape=(12, 399),
            )
            rgb[:] = np.arange(12, dtype=np.uint8)[:, None, None, None]
            scan[:] = np.arange(399, dtype=np.float32)[None, :]

            flush_and_close_memmaps(rgb, scan)

            self.assertTrue(rgb._mmap.closed)
            self.assertTrue(scan._mmap.closed)
            hashes = {path.name: sha256_file(path) for path in (rgb_path, scan_path)}

            reopened_rgb = np.load(rgb_path, mmap_mode="r")
            reopened_scan = np.load(scan_path, mmap_mode="r")
            try:
                self.assertEqual(reopened_rgb.shape, (12, 240, 320, 3))
                self.assertEqual(reopened_scan.shape, (12, 399))
                self.assertEqual(int(reopened_rgb[11, 0, 0, 0]), 11)
                self.assertEqual(float(reopened_scan[0, 398]), 398.0)
                self.assertEqual(hashes["rgb.npy"], sha256_file(rgb_path))
                self.assertEqual(hashes["scan.npy"], sha256_file(scan_path))
            finally:
                flush_and_close_memmaps(reopened_rgb, reopened_scan)


if __name__ == "__main__":
    unittest.main()
