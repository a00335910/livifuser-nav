import unittest

from livifuser_nav.association import nearest_sample
from livifuser_nav.contracts import StampedValue


class NearestSampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [
            StampedValue(0, "a"),
            StampedValue(200_000_000, "b"),
            StampedValue(400_000_000, "c"),
        ]

    def test_selects_nearest_lidar_scan(self) -> None:
        self.assertEqual(nearest_sample(290_000_000, self.samples).value, "b")
        self.assertEqual(nearest_sample(310_000_000, self.samples).value, "c")

    def test_prefers_earlier_scan_on_tie(self) -> None:
        self.assertEqual(nearest_sample(300_000_000, self.samples).value, "b")

    def test_rejects_stale_scan(self) -> None:
        with self.assertRaises(ValueError):
            nearest_sample(700_000_000, self.samples, max_delta_ns=100_000_000)

    def test_rejects_unsorted_input(self) -> None:
        with self.assertRaises(ValueError):
            nearest_sample(1, list(reversed(self.samples)))


if __name__ == "__main__":
    unittest.main()

