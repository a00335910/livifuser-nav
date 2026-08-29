from __future__ import annotations

import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "ros2_ws"
    / "src"
    / "livifuser_command_watchdog"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from livifuser_command_watchdog.shutdown_policy import FirstSignalGate  # noqa: E402


class FirstSignalGateTests(unittest.TestCase):
    def test_only_first_signal_requests_unwind(self) -> None:
        gate = FirstSignalGate()

        self.assertFalse(gate.received)
        self.assertTrue(gate.accept())
        self.assertTrue(gate.received)
        self.assertFalse(gate.accept())
        self.assertFalse(gate.accept())


if __name__ == "__main__":
    unittest.main()
