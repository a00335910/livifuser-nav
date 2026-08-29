"""Real-process regression test for the watchdog's SIGTERM zero burst.

The test uses localhost-only discovery on an isolated ROS domain. It cannot
reach a physical robot even if one is powered and visible on the normal domain.

The child process receives that isolation explicitly rather than inheriting
whatever the parent's environment happens to hold: `unittest discover` imports
every test module before running any, so module-scope assignments are decided by
sort order, not by which test is executing.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "ros2_ws"
    / "src"
    / "livifuser_command_watchdog"
)
sys.path.insert(0, str(PACKAGE_ROOT))

ISOLATION_ENVIRONMENT = {"ROS_DOMAIN_ID": "79", "ROS_LOCALHOST_ONLY": "1"}
os.environ.update(ISOLATION_ENVIRONMENT)

try:
    import rclpy  # noqa: F401 - availability is the skip condition

    ROS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised off the ROS host
    ROS_AVAILABLE = False


@unittest.skipUnless(ROS_AVAILABLE, "rclpy unavailable")
class WatchdogSigtermTests(unittest.TestCase):
    def test_sigterm_publishes_final_zero_burst_before_exit(self) -> None:
        environment = os.environ.copy()
        environment.update(ISOLATION_ENVIRONMENT)
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (str(PACKAGE_ROOT), existing_pythonpath)
            if part
        )
        child_code = """
import os
import signal

from livifuser_command_watchdog import node as watchdog_node

original = watchdog_node.CommandWatchdog.publish_final_zero
original_init = watchdog_node.CommandWatchdog.__init__
final_zero_count = 0

def observed_init(self):
    original_init(self)
    print("WATCHDOG_READY", flush=True)

def observed_final_zero(self):
    global final_zero_count
    original(self)
    final_zero_count += 1
    print("FINAL_ZERO_PUBLISHED", flush=True)
    if final_zero_count == 1:
        # Reproduce the live systemd failure: another termination signal arrives
        # after cleanup has begun. It must be absorbed, not abort the burst.
        os.kill(os.getpid(), signal.SIGTERM)

watchdog_node.CommandWatchdog.__init__ = observed_init
watchdog_node.CommandWatchdog.publish_final_zero = observed_final_zero
watchdog_node.main()
"""
        process = subprocess.Popen(
            [sys.executable, "-c", child_code],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert process.stdout is not None
            startup_lines: list[str] = []
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline and process.poll() is None:
                readable, _, _ = select.select([process.stdout], [], [], 0.1)
                if readable:
                    line = process.stdout.readline()
                    startup_lines.append(line)
                    if line.strip() == "WATCHDOG_READY":
                        break
            if not any(line.strip() == "WATCHDOG_READY" for line in startup_lines):
                if process.poll() is None:
                    process.kill()
                stdout, stderr = process.communicate()
                self.fail(
                    f"watchdog was not ready before SIGTERM\n"
                    f"stdout:\n{''.join(startup_lines)}{stdout}\n"
                    f"stderr:\n{stderr}"
                )

            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=5.0)
            stdout = "".join(startup_lines) + stdout
            self.assertEqual(process.returncode, 0, stdout + stderr)
            self.assertEqual(stdout.count("FINAL_ZERO_PUBLISHED"), 5)
            self.assertNotIn("shutdown already called", stderr)
            self.assertNotIn("KeyboardInterrupt", stderr)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
