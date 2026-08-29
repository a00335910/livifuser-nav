"""Nounset must not be active when a script sources ROS's setup.bash.

ROS 2 Humble's `/opt/ros/humble/setup.bash` dereferences
`AMENT_TRACE_SETUP_FILES` and related variables without defaulting them. Under
`set -u` the source aborts on its eighth line, before the simulator, the runner,
or the bag recorder is ever reached, and the failure message names an internal
ament variable rather than anything in this repository.

The byte-frozen `scripts/run_confirmatory_sim_episode.sh` established the
correct ordering across 282 collection attempts: `set -eo pipefail`, source both
setups, then `set -u`. This pins that ordering for every script, because the
bug has now appeared twice in scripts written later.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# `set -u`, `set -eu`, `set -euo pipefail`, `set -o nounset`
NOUNSET_ON = re.compile(r"^\s*set\s+(-[a-tv-z]*u[a-z]*|-o\s+nounset)\b")
NOUNSET_OFF = re.compile(r"^\s*set\s+(\+[a-tv-z]*u[a-z]*|\+o\s+nounset)\b")
SOURCES_ROS = re.compile(r"^\s*(source|\.)\s+.*setup\.bash")


def _nounset_active_at_ros_source(text: str) -> list[int]:
    """Return 1-indexed lines that source ROS while nounset is active."""

    offenders = []
    active = False
    for number, line in enumerate(text.splitlines(), start=1):
        if NOUNSET_OFF.match(line):
            active = False
        elif NOUNSET_ON.match(line):
            active = True
        elif SOURCES_ROS.match(line) and active:
            offenders.append(number)
    return offenders


class RosSourcingNounsetTests(unittest.TestCase):
    def test_no_script_sources_ros_under_nounset(self) -> None:
        failures = {}
        for script in sorted(SCRIPTS.glob("*.sh")):
            offenders = _nounset_active_at_ros_source(script.read_text(encoding="utf-8"))
            if offenders:
                failures[script.name] = offenders
        self.assertEqual(
            failures,
            {},
            "these scripts source ROS setup.bash while `set -u` is active, which "
            f"aborts before anything runs: {failures}",
        )

    def test_the_detector_recognises_the_failure_it_guards(self) -> None:
        broken = "set -euo pipefail\nsource /opt/ros/humble/setup.bash\n"
        self.assertEqual(_nounset_active_at_ros_source(broken), [2])

        frozen_pattern = (
            "set -eo pipefail\nsource /opt/ros/humble/setup.bash\nset -u\n"
        )
        self.assertEqual(_nounset_active_at_ros_source(frozen_pattern), [])

        relaxed_pattern = (
            "set -euo pipefail\nset +u\nsource /opt/ros/humble/setup.bash\nset -u\n"
        )
        self.assertEqual(_nounset_active_at_ros_source(relaxed_pattern), [])

    def test_the_frozen_collection_runner_still_shows_the_correct_ordering(self) -> None:
        # This file is byte-frozen by the v3 recollection freeze and is the
        # reference for the pattern; if it ever changes, this test should be
        # revisited rather than silently relaxed.
        frozen = SCRIPTS / "run_confirmatory_sim_episode.sh"
        self.assertEqual(_nounset_active_at_ros_source(frozen.read_text(encoding="utf-8")), [])


class ReservedSimulationDomainTests(unittest.TestCase):
    """Simulation runs on the reserved domain 97 and nowhere else.

    Keeping every simulated run on one known domain is what guarantees a
    simulated command cannot reach the physical robot's domain; the launch file
    enforces it and refuses to start otherwise. A per-episode domain was tried
    to isolate leaked nodes and traded that safety invariant for a cleanup
    problem the process-group teardown already solves.
    """

    def test_episode_runners_use_the_reserved_domain(self) -> None:
        for name in (
            "run_live_sim_development_episode.sh",
            "run_confirmatory_sim_episode.sh",
        ):
            text = (SCRIPTS / name).read_text(encoding="utf-8")
            assignments = re.findall(r"^\s*export ROS_DOMAIN_ID=(.+)$", text, re.MULTILINE)
            self.assertTrue(assignments, f"{name} sets no ROS_DOMAIN_ID")
            for value in assignments:
                self.assertEqual(value.strip(), "97", f"{name} must use the reserved domain")

    def test_localhost_only_is_set(self) -> None:
        text = (SCRIPTS / "run_live_sim_development_episode.sh").read_text(encoding="utf-8")
        self.assertIn("ROS_LOCALHOST_ONLY=1", text)


if __name__ == "__main__":
    unittest.main()
