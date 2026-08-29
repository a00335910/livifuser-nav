"""Pure shutdown-signal policy shared by robot-local command processes."""

from __future__ import annotations


class FirstSignalGate:
    """Let the first termination signal unwind execution and absorb repeats.

    systemd may signal more than one process in a service control group. A ROS
    launcher wrapper can consequently cause the node to observe another signal
    while it is already publishing its final safety zeros. Python signal
    handlers execute on the main thread, so this small latch needs no lock.
    """

    def __init__(self) -> None:
        self._received = False

    @property
    def received(self) -> bool:
        return self._received

    def accept(self) -> bool:
        if self._received:
            return False
        self._received = True
        return True
