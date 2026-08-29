"""Tests for the robot-side link probe agent.

The agent runs on the Pi and is the thing being trusted to say what the *link*
did. These tests exist because the first version quietly attributed its own disk
stalls to the network: a 1299 ms and a 1599 ms pause between receiving a probe
and echoing it produced a burst of replies that summarised as a link latency
spike, while ICMP over the same link never exceeded 9 ms. A measurement tool
that blames the network for its own blocking is worse than no tool.
"""

import importlib.util
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "link_probe_agent.py"
SPEC = importlib.util.spec_from_file_location("link_probe_agent", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestJsonlWriter(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)

    def test_write_returns_without_waiting_for_the_disk(self):
        """The property the whole fix exists for.

        A write that blocks the caller is a write that blocks ``recvfrom``.
        """

        writer = MODULE.JsonlWriter(self.directory / "slow.jsonl")
        self.addCleanup(writer.close)

        real_flush = writer._handle.flush
        released = threading.Event()

        def slow_flush():
            released.wait(timeout=5)
            real_flush()

        writer._handle.flush = slow_flush

        started = time.perf_counter()
        for index in range(20):
            writer.write(seq=index)
        elapsed = time.perf_counter() - started

        # The drain thread is stuck in the first flush for as long as we hold
        # the event, yet the caller has already returned from all 20 writes.
        self.assertLess(elapsed, 0.5)
        released.set()

    def test_lines_reach_disk_in_the_order_they_were_handed_over(self):
        """Ordering is load-bearing: it is what pins receipt before echo."""

        path = self.directory / "ordered.jsonl"
        writer = MODULE.JsonlWriter(path)
        for index in range(200):
            writer.write(seq=index)
        writer.close()

        written = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([row["seq"] for row in written], list(range(200)))

    def test_close_drains_what_was_queued(self):
        """A run that ends must not lose the tail it had already accepted."""

        path = self.directory / "drained.jsonl"
        writer = MODULE.JsonlWriter(path)
        for index in range(50):
            writer.write(seq=index)
        writer.close()

        self.assertEqual(len(path.read_text().splitlines()), 50)

    def test_pending_returns_to_zero_once_drained(self):
        path = self.directory / "pending.jsonl"
        writer = MODULE.JsonlWriter(path)
        self.addCleanup(writer.close)
        for index in range(30):
            writer.write(seq=index)
        deadline = time.monotonic() + 5
        while writer.pending and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(writer.pending, 0)


class TestEchoResponder(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)

        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server.bind(("127.0.0.1", 0))
        self.port = self.server.getsockname()[1]
        self.addCleanup(self.server.close)

        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client.settimeout(5)
        self.addCleanup(self.client.close)

        self.writer = MODULE.JsonlWriter(self.directory / "echo.jsonl")
        self.addCleanup(self.writer.close)
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=MODULE.echo_responder, args=(self.server, self.writer, self.stop)
        )
        self.thread.daemon = True
        self.thread.start()
        self.addCleanup(self._shutdown)

    def _shutdown(self):
        self.stop.set()
        self.thread.join(timeout=5)

    def _send(self, seq):
        payload = json.dumps({"seq": seq, "t_send": time.time()}).encode()
        self.client.sendto(payload, ("127.0.0.1", self.port))

    def test_probe_is_echoed_with_both_robot_timestamps(self):
        self._send(7)
        reply = json.loads(self.client.recv(65535).decode())
        self.assertEqual(reply["seq"], 7)
        self.assertLessEqual(reply["t_pi_recv"], reply["t_pi_send"])

    def test_echo_is_not_delayed_by_a_stalled_disk(self):
        """The regression this file is really about.

        With the flush inline, holding the disk for a second held the echo for a
        second too, and every probe behind it queued in the socket buffer. The
        reply must now come back while the writer is still stuck.
        """

        real_flush = self.writer._handle.flush
        released = threading.Event()

        def slow_flush():
            released.wait(timeout=5)
            real_flush()

        self.writer._handle.flush = slow_flush

        started = time.perf_counter()
        self._send(11)
        reply = json.loads(self.client.recv(65535).decode())
        elapsed = time.perf_counter() - started

        self.assertEqual(reply["seq"], 11)
        self.assertLess(elapsed, 0.5)
        released.set()

    def test_a_malformed_probe_is_still_recorded_and_answered(self):
        self.client.sendto(b"not json at all", ("127.0.0.1", self.port))
        reply = json.loads(self.client.recv(65535).decode())
        self.assertIsNone(reply["seq"])

    def test_receive_gap_is_recorded_for_every_probe_after_the_first(self):
        """Without this field, an agent draining its socket buffer and a link
        going quiet look identical from the robot side."""

        for seq in range(4):
            self._send(seq)
            self.client.recv(65535)

        self.writer.close()
        rows = [
            json.loads(line)
            for line in self.writer.path.read_text().splitlines()
        ]
        self.assertIsNone(rows[0]["since_previous_recv_s"])
        for row in rows[1:]:
            self.assertIsNotNone(row["since_previous_recv_s"])
            self.assertGreaterEqual(row["since_previous_recv_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
