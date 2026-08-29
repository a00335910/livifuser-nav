#!/usr/bin/env python3
"""Robot-side link probe agent. Runs on the Pi; transfer this file and run it.

Standard library only, Python 3.10 compatible. Nothing to install.

It does two jobs, and both exist because the operator machine alone cannot
answer the questions that matter:

**UDP echo responder.** The laptop sends numbered probes; this stamps each with
its own receive time and echoes it back. A probe that never returns tells the
laptop only that *something* failed. Comparing what this agent logged against
what the laptop received separates the two cases: a command lost on the way out
means the robot never acted, while a reply lost on the way back means it acted
and the operator did not see it. For a moving robot those are different
failures with different consequences.

**Local state logger.** Interface counters, signal, load and temperature written
to the Pi's own disk. Anything pulled over SSH disappears exactly when the link
drops, which is the moment worth recording, so this writes locally and is
collected afterwards.

Typical use on the robot::

    nohup python3 link_probe_agent.py \\
        --output ~/link_logs/c2_run1_pi --duration 900 > ~/link_logs/agent.out 2>&1 &

Then afterwards, from the laptop::

    scp pi@robot:~/link_logs/'c2_run1_pi*' artifacts/network/

Writes ``<output>.echo.jsonl`` (one line per probe received) and
``<output>.state.jsonl`` (periodic robot state).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import queue
import signal
import socket
import subprocess
import threading
import time
from typing import Any

SCHEMA_VERSION = "1.1.0"
DEFAULT_PORT = 47821

#: Local state sources. Output is stored verbatim and never parsed here, so a
#: field nobody thought to extract today can still be extracted later.
STATE_SOURCES: tuple[tuple[str, list[str] | str], ...] = (
    ("net_dev", "cat /proc/net/dev"),
    ("wireless", "cat /proc/net/wireless 2>/dev/null || true"),
    ("loadavg", "cat /proc/loadavg"),
    ("meminfo", "head -5 /proc/meminfo"),
    ("uptime", "uptime"),
    ("temp", "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || true"),
    ("throttled", "vcgencmd get_throttled 2>/dev/null || true"),
    ("routes", "ip route 2>/dev/null || true"),
    ("addrs", "ip -brief addr 2>/dev/null || true"),
    ("modem", "mmcli -m any 2>/dev/null || true"),
    ("modem_signal", "mmcli -m any --signal-get 2>/dev/null || true"),
    ("tailscale", "tailscale status --json 2>/dev/null || true"),
)


class JsonlWriter:
    """Append-only JSONL writer that flushes every line off the calling thread.

    Flushing per line is deliberate. This process may be killed abruptly when a
    run ends or the robot is powered down, and a buffered tail would lose
    precisely the final seconds before whatever went wrong.

    The flush does *not* happen on the caller's thread, and that part was
    learned the hard way. The first version flushed inline, which put an SD-card
    write in the middle of the echo path: measured stalls of 1299 ms and 1599 ms
    between receiving a probe and echoing it, during which nothing called
    ``recvfrom`` and probes queued in the socket buffer. The queued probes then
    came back in a single burst and were summarised as a link latency spike,
    even though ICMP over the same link never moved above 9 ms at the time.
    Handing the line to a queue costs microseconds and keeps the measurement
    about the network rather than about this agent's disk.
    """

    def __init__(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self._queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()
        self.path = path
        #: Lines handed over but not yet on disk. Recorded per probe so a future
        #: stall is visible in the data instead of being mistaken for the link.
        self.pending = 0
        self._pending_lock = threading.Lock()

    def _drain(self) -> None:
        while True:
            line = self._queue.get()
            if line is None:
                return
            self._handle.write(line + "\n")
            self._handle.flush()
            with self._pending_lock:
                self.pending -= 1

    def write(self, **fields: Any) -> None:
        line = json.dumps(fields)
        with self._pending_lock:
            self.pending += 1
        self._queue.put(line)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=10)
        self._handle.close()


def echo_responder(
    sock: socket.socket, writer: JsonlWriter, stop: threading.Event
) -> None:
    """Log and echo every probe received.

    The receipt is recorded *before* the echo is attempted, so a probe that
    arrived is recorded as arrived even if the reply never leaves. That ordering
    is what makes uplink and downlink loss distinguishable afterwards, and it is
    preserved here — ``writer.write`` enqueues in order and returns immediately,
    so only the disk write moved off this thread, not the ordering.
    """

    sock.settimeout(0.5)
    previous_recv: float | None = None
    while not stop.is_set():
        try:
            data, address = sock.recvfrom(65535)
        except TimeoutError:
            continue
        except OSError:
            break

        t_recv = time.time()
        try:
            probe = json.loads(data.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            probe = {"unparsed": data[:200].decode("utf-8", "replace")}

        writer.write(
            t_pi_recv=t_recv,
            seq=probe.get("seq"),
            t_send=probe.get("t_send"),
            bytes=len(data),
            peer=f"{address[0]}:{address[1]}",
            # Gap since the previous probe was dequeued. A run of near-zero gaps
            # after one long gap is this agent draining its socket buffer, which
            # is not the same event as the link going quiet — and without this
            # field the two are indistinguishable from the robot side.
            since_previous_recv_s=None if previous_recv is None else t_recv - previous_recv,
            writer_pending=writer.pending,
        )
        previous_recv = t_recv

        reply = {
            "seq": probe.get("seq"),
            "t_send": probe.get("t_send"),
            "t_pi_recv": t_recv,
            "t_pi_send": time.time(),
        }
        try:
            sock.sendto(json.dumps(reply).encode("utf-8"), address)
        except OSError:
            # Reply could not leave. The receipt above is already recorded, so
            # this shows up later as downlink loss rather than vanishing.
            continue


def collect_state() -> dict[str, Any]:
    captured: dict[str, Any] = {}
    for name, command in STATE_SOURCES:
        try:
            done = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=8, check=False
            )
            captured[name] = done.stdout
        except (OSError, subprocess.TimeoutExpired) as error:
            captured[name] = f"<error: {error}>"
    return captured


def state_logger(
    writer: JsonlWriter, interval: float, stop: threading.Event
) -> None:
    while not stop.is_set():
        writer.write(t_pi=time.time(), monotonic=time.monotonic(), state=collect_state())
        stop.wait(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="output path prefix on the robot")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="UDP port to listen on")
    parser.add_argument("--bind", default="0.0.0.0", help="address to bind")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds to run; 0 means run until stopped",
    )
    parser.add_argument("--state-interval", type=float, default=5.0)
    parser.add_argument("--no-state", action="store_true", help="echo responder only")
    args = parser.parse_args(argv)

    base = pathlib.Path(args.output).expanduser()
    echo_writer = JsonlWriter(base.with_suffix(".echo.jsonl"))
    state_writer = None if args.no_state else JsonlWriter(base.with_suffix(".state.jsonl"))

    meta = {
        "schema_version": SCHEMA_VERSION,
        "role": "robot_agent",
        "started_wall": time.time(),
        "started_monotonic": time.monotonic(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "port": args.port,
    }
    base.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    except OSError:
        pass
    sock.bind((args.bind, args.port))

    stop = threading.Event()

    def handle_signal(signum, _frame):
        print(f"agent: signal {signum}, stopping", flush=True)
        stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    threads = [threading.Thread(target=echo_responder, args=(sock, echo_writer, stop))]
    if state_writer is not None:
        threads.append(
            threading.Thread(
                target=state_logger, args=(state_writer, args.state_interval, stop)
            )
        )
    for thread in threads:
        thread.daemon = True
        thread.start()

    print(
        f"agent listening on {args.bind}:{args.port}; "
        f"echo -> {echo_writer.path.name}"
        + (f", state -> {state_writer.path.name}" if state_writer else ""),
        flush=True,
    )

    started = time.time()
    try:
        while not stop.is_set():
            if args.duration and (time.time() - started) >= args.duration:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=4)
        sock.close()
        echo_writer.close()
        if state_writer is not None:
            state_writer.close()

    print(f"agent stopped after {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
