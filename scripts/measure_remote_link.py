#!/usr/bin/env python3
"""Record remote-operation link quality for a robot on a cellular modem.

Standalone: no ROS, no project dependencies beyond the standard library. Run it
on the operator machine while driving the robot however you normally would.

The question it exists to answer is whether a robot on a 5G modem can be driven
remotely at all, so it collects broadly and writes **raw time series** rather than
only summaries. Analysis happens afterwards, against the recorded samples.

What it samples continuously, each with a wall-clock timestamp so the streams can
be cross-correlated later:

  icmp    round-trip time, and losses counted as losses rather than skipped
  tcp     time to complete a TCP handshake, which reveals congestion ICMP misses
  ssh     round-trip of a trivial remote command over a persistent connection,
          which is the closest proxy for "operator presses a key, robot reacts"
  modem   optional signal metrics polled from the modem, if a source is given

Signal metrics matter more than they look for a moving outdoor robot: latency
spikes are usually handovers or fades, and without a signal series alongside you
can see that a spike happened but never why.

Example::

    python3 scripts/measure_remote_link.py \\
        --label direct_d501_outdoor_run1 \\
        --target 100.x.y.z --ssh-target pi@100.x.y.z \\
        --duration 600 \\
        --output artifacts/network/link_direct_run1

Writes ``<output>.jsonl`` (every sample) and ``<output>.summary.json``.
"""

from __future__ import annotations

import argparse
import json
import platform
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from livifuser_nav.network_metrics import (  # noqa: E402
    CONTROL_BANDS_MS,
    attribute_loss,
    clock_offset_estimate,
    control_quality_bands,
    correlate,
    jitter_summary,
    outage_windows,
    parse_ping_rtt_ms,
    reordering_summary,
    split_by_state,
    summarize_ms,
)

SCHEMA_VERSION = "1.0.0"


class SampleWriter:
    """Serialises samples from several probe threads to one JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._handle = path.open("w", encoding="utf-8")
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            self._handle.write(json.dumps(item) + "\n")
        self._handle.flush()

    def write(self, **fields: Any) -> None:
        self._queue.put(fields)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)
        self._handle.close()


_SEQ = re.compile(r"(?:icmp_seq|seq)[=\s]+(\d+)", re.IGNORECASE)
_TTL = re.compile(r"ttl[=\s]+(\d+)", re.IGNORECASE)


def icmp_probe(target: str, writer: SampleWriter, stop: threading.Event, t0: float) -> None:
    """Continuous ping, recording every line verbatim.

    The parsed fields are a convenience; the raw line is the evidence. Sequence
    numbers in particular are kept because loss, reordering and duplication are
    three different failures that a latency series alone cannot tell apart, and
    which one is happening changes what you would do about it.
    """

    windows = platform.system() == "Windows"
    command = (
        ["ping", "-n", "100000", target]
        if windows
        else ["ping", "-i", "0.2", "-c", "100000", target]
    )
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
    except OSError as error:
        print(f"  icmp probe unavailable: {error}", flush=True)
        return
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if stop.is_set():
                break
            text = line.rstrip("\n")
            if not text.strip():
                continue
            lowered = text.lower()
            seq = _SEQ.search(text)
            ttl = _TTL.search(text)
            writer.write(
                t=time.time() - t0,
                wall=time.time(),
                probe="icmp",
                rtt_ms=parse_ping_rtt_ms(text),
                seq=int(seq.group(1)) if seq else None,
                ttl=int(ttl.group(1)) if ttl else None,
                lost=("unreachable" in lowered or "timed out" in lowered),
                duplicate=("dup!" in lowered),
                raw=text,
            )
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()


#: Robot-side state pulled over the existing SSH connection. Output is stored
#: verbatim and never parsed at capture time, so a field nobody thought to
#: extract today is still there to extract later.
DEFAULT_REMOTE_PROBES: tuple[tuple[str, str], ...] = (
    ("net_dev", "cat /proc/net/dev"),
    ("wireless", "cat /proc/net/wireless 2>/dev/null || true"),
    ("loadavg", "cat /proc/loadavg"),
    ("uptime", "uptime"),
    ("temp", "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || true"),
    ("routes", "ip route 2>/dev/null || true"),
    ("modem", "mmcli -m any 2>/dev/null || true"),
    ("qmi_signal", "mmcli -m any --signal-get 2>/dev/null || true"),
)


def remote_state_probe(
    ssh_target: str, interval: float, writer: SampleWriter,
    stop: threading.Event, t0: float,
) -> None:
    """Pull robot-side state over the persistent SSH connection.

    Interface byte and error counters, load, temperature and modem status all
    live on the robot, not on the operator machine, and they are what explain a
    latency spike after the fact. Reusing the SSH master connection avoids
    needing any agent deployed on the robot.
    """

    # Deliberately no connection multiplexing: it is unsupported on Windows and
    # cost a full run of state data once. This pays a handshake every poll, which
    # is irrelevant here because only the *content* is wanted, never the timing.
    base = ["ssh", *SSH_BASE_OPTIONS, ssh_target]
    joined = " ; ".join(
        f"echo '==={name}==='; {command}" for name, command in DEFAULT_REMOTE_PROBES
    )
    while not stop.is_set():
        try:
            done = subprocess.run(
                [*base, joined], capture_output=True, text=True, timeout=20, check=False
            )
            writer.write(
                t=time.time() - t0,
                wall=time.time(),
                probe="remote_state",
                returncode=done.returncode,
                raw=done.stdout,
                stderr=done.stderr[:2000] or None,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            writer.write(
                t=time.time() - t0,
                wall=time.time(),
                probe="remote_state",
                returncode=None,
                raw=None,
                error=str(error),
            )
        stop.wait(interval)


DEFAULT_UDP_PORT = 47821


def udp_echo_probe(
    target: str, port: int, rate_hz: float, payload_bytes: int,
    writer: SampleWriter, stop: threading.Event, t0: float,
) -> None:
    """Numbered UDP probes against the robot-side agent.

    This is the closest analogue of an actual command: a small datagram sent to
    the robot, acted on, and acknowledged. Unlike ICMP it is not deprioritised
    by carriers, and because every probe carries a sequence number the robot's
    own log can later be compared against this one to say whether a missing
    probe was lost on the way out or on the way back.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.5)
    padding = "x" * max(0, payload_bytes - 120)
    period = 1.0 / rate_hz if rate_hz > 0 else 0.1
    seq = 0

    def receive_loop() -> None:
        while not stop.is_set():
            try:
                data, _ = sock.recvfrom(65535)
            except (TimeoutError, OSError):
                continue
            t_final = time.time()
            try:
                reply = json.loads(data.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue
            fields = {}
            try:
                fields = clock_offset_estimate(
                    float(reply["t_send"]),
                    float(reply["t_pi_recv"]),
                    float(reply["t_pi_send"]),
                    t_final,
                )
            except (KeyError, TypeError, ValueError):
                pass
            writer.write(
                t=t_final - t0, wall=t_final, probe="udp_reply",
                seq=reply.get("seq"), t_send=reply.get("t_send"),
                t_pi_recv=reply.get("t_pi_recv"), t_pi_send=reply.get("t_pi_send"),
                **fields,
            )

    reader = threading.Thread(target=receive_loop, daemon=True)
    reader.start()

    while not stop.is_set():
        now = time.time()
        message = json.dumps({"seq": seq, "t_send": now, "pad": padding})
        try:
            sock.sendto(message.encode("utf-8"), (target, port))
            writer.write(t=now - t0, wall=now, probe="udp_send", seq=seq,
                         bytes=len(message), failed=False)
        except OSError as error:
            writer.write(t=now - t0, wall=now, probe="udp_send", seq=seq,
                         failed=True, error=str(error))
        seq += 1
        stop.wait(period)

    reader.join(timeout=2)
    sock.close()


def tailscale_mode(status: dict[str, Any], peer_hint: str | None) -> tuple[str, dict[str, Any]]:
    """Determine whether the peer is reached directly or via a DERP relay.

    Tailscale prefers a direct WireGuard path and falls back to a relay when NAT
    traversal fails. On cellular this is not a rare edge case: carrier CGNAT
    frequently prevents a direct path, and the mode can change mid-session. A
    session recorded without this is a mixture of two very different links.
    """

    peers = status.get("Peer") or {}
    chosen: dict[str, Any] | None = None
    for peer in peers.values():
        if not isinstance(peer, dict):
            continue
        if peer_hint:
            haystack = " ".join(
                str(peer.get(k, "")) for k in ("HostName", "DNSName", "TailscaleIPs")
            )
            if peer_hint.lower() not in haystack.lower():
                continue
        if chosen is None or peer.get("Online"):
            chosen = peer
        if peer_hint:
            break

    if chosen is None:
        return "unknown", {}
    detail = {
        "hostname": chosen.get("HostName"),
        "online": chosen.get("Online"),
        "cur_addr": chosen.get("CurAddr"),
        "relay": chosen.get("Relay"),
        "rx_bytes": chosen.get("RxBytes"),
        "tx_bytes": chosen.get("TxBytes"),
    }
    if chosen.get("CurAddr"):
        return "direct", detail
    if chosen.get("Relay"):
        return f"derp:{chosen['Relay']}", detail
    return "unknown", detail


#: Underlay endpoint address ranges that mean "this never left the local
#: network". Tailscale reports the *underlay* endpoint it is using, so a
#: private address here means the two nodes found each other on the LAN
#: regardless of what the routing table says.
_PRIVATE_PREFIXES = ("10.", "192.168.", "127.", "169.254.", "::1", "fe80:")


def endpoint_class(cur_addr: str | None) -> str:
    """Classify a Tailscale underlay endpoint as ``lan``, ``wan`` or ``unknown``.

    This exists because a cellular run once recorded 100% of its traffic over
    WiFi while every pre-run check passed. The default route pointed at the
    modem the whole time; Tailscale simply preferred a direct LAN path to the
    peer and never used it. Routing tables cannot detect that. The underlay
    endpoint can, and it is the only field that can.
    """

    if not cur_addr:
        return "unknown"
    host = cur_addr.rsplit(":", 1)[0].strip("[]")
    if host.startswith(_PRIVATE_PREFIXES):
        return "lan"
    if host.startswith("172."):
        try:
            second = int(host.split(".")[1])
        except (IndexError, ValueError):
            return "unknown"
        return "lan" if 16 <= second <= 31 else "wan"
    return "wan"


def check_path_requirement(peer_hint: str | None, required: str) -> tuple[bool, str]:
    """Preflight the underlay path before a single sample is recorded.

    Refusing to start is deliberate. A run that took the wrong path is not
    partially useful — it is a different experiment wearing the wrong label,
    and it costs a battery charge and a driving session to discover afterwards.
    """

    if required == "any":
        return True, "no path requirement"
    try:
        done = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        status = json.loads(done.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        return False, f"could not read tailscale status: {error}"

    mode, detail = tailscale_mode(status, peer_hint)
    observed = endpoint_class(detail.get("cur_addr"))
    where = detail.get("cur_addr") or detail.get("relay") or "unknown"
    if mode.startswith("derp") and required == "wan":
        # A relay is off-LAN by definition, so it satisfies "not local".
        return True, f"relayed via {where} (satisfies --require-path wan)"
    if observed == required:
        return True, f"path is {observed} via {where}"
    return False, (
        f"path is {observed} via {where}, but --require-path {required} was given. "
        "For a cellular run this usually means WiFi is still associated and "
        "Tailscale is preferring the LAN; the default route does not control this."
    )


def tailscale_probe(
    peer_hint: str | None, interval: float, writer: SampleWriter,
    stop: threading.Event, t0: float,
) -> None:
    """Poll Tailscale connection state so every latency sample can be attributed."""

    while not stop.is_set():
        try:
            done = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            try:
                status = json.loads(done.stdout or "{}")
            except json.JSONDecodeError:
                status = {}
            mode, detail = tailscale_mode(status, peer_hint)
            writer.write(
                t=time.time() - t0,
                wall=time.time(),
                probe="tailscale",
                mode=mode,
                peer=detail,
                raw=done.stdout[:8000] or None,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            writer.write(
                t=time.time() - t0, wall=time.time(), probe="tailscale",
                mode="unavailable", peer={}, error=str(error),
            )
        stop.wait(interval)


def tcp_probe(
    target: str, port: int, interval: float, writer: SampleWriter,
    stop: threading.Event, t0: float,
) -> None:
    """Time a TCP handshake. Catches congestion and middlebox delay ICMP misses."""

    while not stop.is_set():
        started = time.perf_counter()
        try:
            with socket.create_connection((target, port), timeout=5.0):
                elapsed = (time.perf_counter() - started) * 1000.0
            writer.write(
                t=time.time() - t0, wall=time.time(), probe="tcp",
                connect_ms=elapsed, failed=False,
            )
        except OSError:
            writer.write(
                t=time.time() - t0, wall=time.time(), probe="tcp",
                connect_ms=None, failed=True,
            )
        stop.wait(interval)


SSH_BASE_OPTIONS = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=8",
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=2",
]


def ssh_probe(
    ssh_target: str, interval: float, writer: SampleWriter,
    stop: threading.Event, t0: float, _unused: Path | None = None,
) -> None:
    """Round-trip a line over one long-lived SSH session.

    We want the marginal cost of a command on a session the operator already
    holds open, not the cost of establishing one; a fresh ``ssh`` per probe would
    measure handshake and key exchange that a real operator never pays per
    keystroke.

    The earlier implementation got that persistence from OpenSSH connection
    multiplexing (``ControlMaster``/``ControlPath``). **Windows OpenSSH does not
    support multiplexing**, and it fails there with ``getsockname failed: Not a
    socket``, which silently cost an entire run's SSH data. This version instead
    holds a single ``ssh <target> cat`` open and times a line written to it and
    echoed back — a persistent session on every platform, with no multiplexing
    involved.
    """

    command = ["ssh", *SSH_BASE_OPTIONS, ssh_target, "cat"]
    process: subprocess.Popen[str] | None = None
    replies: queue.Queue[tuple[str, float]] = queue.Queue()

    def start_session() -> subprocess.Popen[str] | None:
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            print(f"  ssh probe unavailable: {error}", flush=True)
            return None

        def reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                replies.put((line.strip(), time.perf_counter()))

        threading.Thread(target=reader, daemon=True).start()
        return proc

    process = start_session()
    if process is None:
        return

    seq = 0
    while not stop.is_set():
        if process.poll() is not None:
            writer.write(
                t=time.time() - t0, wall=time.time(), probe="ssh",
                round_trip_ms=None, failed=True, reason="session_dropped",
            )
            stop.wait(interval)
            process = start_session()
            if process is None:
                return
            continue

        token = f"probe-{seq}"
        seq += 1
        started = time.perf_counter()
        try:
            assert process.stdin is not None
            process.stdin.write(token + "\n")
            process.stdin.flush()
        except (OSError, ValueError):
            writer.write(
                t=time.time() - t0, wall=time.time(), probe="ssh",
                round_trip_ms=None, failed=True, reason="write_failed",
            )
            stop.wait(interval)
            continue

        deadline = time.perf_counter() + 10.0
        matched = None
        while time.perf_counter() < deadline:
            try:
                line, at = replies.get(timeout=0.2)
            except queue.Empty:
                if stop.is_set():
                    break
                continue
            if line == token:
                matched = (at - started) * 1000.0
                break
            # A stale reply from a previous probe: discard and keep waiting.

        if matched is None:
            writer.write(
                t=time.time() - t0, wall=time.time(), probe="ssh",
                round_trip_ms=None, failed=True, reason="timeout", seq=seq - 1,
            )
        else:
            writer.write(
                t=time.time() - t0, wall=time.time(), probe="ssh",
                round_trip_ms=matched, failed=False, seq=seq - 1,
            )
        stop.wait(interval)

    if process is not None and process.poll() is None:
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()


def modem_probe(
    command: str, interval: float, writer: SampleWriter,
    stop: threading.Event, t0: float,
) -> None:
    """Poll a user-supplied command for modem signal metrics.

    Left deliberately open: modems expose signal data in incompatible ways, so
    rather than guess at the D501's interface the command is supplied by the
    operator and is only required to print JSON on stdout. Anything it prints is
    stored verbatim, so fields can be interpreted later without re-recording.
    """

    while not stop.is_set():
        try:
            done = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=10, check=False
            )
            payload: Any
            try:
                payload = json.loads(done.stdout.strip() or "null")
            except json.JSONDecodeError:
                payload = {"raw": done.stdout.strip()[:2000]}
            writer.write(
                t=time.time() - t0, wall=time.time(), probe="modem",
                data=payload, raw=done.stdout[:4000],
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            writer.write(
                t=time.time() - t0, wall=time.time(), probe="modem",
                data=None, error=str(error),
            )
        stop.wait(interval)


def _agent_attribution(
    args: argparse.Namespace, sent_seq: list[int], replied_seq: list[int]
) -> dict[str, Any] | str:
    """Attribute UDP loss to a direction, given the robot agent's own log."""

    path = getattr(args, "merge_agent", None)
    if path is None:
        return "requires the robot agent log; pass --merge-agent to compute it"
    if not path.exists():
        return f"agent log not found: {path}"

    robot_seq: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("seq") is not None:
                robot_seq.append(int(row["seq"]))

    result = attribute_loss(sent_seq, robot_seq, replied_seq)
    result["agent_log"] = str(path)
    return result


def summarize(samples_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    icmp_rtt, icmp_t, icmp_lost = [], [], 0
    tcp_ms, tcp_t, tcp_fail = [], [], 0
    ssh_ms, ssh_t, ssh_fail = [], [], 0
    modem_rows: list[dict[str, Any]] = []
    ts_t, ts_mode = [], []
    udp_sent_seq, udp_reply_seq, udp_rtt, udp_t = [], [], [], []

    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            probe = row.get("probe")
            if probe == "icmp":
                if row.get("lost"):
                    icmp_lost += 1
                elif row.get("rtt_ms") is not None:
                    icmp_rtt.append(row["rtt_ms"])
                    icmp_t.append(row["t"])
            elif probe == "tcp":
                if row.get("failed"):
                    tcp_fail += 1
                elif row.get("connect_ms") is not None:
                    tcp_ms.append(row["connect_ms"])
                    tcp_t.append(row["t"])
            elif probe == "ssh":
                if row.get("failed"):
                    ssh_fail += 1
                elif row.get("round_trip_ms") is not None:
                    ssh_ms.append(row["round_trip_ms"])
                    ssh_t.append(row["t"])
            elif probe == "modem":
                modem_rows.append(row)
            elif probe == "udp_send":
                if not row.get("failed"):
                    udp_sent_seq.append(row["seq"])
            elif probe == "udp_reply":
                if row.get("seq") is not None:
                    udp_reply_seq.append(row["seq"])
                if row.get("round_trip_ms") is not None:
                    udp_rtt.append(row["round_trip_ms"])
                    udp_t.append(row["t"])
            elif probe == "tailscale":
                ts_t.append(row["t"])
                ts_mode.append(row.get("mode") or "unknown")

    signal_correlation = None
    numeric_field = None
    if modem_rows and icmp_rtt:
        candidates = ("rsrp", "rsrq", "sinr", "rssi", "signal", "strength")
        for row in modem_rows:
            data = row.get("data")
            if isinstance(data, dict):
                for key in data:
                    if key.lower() in candidates and isinstance(data[key], (int, float)):
                        numeric_field = key
                        break
            if numeric_field:
                break
        if numeric_field:
            m_t = [r["t"] for r in modem_rows if isinstance(r.get("data"), dict)]
            m_v = [
                float(r["data"][numeric_field])
                for r in modem_rows
                if isinstance(r.get("data"), dict) and numeric_field in r["data"]
            ]
            if len(m_v) == len(m_t) and len(m_v) >= 3:
                signal_correlation = {
                    "field": numeric_field,
                    **correlate(icmp_t, icmp_rtt, m_t, m_v, window_s=2.0),
                }

    return {
        "schema_version": SCHEMA_VERSION,
        "label": args.label,
        "notes": args.notes,
        "disposition": (
            "Standalone remote-operation link measurement. Raw samples are in the "
            "companion .jsonl file; these summaries are derived and can be "
            "recomputed from it."
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "host": platform.node(),
        },
        "parameters": {
            "target": args.target,
            "ssh_target": args.ssh_target,
            "tcp_port": args.tcp_port,
            "duration_s": args.duration,
            "probe_interval_s": args.interval,
            "modem_command": args.modem_command,
        },
        "icmp": {
            "replies": len(icmp_rtt),
            "lost": icmp_lost,
            "loss_fraction": (
                icmp_lost / (icmp_lost + len(icmp_rtt))
                if (icmp_lost + len(icmp_rtt)) > 0
                else None
            ),
            "rtt": summarize_ms(icmp_rtt),
            "jitter": jitter_summary(icmp_rtt),
            "control_bands": control_quality_bands(icmp_rtt),
            "outages": outage_windows(icmp_t) if icmp_t else None,
            "by_tailscale_mode": (
                split_by_state(icmp_t, icmp_rtt, ts_t, ts_mode) if ts_t and icmp_t else None
            ),
        },
        "tcp_connect": {
            "successes": len(tcp_ms),
            "failures": tcp_fail,
            "connect": summarize_ms(tcp_ms),
        },
        "ssh_round_trip": {
            "successes": len(ssh_ms),
            "failures": ssh_fail,
            "round_trip": summarize_ms(ssh_ms),
            "jitter": jitter_summary(ssh_ms),
            "control_bands": control_quality_bands(ssh_ms),
            "by_tailscale_mode": (
                split_by_state(ssh_t, ssh_ms, ts_t, ts_mode) if ts_t and ssh_t else None
            ),
            "note": (
                "measured over a persistent connection, so it excludes handshake "
                "cost an operator holding a session would not pay"
            ),
        },
        "udp_echo": {
            "sent": len(udp_sent_seq),
            "replied": len(udp_reply_seq),
            "round_trip": summarize_ms(udp_rtt),
            "jitter": jitter_summary(udp_rtt),
            "reordering": reordering_summary(udp_reply_seq),
            "control_bands": control_quality_bands(udp_rtt),
            "by_tailscale_mode": (
                split_by_state(udp_t, udp_rtt, ts_t, ts_mode) if ts_t and udp_t else None
            ),
            "loss_attribution": _agent_attribution(args, udp_sent_seq, udp_reply_seq),
        },
        "tailscale": {
            "samples": len(ts_t),
            "modes_seen": sorted(set(ts_mode)),
            "changed_during_session": len(set(ts_mode)) > 1,
        },
        "modem": {
            "samples": len(modem_rows),
            "latency_vs_signal": signal_correlation,
        },
        "limitations": [
            "ICMP may be deprioritised or rate-limited by the carrier, so it is a "
            "floor rather than the experience of a real data stream.",
            "A single session captures one route, one time of day and one set of "
            "radio conditions; outdoor cellular varies enormously with all three.",
            "Correlation between latency and signal is association only and does "
            "not establish cause.",
            "Control-quality band edges are conventional teleoperation thresholds, "
            "not values measured here.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="configuration and run identifier")
    parser.add_argument("--target", required=True, help="robot address to probe")
    parser.add_argument("--ssh-target", default=None, help="user@host for the SSH probe")
    parser.add_argument("--tcp-port", type=int, default=22, help="port for the TCP probe")
    parser.add_argument("--duration", type=float, default=300.0, help="seconds to record")
    parser.add_argument("--interval", type=float, default=1.0, help="TCP/SSH probe interval")
    parser.add_argument(
        "--modem-command",
        default=None,
        help="shell command printing modem signal JSON on stdout, polled periodically",
    )
    parser.add_argument("--modem-interval", type=float, default=5.0)
    parser.add_argument(
        "--remote-state-interval",
        type=float,
        default=5.0,
        help="how often to pull robot-side interface, load and modem state over SSH",
    )
    parser.add_argument(
        "--no-remote-state",
        action="store_true",
        help="disable robot-side state collection (it needs a working SSH target)",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=DEFAULT_UDP_PORT,
        help="port of link_probe_agent.py on the robot; 0 disables the UDP probe",
    )
    parser.add_argument("--udp-rate", type=float, default=10.0, help="probes per second")
    parser.add_argument(
        "--udp-payload-bytes",
        type=int,
        default=200,
        help="datagram size; raise it to approximate a telemetry stream",
    )
    parser.add_argument(
        "--merge-agent",
        type=Path,
        default=None,
        metavar="AGENT_ECHO_JSONL",
        help="robot agent .echo.jsonl to merge, attributing loss to a direction",
    )
    parser.add_argument(
        "--tailscale-peer",
        default=None,
        help="hostname or IP fragment identifying the robot peer in tailscale status",
    )
    parser.add_argument("--tailscale-interval", type=float, default=3.0)
    parser.add_argument(
        "--no-tailscale",
        action="store_true",
        help="skip the connection-mode probe (use for combination 1, which has no tunnel)",
    )
    parser.add_argument("--notes", default="", help="route, weather, time of day, obstacles")
    parser.add_argument("--output", type=Path, required=True, help="output path without suffix")
    parser.add_argument(
        "--resummarize",
        type=Path,
        default=None,
        metavar="SAMPLES_JSONL",
        help=(
            "recompute the summary from an existing samples file. Use with "
            "--merge-agent after copying the robot agent log back."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.resummarize:
        if not args.resummarize.exists():
            parser.error(f"{args.resummarize} not found")
        summary = summarize(args.resummarize, args)
        out = args.output.with_suffix(".summary.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {out}")
        print(f"  udp loss attribution: {summary['udp_echo']['loss_attribution']}")
        return 0

    samples_path = args.output.with_suffix(".jsonl")
    summary_path = args.output.with_suffix(".summary.json")
    for path in (samples_path, summary_path):
        if path.exists() and not args.force:
            parser.error(f"{path} exists; refusing to overwrite recorded evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    writer = SampleWriter(samples_path)
    stop = threading.Event()
    t0 = time.time()
    threads = []

    threads.append(threading.Thread(target=icmp_probe, args=(args.target, writer, stop, t0)))
    threads.append(
        threading.Thread(
            target=tcp_probe,
            args=(args.target, args.tcp_port, args.interval, writer, stop, t0),
        )
    )
    if args.ssh_target:
        if shutil.which("ssh") is None:
            print("  ssh not found on PATH; skipping the ssh probe", flush=True)
        else:
            threads.append(
                threading.Thread(
                    target=ssh_probe,
                    args=(args.ssh_target, args.interval, writer, stop, t0),
                )
            )
            if not args.no_remote_state:
                threads.append(
                    threading.Thread(
                        target=remote_state_probe,
                        args=(
                            args.ssh_target, args.remote_state_interval,
                            writer, stop, t0,
                        ),
                    )
                )
    if args.udp_port:
        threads.append(
            threading.Thread(
                target=udp_echo_probe,
                args=(
                    args.target, args.udp_port, args.udp_rate,
                    args.udp_payload_bytes, writer, stop, t0,
                ),
            )
        )

    if not args.no_tailscale:
        if shutil.which("tailscale") is None:
            print("  tailscale not found on PATH; skipping connection-mode probe", flush=True)
        else:
            threads.append(
                threading.Thread(
                    target=tailscale_probe,
                    args=(args.tailscale_peer, args.tailscale_interval, writer, stop, t0),
                )
            )

    if args.modem_command:
        threads.append(
            threading.Thread(
                target=modem_probe,
                args=(args.modem_command, args.modem_interval, writer, stop, t0),
            )
        )

    for thread in threads:
        thread.daemon = True
        thread.start()

    print(f"recording for {args.duration:.0f} s - drive the robot now", flush=True)
    try:
        while True:
            remaining = args.duration - (time.time() - t0)
            if remaining <= 0:
                break
            time.sleep(min(5.0, remaining))
            print(f"  {time.time() - t0:5.0f}s elapsed", flush=True)
    except KeyboardInterrupt:
        print("\n  interrupted; summarising what was captured", flush=True)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=6)
        writer.close()

    summary = summarize(samples_path, args)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nsamples : {samples_path}")
    print(f"summary : {summary_path}")
    icmp = summary["icmp"]
    if icmp["rtt"]:
        print(
            f"  icmp  median {icmp['rtt']['median_ms']:7.1f} ms   "
            f"p95 {icmp['rtt']['p95_ms']:7.1f}   max {icmp['rtt']['max_ms']:8.1f}   "
            f"lost {icmp['lost']}"
        )
        for name, _, _ in CONTROL_BANDS_MS:
            fraction = icmp["control_bands"]["fractions"].get(name, 0.0)
            print(f"    {name:16s} {fraction * 100:5.1f}%")
    ts = summary["tailscale"]
    if ts["samples"]:
        print(f"  tailscale modes seen: {', '.join(ts['modes_seen'])}")
        if ts["changed_during_session"]:
            print("    NOTE: mode changed mid-session; see icmp.by_tailscale_mode")
    ssh = summary["ssh_round_trip"]
    if ssh["round_trip"]:
        print(
            f"  ssh   median {ssh['round_trip']['median_ms']:7.1f} ms   "
            f"p95 {ssh['round_trip']['p95_ms']:7.1f}   failures {ssh['failures']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
