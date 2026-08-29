#!/usr/bin/env python3
"""Measure network delivery quality during a live teleoperation session.

**This script is observation-only and cannot move the robot.** It creates no
publisher of any kind; it only subscribes. Drive the robot with your existing
teleoperation setup and run this alongside it. Keeping the measurement out of
the control path is both safer and better measurement, because the tool does not
perturb the thing it is measuring.

What it records, per topic, as seen from wherever this script runs:

  * delivery latency  - arrival time minus the message's own header stamp, i.e.
                        how stale the data is by the time a consumer sees it
  * inter-arrival     - achieved rate and gaps, which exposes silent dropping
  * spikes            - excursions located in time, not merely counted
  * command path      - operator intent stamp to gated final command stamp
  * link latency      - a parallel ICMP ping, for reference only

Typical use, on the machine that would run the policy::

    python3 scripts/measure_network_session.py \\
        --label direct_d501_tailscale \\
        --ping-target 192.168.0.33 \\
        --duration 180 \\
        --output artifacts/network/session_direct_01.json

Then repeat with --label router_5g_sa over the other configuration and compare
with --compare.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from livifuser_nav.network_metrics import (  # noqa: E402
    PingSession,
    command_round_trip,
    compare_sessions,
    delivery_summary,
    network_cost,
)

SCHEMA_VERSION = "1.0.0"

#: Topics observed, with their nominal publication rate where one is defined.
#: A rate of None means delivered-fraction cannot be estimated and will be
#: reported as unknown rather than assumed complete.
OBSERVED_TOPICS: dict[str, dict[str, Any]] = {
    "/camera/image_raw": {"type": "sensor_msgs/msg/Image", "expected_hz": 30.0},
    "/scan": {"type": "sensor_msgs/msg/LaserScan", "expected_hz": 10.0},
    "/odom": {"type": "nav_msgs/msg/Odometry", "expected_hz": 20.0},
    "/livifuser/goal_relative": {
        "type": "livifuser_interfaces/msg/RelativeGoal",
        "expected_hz": 10.0,
    },
    "/livifuser/teleop_intent_stamped": {
        "type": "geometry_msgs/msg/TwistStamped",
        "expected_hz": 10.0,
    },
    "/livifuser/cmd_vel_stamped": {
        "type": "geometry_msgs/msg/TwistStamped",
        "expected_hz": 10.0,
    },
}

GAP_THRESHOLDS_MS = (150.0, 250.0, 500.0, 1000.0)

#: An excursion above this is treated as a spike. 100 ms is the architecture's
#: end-to-end budget, so a delivery latency above it has already consumed the
#: entire allowance before the policy has run.
SPIKE_THRESHOLD_MS = 100.0

#: Any publisher name matching these would defeat the purpose of the tool. It
#: creates none, and asserts so before exiting.
FORBIDDEN_PUBLISHER = re.compile(r"(^|/)cmd_vel|motor|wheel_?cmd", re.IGNORECASE)


def ping_worker(target: str, session: PingSession, stop: threading.Event) -> None:
    """Run a continuous ping and feed replies to the session until stopped."""

    flag = "-n" if platform.system() == "Windows" else "-c"
    interval = [] if platform.system() == "Windows" else ["-i", "0.2"]
    try:
        process = subprocess.Popen(
            ["ping", *interval, flag, "100000", target],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except OSError as error:  # ping missing, or blocked
        print(f"  ping unavailable ({error}); link latency will not be recorded", flush=True)
        return

    try:
        assert process.stdout is not None
        for line in process.stdout:
            if stop.is_set():
                break
            session.record_line(line)
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()


def collect(args: argparse.Namespace) -> dict[str, Any]:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from rosidl_runtime_py.utilities import get_message

    sensor_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=30,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )

    records: dict[str, dict[str, list[int]]] = {
        topic: {"arrival_ns": [], "header_ns": []} for topic in OBSERVED_TOPICS
    }

    class NetworkObserver(Node):
        """Subscribes only. Constructs no publisher."""

        def __init__(self) -> None:
            super().__init__("livifuser_network_observer")
            self._subs = []
            for topic, spec in OBSERVED_TOPICS.items():
                try:
                    message_class = get_message(spec["type"])
                except (ImportError, ValueError, AttributeError) as error:
                    print(f"  skipping {topic}: {error}", flush=True)
                    continue
                self._subs.append(
                    self.create_subscription(
                        message_class,
                        topic,
                        self._make_callback(topic),
                        sensor_qos,
                    )
                )

        def _make_callback(self, topic: str):
            store = records[topic]

            def callback(message) -> None:
                arrival = time.time_ns()
                header = getattr(message, "header", None)
                if header is None:
                    return
                stamp = header.stamp
                store["arrival_ns"].append(arrival)
                store["header_ns"].append(
                    int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
                )

            return callback

        def assert_no_publishers(self) -> list[str]:
            names = [name for name, _ in self.get_publisher_names_and_types_by_node(
                self.get_name(), self.get_namespace()
            )]
            offending = [n for n in names if FORBIDDEN_PUBLISHER.search(n)]
            if offending:
                raise RuntimeError(f"observer created forbidden publishers: {offending}")
            return names

    rclpy.init(args=None)
    node = NetworkObserver()

    ping = PingSession(target=args.ping_target) if args.ping_target else None
    stop = threading.Event()
    thread = None
    if ping is not None:
        thread = threading.Thread(target=ping_worker, args=(args.ping_target, ping, stop))
        thread.daemon = True
        thread.start()

    print(
        f"observing for {args.duration:.0f} s "
        f"(label: {args.label}) - drive the robot now with your normal teleop",
        flush=True,
    )
    started = time.time()
    try:
        while time.time() - started < args.duration:
            rclpy.spin_once(node, timeout_sec=0.2)
            elapsed = time.time() - started
            if int(elapsed) % 15 == 0 and elapsed > 1:
                counts = {t.split("/")[-1]: len(r["arrival_ns"]) for t, r in records.items()}
                print(f"  {elapsed:5.0f}s  {counts}", flush=True)
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n  interrupted; summarising what was captured", flush=True)
    finally:
        publisher_names = node.assert_no_publishers()
        stop.set()
        if thread is not None:
            thread.join(timeout=4)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    topics: dict[str, Any] = {}
    for topic, spec in OBSERVED_TOPICS.items():
        store = records[topic]
        topics[topic] = delivery_summary(
            store["arrival_ns"],
            store["header_ns"],
            expected_hz=spec["expected_hz"],
            gap_thresholds_ms=GAP_THRESHOLDS_MS,
            spike_threshold_ms=SPIKE_THRESHOLD_MS,
        )

    round_trip = command_round_trip(
        records["/livifuser/teleop_intent_stamped"]["header_ns"],
        records["/livifuser/cmd_vel_stamped"]["header_ns"],
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "label": args.label,
        "vantage": args.vantage,
        "notes": args.notes,
        "disposition": (
            "Observation-only network measurement taken during live "
            "teleoperation. This tool created no publisher and could not "
            "command motion."
        ),
        "safety": {
            "publishers_created": publisher_names,
            "forbidden_publisher_check": "passed",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "host": platform.node(),
        },
        "measurement_parameters": {
            "duration_s": args.duration,
            "gap_thresholds_ms": list(GAP_THRESHOLDS_MS),
            "spike_threshold_ms": SPIKE_THRESHOLD_MS,
            "spike_threshold_rationale": (
                "the architecture's end-to-end budget is 100 ms, so a delivery "
                "latency above this has consumed the whole allowance before the "
                "policy has run"
            ),
        },
        "topics": topics,
        "command_round_trip": round_trip,
        "link_latency": ping.summary(spike_threshold_ms=SPIKE_THRESHOLD_MS) if ping else None,
        "limitations": [
            "Delivery latency depends on clock agreement between the robot and "
            "this host; a skewed clock offsets every value by a constant.",
            "Link latency from ICMP is a floor, not the experience: it excludes "
            "serialisation, middleware queuing and subscriber scheduling.",
            "Delivered fraction assumes the nominal publication rate was actually "
            "achieved at the source; a slow publisher reads as a lossy link.",
            "A single session measures one set of radio conditions at one time of "
            "day and is not a general characterisation of the network.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        required=False,
        help="configuration being measured, e.g. router_5g_sa or direct_d501_tailscale",
    )
    parser.add_argument(
        "--vantage",
        choices=("robot", "host"),
        required=False,
        help=(
            "where this observer is running. 'robot' gives the baseline the "
            "sensor pipeline already has before any network; 'host' gives what "
            "a consumer across the link actually experiences"
        ),
    )
    parser.add_argument("--duration", type=float, default=180.0, help="seconds to observe")
    parser.add_argument("--ping-target", default=None, help="robot IP for the reference ping")
    parser.add_argument("--notes", default="", help="free-text conditions, location, time of day")
    parser.add_argument("--output", type=Path, help="JSON artifact to write")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("SESSION_A", "SESSION_B"),
        type=Path,
        help="compare two previously written session artifacts and exit",
    )
    parser.add_argument(
        "--network-cost",
        nargs=2,
        metavar=("ROBOT_LOCAL", "HOST_OBSERVED"),
        type=Path,
        help=(
            "subtract a robot-local baseline from a host measurement to isolate "
            "the latency actually added by the link, and exit"
        ),
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing artifact")
    args = parser.parse_args(argv)

    if args.compare:
        first, second = (json.loads(p.read_text(encoding="utf-8")) for p in args.compare)
        table = {
            "schema_version": SCHEMA_VERSION,
            "comparison": {
                topic: compare_sessions(
                    first["label"],
                    first["topics"].get(topic, {}),
                    second["label"],
                    second["topics"].get(topic, {}),
                )
                for topic in OBSERVED_TOPICS
            },
            "command_round_trip": {
                first["label"]: first.get("command_round_trip"),
                second["label"]: second.get("command_round_trip"),
            },
            "link_latency": {
                first["label"]: first.get("link_latency"),
                second["label"]: second.get("link_latency"),
            },
        }
        text = json.dumps(table, indent=2, sort_keys=True, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
            print(f"wrote {args.output}")
        else:
            print(text)
        return 0

    if args.network_cost:
        baseline, observed = (
            json.loads(p.read_text(encoding="utf-8")) for p in args.network_cost
        )
        if baseline.get("vantage") == "host" or observed.get("vantage") == "robot":
            parser.error(
                "argument order is ROBOT_LOCAL then HOST_OBSERVED; the recorded "
                f"vantages are {baseline.get('vantage')!r} then {observed.get('vantage')!r}"
            )
        result = {
            "schema_version": SCHEMA_VERSION,
            "baseline_label": baseline.get("label"),
            "observed_label": observed.get("label"),
            "per_topic": {
                topic: network_cost(
                    baseline["topics"].get(topic, {}),
                    observed["topics"].get(topic, {}),
                )
                for topic in OBSERVED_TOPICS
            },
            "caveat": (
                "valid only if both sessions covered comparable routes and "
                "traffic; a different route changes the load, not the link"
            ),
        }
        text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
            print(f"wrote {args.output}")
        else:
            print(text)
        return 0

    if not args.label or not args.output or not args.vantage:
        parser.error(
            "--label, --vantage and --output are required unless --compare or "
            "--network-cost is used"
        )
    if args.output.exists() and not args.force:
        parser.error(f"{args.output} exists; refusing to overwrite recorded evidence")

    payload = collect(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")

    for topic, summary in payload["topics"].items():
        latency = summary.get("delivery_latency")
        if not latency:
            print(f"  {topic:38s} no samples")
            continue
        fraction = summary.get("delivered_fraction")
        delivered = f"{fraction * 100:5.1f}%" if fraction is not None else "    ?"
        print(
            f"  {topic:38s} median {latency['median_ms']:7.1f} ms  "
            f"p95 {latency['p95_ms']:7.1f}  max {latency['max_ms']:8.1f}  "
            f"delivered {delivered}  spikes {summary['spike_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
