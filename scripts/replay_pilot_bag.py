#!/usr/bin/env python3
"""Isolated, motion-incapable replay of a recorded pilot bag.

Purpose: exercise preprocessing, fusion, and inference against recorded sensor
data without powering or moving the robot.

The motion-incapability policy itself lives in
:mod:`livifuser_nav.replay_safety` so it is unit tested rather than trusted. This
script applies it and adds the two environment-level layers:

* `ROS_LOCALHOST_ONLY=1` and a dedicated `ROS_DOMAIN_ID`, set before the ROS
  context initializes, so replay traffic cannot reach the robot even if powered.
* A live-graph probe that aborts on visible TurtleBot nodes or command
  subscribers. Deliberately last, because DDS discovery races.

Default behaviour is audit-only: the topic map is printed and the process exits
without creating a single publisher. `--play` is required to publish.

Usage (on the ROS host):

    python3 scripts/replay_pilot_bag.py bags/stationary_pilot_2026-07-29_01
    python3 scripts/replay_pilot_bag.py bags/stationary_pilot_2026-07-29_01 --play
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from livifuser_nav.replay_safety import (  # noqa: E402
    COMMAND_TOPICS_TO_PROBE,
    DEFAULT_REPLAY_DOMAIN_ID,
    SafetyAuditError,
    build_topic_map,
    evaluate_graph,
)


def isolate_network(domain_id: str) -> dict[str, str]:
    """Pin the ROS context to an isolated, localhost-only domain.

    Must run before the ROS context initializes, because the middleware reads
    these variables at context creation.
    """

    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    # Iron and newer honour this instead of ROS_LOCALHOST_ONLY; set both so the
    # isolation does not silently lapse on a future distro.
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    os.environ["ROS_DOMAIN_ID"] = domain_id
    return {
        "ROS_DOMAIN_ID": os.environ["ROS_DOMAIN_ID"],
        "ROS_LOCALHOST_ONLY": os.environ["ROS_LOCALHOST_ONLY"],
        "ROS_AUTOMATIC_DISCOVERY_RANGE": os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"],
    }


def open_reader(bag: Path):  # noqa: ANN201 - rosbag2_py types are import-time
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    return reader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument(
        "--play",
        action="store_true",
        help="actually publish; without this the harness audits and exits",
    )
    parser.add_argument("--rate", type=float, default=1.0, help="playback rate multiplier")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--domain-id",
        default=DEFAULT_REPLAY_DOMAIN_ID,
        help=f"isolated ROS_DOMAIN_ID for replay (default {DEFAULT_REPLAY_DOMAIN_ID})",
    )
    parser.add_argument(
        "--include-reference-commands",
        action="store_true",
        help=(
            "also publish recorded commands under /livifuser/replay/* for inspection; "
            "never on a command topic"
        ),
    )
    parser.add_argument(
        "--discovery-wait",
        type=float,
        default=2.0,
        help="seconds to wait for graph discovery before the secondary safety probe",
    )
    args = parser.parse_args()

    if args.rate <= 0:
        print("--rate must be positive", file=sys.stderr)
        return 2
    if not args.bag.exists():
        print(f"bag not found: {args.bag}", file=sys.stderr)
        return 2

    isolation = isolate_network(args.domain_id)

    reader = open_reader(args.bag)
    bag_topics = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}

    try:
        topic_map = build_topic_map(
            bag_topics, include_reference_commands=args.include_reference_commands
        )
    except SafetyAuditError as error:
        print(f"SAFETY AUDIT FAILED: {error}", file=sys.stderr)
        return 3
    excluded = sorted(set(bag_topics) - set(topic_map))

    print("=" * 74)
    print("LiViFuser replay safety audit")
    print("=" * 74)
    print(f"bag                  : {args.bag}")
    for key, value in isolation.items():
        print(f"{key:21}: {value}")
    print(f"\npublished topics ({len(topic_map)}):")
    for source, target in sorted(topic_map.items()):
        arrow = "->" if source != target else "=="
        print(f"  {source:34} {arrow} {target}   [{bag_topics[source]}]")
    print(f"\nrecorded but NOT published ({len(excluded)}):")
    for topic in excluded:
        print(f"  {topic:34}    [{bag_topics[topic]}]")
    print("\nallowlist audit      : PASS (no command-capable publisher name)")

    if not args.play:
        print(
            "\nAudit only. No publisher was created and nothing was published.\n"
            "Re-run with --play to replay the topic map above."
        )
        return 0

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    rclpy.init()
    node = Node("livifuser_replay")
    exit_code = 0
    try:
        # The graph probe is mandatory. It is the weakest of the three layers
        # because DDS discovery races, which is a reason to keep it cheap, not a
        # reason to make it skippable.
        time.sleep(args.discovery_wait)
        visible = node.get_node_names()
        probe = evaluate_graph(
            visible,
            {topic: node.count_subscribers(topic) for topic in COMMAND_TOPICS_TO_PROBE},
        )
        if not probe.is_safe:
            print(
                "\nSAFETY ABORT: a live robot appears reachable."
                f"\n  robot nodes         : {list(probe.robot_nodes)}"
                f"\n  command subscribers : {list(probe.command_subscribers)}",
                file=sys.stderr,
            )
            return 4
        print(f"graph probe          : PASS ({len(visible)} nodes, no robot found)")

        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        message_types = {name: get_message(kind) for name, kind in bag_topics.items()}
        publishers = {
            source: node.create_publisher(
                message_types[source],
                target,
                latched_qos if source == "/tf_static" else sensor_qos,
            )
            for source, target in topic_map.items()
        }

        print(f"\nreplaying at {args.rate}x — Ctrl-C to stop\n")
        iteration = 0
        while True:
            iteration += 1
            published = 0
            wall_start = time.monotonic()
            bag_start: int | None = None
            while reader.has_next():
                topic, serialized, receive_ns = reader.read_next()
                if topic not in publishers:
                    continue
                if bag_start is None:
                    bag_start = receive_ns
                target_elapsed = (receive_ns - bag_start) / 1e9 / args.rate
                drift = target_elapsed - (time.monotonic() - wall_start)
                if drift > 0:
                    time.sleep(drift)
                publishers[topic].publish(
                    deserialize_message(serialized, message_types[topic])
                )
                published += 1
            print(f"pass {iteration}: published {published} messages")
            if not args.loop:
                break
            reader = open_reader(args.bag)
    except KeyboardInterrupt:
        print("\nstopped by operator")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
