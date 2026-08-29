"""Motion-incapability policy for bag replay.

This lives in the tested library rather than in the replay script because it is
the guarantee that recorded velocity commands can never reach a motor interface.

The layers, strongest first:

1. A positive allowlist: only named sensor, goal, and TF topics are publishable,
   so no code path constructs a command publisher.
2. A forbidden-name assertion over every resolved publisher name, so a later edit
   to the allowlist cannot quietly reintroduce a command topic.
3. Network isolation and live-graph probing, applied by the script. Discovery
   races, so those can only ever be secondary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Isolated domain used for replay so traffic cannot reach the robot.
DEFAULT_REPLAY_DOMAIN_ID = "77"

#: source topic -> replayed topic. Sensors, goal, and TF only.
REPLAY_ALLOWLIST: dict[str, str] = {
    "/camera/image_raw": "/camera/image_raw",
    "/camera/camera_info": "/camera/camera_info",
    "/scan": "/scan",
    "/odom": "/odom",
    "/livifuser/goal_relative": "/livifuser/goal_relative",
    "/tf": "/tf",
    "/tf_static": "/tf_static",
}

#: Recorded commands, remapped to inert names for inspection only. Nothing on the
#: robot subscribes to these, and they are only published on explicit request.
REFERENCE_COMMAND_TOPICS: dict[str, str] = {
    "/cmd_vel": "/livifuser/replay/reference_cmd_vel",
    "/livifuser/cmd_vel_stamped": "/livifuser/replay/reference_cmd_vel_stamped",
    "/livifuser/teleop_intent_stamped": "/livifuser/replay/reference_teleop_intent",
}

#: Any resolved publisher name matching these is refused outright.
FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|/)cmd_vel($|/)"),
    re.compile(r"(^|/)cmd_vel_stamped($|/)"),
    re.compile(r"motor"),
    re.compile(r"wheel_?cmd"),
    re.compile(r"(^|/)joint_trajectory($|/)"),
)

#: Exactly the reference-inspection targets are exempt from the pattern check.
#: This is an exact-match set, never a prefix: a prefix exemption would also
#: excuse `/livifuser/replay/cmd_vel`, reopening the hole it was meant to close.
EXEMPT_PUBLISHERS: frozenset[str] = frozenset(
    {
        "/livifuser/replay/reference_cmd_vel",
        "/livifuser/replay/reference_cmd_vel_stamped",
        "/livifuser/replay/reference_teleop_intent",
    }
)

#: Node-name fragments indicating a live robot base is reachable.
ROBOT_NODE_FRAGMENTS: tuple[str, ...] = (
    "turtlebot3",
    "diff_drive",
    "dynamixel",
    "opencr",
    "hlds",
    "coin_d4",
)

#: Topics probed for live subscribers before any publishing begins.
COMMAND_TOPICS_TO_PROBE: tuple[str, ...] = (
    "/cmd_vel",
    "/cmd_vel_stamped",
    "/livifuser/cmd_vel_stamped",
)


class SafetyAuditError(RuntimeError):
    """Raised when a replay configuration is not provably motion-incapable."""


def is_forbidden_publisher(name: str) -> bool:
    """Whether publishing on ``name`` could plausibly reach a motor interface."""

    if name in EXEMPT_PUBLISHERS:
        return False
    return any(pattern.search(name) for pattern in FORBIDDEN_PATTERNS)


def assert_publisher_names_safe(names: list[str]) -> None:
    """Refuse any publisher whose name could reach a motor interface."""

    offenders = sorted({name for name in names if is_forbidden_publisher(name)})
    if offenders:
        raise SafetyAuditError(
            "refusing to publish on command-capable topics: " + ", ".join(offenders)
        )


def build_topic_map(
    bag_topics: dict[str, str], *, include_reference_commands: bool = False
) -> dict[str, str]:
    """Resolve the source -> published mapping from the allowlist only.

    Topics absent from the allowlist are never published, whether or not they
    appear in the bag.
    """

    mapping = {
        source: target
        for source, target in REPLAY_ALLOWLIST.items()
        if source in bag_topics
    }
    if include_reference_commands:
        mapping.update(
            {
                source: target
                for source, target in REFERENCE_COMMAND_TOPICS.items()
                if source in bag_topics
            }
        )
    assert_publisher_names_safe(list(mapping.values()))
    return mapping


@dataclass(frozen=True, slots=True)
class GraphProbe:
    """Result of the secondary live-robot discovery check."""

    robot_nodes: tuple[str, ...]
    command_subscribers: tuple[tuple[str, int], ...]

    @property
    def is_safe(self) -> bool:
        return not self.robot_nodes and not self.command_subscribers


def evaluate_graph(
    node_names: list[str], subscriber_counts: dict[str, int]
) -> GraphProbe:
    """Classify a discovered ROS graph as safe or robot-reachable."""

    robot_nodes = tuple(
        sorted(
            name
            for name in node_names
            if any(fragment in name.lower() for fragment in ROBOT_NODE_FRAGMENTS)
        )
    )
    command_subscribers = tuple(
        sorted(
            (topic, count) for topic, count in subscriber_counts.items() if count > 0
        )
    )
    return GraphProbe(robot_nodes=robot_nodes, command_subscribers=command_subscribers)
