"""Versioned rosbag QoS override shared by acquisition recorders."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

QOS_OVERRIDE_FILENAME = "rosbag_qos_overrides_v1.yaml"


def installed_qos_override_path() -> Path:
    """Return the installed recorder override without importing ROS at module load."""

    from ament_index_python.packages import get_package_share_directory

    return (
        Path(get_package_share_directory("livifuser_command_watchdog"))
        / "config"
        / QOS_OVERRIDE_FILENAME
    )


def build_record_command(
    *,
    storage_id: str,
    output_path: Path,
    topics: Iterable[str],
    qos_override_path: Path,
) -> list[str]:
    """Build a rosbag command that always applies the reviewed QoS override."""

    return [
        "ros2",
        "bag",
        "record",
        "-s",
        storage_id,
        "-o",
        str(output_path),
        "--qos-profile-overrides-path",
        str(qos_override_path),
        *topics,
    ]
