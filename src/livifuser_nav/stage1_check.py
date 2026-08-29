"""Validate the local prerequisites and locked Stage 1 repository contract."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REQUIRED_PATHS = (
    "config/stage1.yaml",
    "config/calibration/camera_intrinsics.template.yaml",
    "config/calibration/lidar_camera_extrinsics.template.yaml",
    "ros2_ws/src/livifuser_bringup/package.xml",
)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    failures: list[str] = []

    print("LiViFuser-Nav Stage 1 readiness")
    print(f"  repository: {root}")
    print(f"  Python:     {sys.version.split()[0]}")

    for relative_path in REQUIRED_PATHS:
        exists = (root / relative_path).is_file()
        print(f"  {'OK' if exists else 'MISSING':7} {relative_path}")
        if not exists:
            failures.append(relative_path)

    for command in ("git", "uv", "ros2", "colcon"):
        location = shutil.which(command)
        status = location or "not installed on this host"
        print(f"  {command:7} {status}")

    if shutil.which("ros2") is None:
        print("\nROS 2 is expected on the Ubuntu robot/runtime host; see docs/STAGE1_SETUP.md.")

    if failures:
        print(f"\nStage 1 scaffold is incomplete ({len(failures)} required files missing).")
        return 1

    print("\nStage 1 scaffold is complete.")
    print("Hardware calibration and a pilot bag are still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
