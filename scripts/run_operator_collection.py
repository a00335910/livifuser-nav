#!/usr/bin/env python3
"""Operator-owned, plan-driven LiViFuser episode collection and offload.

Run this from WSL after sourcing ROS Humble and the local workspace. The script
uses SSH key authentication only. It does not contain or prompt for passwords.
"""

from __future__ import annotations

import argparse
import base64
import json
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from livifuser_nav.operator_collection import (  # noqa: E402
    CollectionEpisode,
    load_collection_plan,
    remote_episode_paths,
    sha256_file,
    write_json_exclusive,
)

DEFAULT_ROBOT = "a00335910@192.168.0.33"
DEFAULT_REMOTE_ROOT = "/home/a00335910/livifuser_bags"
DEFAULT_REMOTE_WORKSPACE = "/home/a00335910/ros2_ws"


def run(
    command: list[str], *, capture: bool = False, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def ssh(
    robot: str, script: str, *, capture: bool = False, check: bool = True
) -> subprocess.CompletedProcess[str]:
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        robot,
        f"echo {encoded} | base64 -d | bash",
    ]
    return run(command, capture=capture, check=check)


def unit_name(episode_id: str) -> str:
    return f"livifuser-episode-{episode_id}"


def ros_prelude(workspace: str) -> str:
    return (
        f"source /opt/ros/humble/setup.bash && source {shlex.quote(workspace)}/install/setup.bash"
    )


def start_infrastructure(robot: str, workspace: str) -> None:
    prelude = ros_prelude(workspace)
    watchdog = (
        f"{workspace}/install/livifuser_command_watchdog/lib/"
        "livifuser_command_watchdog/command_watchdog"
    )
    watchdog_config = (
        f"{workspace}/install/livifuser_command_watchdog/share/"
        "livifuser_command_watchdog/config/watchdog.yaml"
    )
    bringup_body = (
        prelude
        + " && export TURTLEBOT3_MODEL=burger LDS_MODEL=LDS-03"
        + " && exec ros2 launch turtlebot3_bringup robot.launch.py"
    )
    camera_body = (
        prelude
        + " && exec ros2 launch turtlebot3_bringup camera.launch.py"
        + " width:=320 height:=240 use_image_view:=false"
    )
    tf_body = prelude + " && exec ros2 launch livifuser_bringup camera_lidar_tf.launch.py"
    watchdog_body = (
        prelude + " && exec " + watchdog + " --ros-args --params-file " + watchdog_config
    )
    graph_probe_body = prelude + " && ros2 topic list"
    script = f"""set -euo pipefail
start_unit() {{
  local unit="$1"
  local body="$2"
  if systemctl --user is-active --quiet "$unit.service"; then
    return
  fi
  systemctl --user reset-failed "$unit.service" >/dev/null 2>&1 || true
  systemd-run --user --quiet --unit="$unit" --collect \\
    -p Restart=on-failure -p KillMode=mixed \\
    -p StandardOutput=journal -p StandardError=journal \\
    /bin/bash -lc "$body"
}}
start_unit livifuser-tb3-bringup {shlex.quote(bringup_body)}
start_unit livifuser-camera {shlex.quote(camera_body)}
start_unit livifuser-camera-tf {shlex.quote(tf_body)}
start_unit livifuser-command-watchdog {shlex.quote(watchdog_body)}
sleep 3
for unit in livifuser-tb3-bringup livifuser-camera \
  livifuser-camera-tf livifuser-command-watchdog; do
  systemctl --user is-active --quiet "$unit.service" || {{
    echo "$unit failed to start" >&2
    exit 1
  }}
done
ready=0
for attempt in $(seq 1 45); do
  topics="$(timeout 4 /bin/bash -lc {shlex.quote(graph_probe_body)} 2>/dev/null || true)"
  if grep -qx /cmd_vel <<<"$topics" && \
     grep -qx /scan <<<"$topics" && \
     grep -qx /odom <<<"$topics" && \
     grep -qx /camera/image_raw <<<"$topics"; then
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo "robot ROS graph did not become ready within the startup deadline" >&2
  exit 1
fi
"""
    ssh(robot, script)


def write_operator_sidecar(
    robot: str,
    path: PurePosixPath,
    document: dict[str, Any],
) -> None:
    payload = base64.b64encode(
        (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).decode("ascii")
    script = f"""set -euo pipefail
test ! -e {shlex.quote(str(path))}
set -o noclobber
echo {payload} | base64 -d > {shlex.quote(str(path))}
"""
    ssh(robot, script)


def start_episode(
    robot: str,
    workspace: str,
    remote_root: str,
    episode: CollectionEpisode,
    revision: str,
    operator_document: dict[str, Any],
) -> str:
    bag_path, result_path, operator_path = remote_episode_paths(remote_root, episode.episode_id)
    prelude = ros_prelude(workspace)
    goal_executable = (
        f"{workspace}/install/livifuser_goal_publisher/lib/"
        "livifuser_goal_publisher/odom_waypoint_goal_publisher"
    )
    manager_executable = (
        f"{workspace}/install/livifuser_command_watchdog/lib/"
        "livifuser_command_watchdog/episode_manager"
    )
    manager_config = (
        f"{workspace}/install/livifuser_command_watchdog/share/"
        "livifuser_command_watchdog/config/episode_manager.yaml"
    )
    unit = unit_name(episode.episode_id)
    probe = f"""set -euo pipefail
test -d {shlex.quote(remote_root)}
test ! -e {shlex.quote(str(bag_path))}
test ! -e {shlex.quote(str(result_path))}
test ! -e {shlex.quote(str(operator_path))}
! systemctl --user is-active --quiet {shlex.quote(unit + ".service")}
"""
    ssh(robot, probe)
    write_operator_sidecar(robot, operator_path, operator_document)

    goal_body = (
        prelude
        + " && exec "
        + goal_executable
        + " --ros-args -p forward_m:="
        + str(episode.forward_m)
        + " -p left_m:="
        + str(episode.left_m)
    )
    manager_args = [
        manager_executable,
        "--ros-args",
        "--params-file",
        manager_config,
        "-p",
        f"episode_id:={episode.episode_id}",
        "-p",
        f"output_path:={bag_path}",
        "-p",
        f"environment_id:={episode.environment_id}",
        "-p",
        f"split:={episode.split}",
        "-p",
        f"route_id:={episode.route_id}",
        "-p",
        f"layout_id:={episode.layout_id}",
        "-p",
        f"code_revision:={revision}",
        "-p",
        f"duration_s:={episode.duration_s}",
        "-p",
        "preflight_timeout_s:=60.0",
    ]
    manager_body = prelude + " && exec " + shlex.join(manager_args)
    script = f"""set -euo pipefail
systemctl --user stop livifuser-goal.service >/dev/null 2>&1 || true
systemctl --user reset-failed livifuser-goal.service >/dev/null 2>&1 || true
systemd-run --user --quiet --unit=livifuser-goal --collect \\
  -p Restart=on-failure -p StandardOutput=journal -p StandardError=journal \\
  /bin/bash -lc {shlex.quote(goal_body)}
systemd-run --user --quiet --unit={shlex.quote(unit)} --collect \\
  -p KillMode=mixed -p StandardOutput=journal -p StandardError=journal \\
  /bin/bash -lc {shlex.quote(manager_body)}
"""
    ssh(robot, script)
    return unit


def run_teleop(episode: CollectionEpisode) -> int:
    command = [
        "ros2",
        "run",
        "livifuser_command_watchdog",
        "release_keyboard_teleop",
        "--ros-args",
        "-p",
        f"episode_id:={episode.episode_id}",
        "-p",
        f"max_runtime_s:={episode.duration_s + 60.0}",
    ]
    return run(command, check=False).returncode


def wait_for_result(
    robot: str,
    remote_root: str,
    episode_id: str,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    _, result_path, _ = remote_episode_paths(remote_root, episode_id)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = ssh(
            robot,
            f"test -f {shlex.quote(str(result_path))} && cat {shlex.quote(str(result_path))}",
            capture=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {result_path}")


def stop_episode_units(robot: str, unit: str) -> None:
    ssh(
        robot,
        "systemctl --user stop "
        + shlex.quote(unit + ".service")
        + " livifuser-goal.service >/dev/null 2>&1 || true",
        check=False,
    )


def shutdown_robot_runtime(robot: str) -> None:
    units = (
        "livifuser-episode-*.service",
        "livifuser-goal.service",
        "livifuser-command-watchdog.service",
        "livifuser-camera-tf.service",
        "livifuser-camera.service",
        "livifuser-tb3-bringup.service",
    )
    ssh(robot, "systemctl --user stop " + " ".join(units) + " || true", check=False)


def completed_ids(robot: str, remote_root: str) -> set[str]:
    command = (
        f"find {shlex.quote(remote_root)} -maxdepth 1 -type f "
        "-name '*.episode.json' -printf '%f\\n'"
    )
    result = ssh(
        robot,
        command,
        capture=True,
    )
    suffix = ".episode.json"
    return {line[: -len(suffix)] for line in result.stdout.splitlines() if line.endswith(suffix)}


def show_episode(episode: CollectionEpisode) -> None:
    print("\n" + "=" * 72)
    print(f"Episode {episode.sequence}: {episode.episode_id}")
    print(f"Split/environment: {episode.split} / {episode.environment_id}")
    print(f"Route/layout: {episode.route_id} / {episode.layout_id}")
    print(
        f"Timer: {episode.duration_s:.1f} s   Goal offset: "
        f"({episode.forward_m:.2f}, {episode.left_m:.2f}) m"
    )
    print(f"Obstacles: {episode.obstacles}")
    print(f"Lighting: {episode.lighting}")
    print(f"Notes: {episode.route_notes}")


def resolve_revision(requested: str | None) -> str:
    if requested:
        return requested
    revision = run(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    dirty = run(
        ["git", "status", "--porcelain", "--untracked-files=no"], capture=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("tracked worktree changes exist; deploy and commit before collection")
    return revision


def session(args: argparse.Namespace) -> int:
    plan = load_collection_plan(args.plan, expected_count=args.expected_count)
    revision = resolve_revision(args.revision)
    existing = completed_ids(args.robot, args.remote_root)
    selected = [
        episode
        for episode in plan.pending(existing)
        if args.start_at is None or episode.sequence >= args.start_at
    ]
    if not selected:
        print("No pending episodes in the selected range.")
        return 0
    print(f"{len(selected)} pending of {len(plan.episodes)} planned episodes.")
    infrastructure_started = False
    try:
        for episode in selected:
            episode.require_confirmed()
            show_episode(episode)
            duration_text = input(
                f"Duration seconds [{episode.duration_s:g}] (Enter keeps plan): "
            ).strip()
            if duration_text:
                episode = replace(episode, duration_s=float(duration_text))
                episode.validate()
                print(f"This episode's robot-local deadline is {episode.duration_s:.1f} s.")
            print("Place the robot at the marked start and arrange the listed layout.")
            confirmation = input(
                f"Type ARM {episode.episode_id} to authorize this exact episode, or q: "
            ).strip()
            if confirmation.lower() == "q":
                break
            if confirmation != f"ARM {episode.episode_id}":
                print("Authorization did not match; episode skipped.")
                continue
            authorized_time = datetime.now().astimezone().isoformat()
            operator_document = episode.operator_record(
                revision=revision,
                authorized_wall_time=authorized_time,
            )
            if not infrastructure_started:
                print("Starting collection infrastructure once for this session...")
                start_infrastructure(args.robot, args.remote_workspace)
                infrastructure_started = True
            unit = start_episode(
                args.robot,
                args.remote_workspace,
                args.remote_root,
                episode,
                revision,
                operator_document,
            )
            try:
                teleop_code = run_teleop(episode)
                result = wait_for_result(
                    args.robot,
                    args.remote_root,
                    episode.episode_id,
                    timeout_s=episode.duration_s + 60.0,
                )
            finally:
                stop_episode_units(args.robot, unit)
            completed = bool(result.get("episode_manager_completed"))
            print(
                f"Saved {episode.episode_id}: lifecycle={'complete' if completed else 'REJECTED'}, "
                f"reason={result.get('reason')}, teleop_rc={teleop_code}."
            )
            print("Deep validation is deferred; the raw episode remains preserved on the Pi.")
            action = input("Press Enter for the next episode, or q to end the session: ").strip()
            if action.lower() == "q":
                break
    except KeyboardInterrupt:
        print("\nOperator interrupted the session; stopping runtime services.")
        return 130
    finally:
        if not args.leave_infrastructure:
            shutdown_robot_runtime(args.robot)
            print("Robot collection services stopped.")
        elif infrastructure_started:
            print(
                "Infrastructure intentionally left running; watchdog output remains locally gated."
            )
    return 0


def remote_hashes(robot: str, remote_root: str, episode_id: str) -> dict[str, str]:
    paths = remote_episode_paths(remote_root, episode_id)
    names = [path.name for path in paths]
    find_args = " ".join(shlex.quote(name) for name in names)
    script = f"""set -euo pipefail
cd {shlex.quote(remote_root)}
find -- {find_args} -type f -print0 | sort -z | xargs -0 sha256sum
"""
    result = ssh(robot, script, capture=True)
    hashes: dict[str, str] = {}
    for line in result.stdout.splitlines():
        digest, relative = line.split(maxsplit=1)
        hashes[relative.lstrip("*")] = digest
    if not hashes:
        raise RuntimeError(f"no remote files found for {episode_id}")
    return hashes


def offload(args: argparse.Namespace) -> int:
    plan = load_collection_plan(args.plan, expected_count=args.expected_count)
    requested = set(args.episodes.split(",")) if args.episodes else set()
    episodes = [
        episode for episode in plan.episodes if not requested or episode.episode_id in requested
    ]
    unknown = requested - {episode.episode_id for episode in episodes}
    if unknown:
        raise ValueError(f"unknown requested episode IDs: {sorted(unknown)}")
    args.destination.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        remote = remote_hashes(args.robot, args.remote_root, episode.episode_id)
        bag_path, result_path, operator_path = remote_episode_paths(
            args.remote_root, episode.episode_id
        )
        local_bag = args.destination / episode.episode_id
        local_result = args.destination / result_path.name
        local_operator = args.destination / operator_path.name
        manifest_path = args.destination / f"{episode.episode_id}.offload.json"
        if any(path.exists() for path in (local_bag, local_result, local_operator, manifest_path)):
            raise FileExistsError(f"offload destination already contains {episode.episode_id}")
        run(["scp", "-pr", f"{args.robot}:{bag_path}", str(args.destination)])
        run(["scp", "-p", f"{args.robot}:{result_path}", str(local_result)])
        run(["scp", "-p", f"{args.robot}:{operator_path}", str(local_operator)])
        local: dict[str, str] = {}
        for relative in remote:
            local_path = args.destination / relative
            if not local_path.is_file():
                raise RuntimeError(f"copied file missing: {local_path}")
            local[relative] = sha256_file(local_path)
        if local != remote:
            raise RuntimeError(f"hash mismatch after offload: {episode.episode_id}")
        journal_unit = shlex.quote(unit_name(episode.episode_id) + ".service")
        journal_command = f"journalctl --user -u {journal_unit} --no-pager --output=short-iso"
        journal = ssh(
            args.robot,
            journal_command,
            capture=True,
            check=False,
        ).stdout
        journal_path = args.destination / f"{episode.episode_id}.journal.txt"
        journal_path.write_text(journal, encoding="utf-8", newline="\n")
        manifest = {
            "schema_version": "1.0.0",
            "episode_id": episode.episode_id,
            "robot": args.robot,
            "remote_root": args.remote_root,
            "offloaded_wall_time": datetime.now().astimezone().isoformat(),
            "verified": True,
            "files": remote,
            "journal_file": journal_path.name,
            "journal_sha256": sha256_file(journal_path),
        }
        write_json_exclusive(manifest_path, manifest)
        print(f"Offloaded and hash-verified: {episode.episode_id}")
    print("Pi originals were preserved. Use the separate purge command only after backup.")
    return 0


def purge(args: argparse.Namespace) -> int:
    paths = remote_episode_paths(args.remote_root, args.episode)
    manifest_path = args.destination / f"{args.episode}.offload.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("episode_id") != args.episode or manifest.get("verified") is not True:
        raise ValueError("offload manifest is not a verified match for the requested episode")
    expected = dict(manifest.get("files", {}))
    local = {relative: sha256_file(args.destination / relative) for relative in expected}
    if local != expected:
        raise RuntimeError("local offload no longer matches its verified manifest")
    current_remote = remote_hashes(args.robot, args.remote_root, args.episode)
    if current_remote != expected:
        raise RuntimeError("Pi originals no longer match the verified offload")
    exact = str(paths[0])
    confirmation = input(
        f"Type DELETE {exact} to remove this exact Pi episode and sidecars: "
    ).strip()
    if confirmation != f"DELETE {exact}":
        print("Deletion cancelled.")
        return 1
    quoted = " ".join(shlex.quote(str(path)) for path in paths)
    ssh(args.robot, f"rm -rf -- {quoted}")
    check = ssh(
        args.robot,
        "test ! -e " + " && test ! -e ".join(shlex.quote(str(path)) for path in paths),
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError("one or more requested Pi paths remain after deletion")
    print(f"Deleted exact verified Pi episode: {exact}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default=DEFAULT_ROBOT)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    session_parser = subparsers.add_parser("session", help="record pending plan episodes")
    session_parser.add_argument("--plan", type=Path, required=True)
    session_parser.add_argument(
        "--revision",
        help="deployed Git revision; default current clean HEAD",
    )
    session_parser.add_argument("--expected-count", type=int, default=30)
    session_parser.add_argument("--start-at", type=int)
    session_parser.add_argument("--remote-workspace", default=DEFAULT_REMOTE_WORKSPACE)
    session_parser.add_argument("--leave-infrastructure", action="store_true")
    session_parser.set_defaults(function=session)

    offload_parser = subparsers.add_parser("offload", help="copy and hash-verify episodes")
    offload_parser.add_argument("--plan", type=Path, required=True)
    offload_parser.add_argument("--destination", type=Path, required=True)
    offload_parser.add_argument("--episodes", help="comma-separated episode IDs; default all")
    offload_parser.add_argument("--expected-count", type=int, default=30)
    offload_parser.set_defaults(function=offload)

    purge_parser = subparsers.add_parser("purge", help="delete one exact verified Pi episode")
    purge_parser.add_argument("--episode", required=True)
    purge_parser.add_argument("--destination", type=Path, required=True)
    purge_parser.set_defaults(function=purge)

    shutdown_parser = subparsers.add_parser("shutdown", help="stop collection runtime services")
    shutdown_parser.set_defaults(function=lambda args: (shutdown_robot_runtime(args.robot), 0)[1])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.function(args))
    except (
        OSError,
        ValueError,
        RuntimeError,
        TimeoutError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
