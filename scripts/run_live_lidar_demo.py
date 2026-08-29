#!/usr/bin/env python3
"""Live read-only policy overlay for the teleop demonstration.

The operator drives the robot with the existing keyboard path. This process
subscribes to the sensor and command streams, runs a trained policy beside them
at 10 Hz, and serves a browser dashboard showing what the policy *would* have
commanded, with its predictive uncertainty, next to what the operator actually
commanded.

It is motion-incapable by construction: this process creates no publisher of its
own, and at startup the node's resolved publisher list is asserted to hold
nothing beyond the `/rosout` and `/parameter_events` pair rclpy creates for every
node. The camera is not used at all, so this runs with the Camera Module v3
ribbon failure unrepaired.

Two sources are supported:

    --source ros      subscribe to a live robot (requires a sourced ROS 2 env)
    --source replay   step a recorded export at 10 Hz, for dry runs off-robot

Example, on the ROS host after sourcing both setups:

    python3 scripts/run_live_lidar_demo.py --source ros \
        --checkpoint artifacts/experiments/<sweep>/<fold>/lidar_only/<seed>/checkpoint.pt

Example dry run, no robot and no ROS:

    python scripts/run_live_lidar_demo.py --source replay \
        --replay-export artifacts/export/protocol_clean_30/<run> \
        --checkpoint <path>/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np  # noqa: E402

from livifuser_nav.live_overlay import (  # noqa: E402
    TICK_PERIOD_S,
    LiveWindow,
    OverlayConfig,
    agreement_error,
    assert_publishers_are_inert,
    calibration_context,
    goal_xy,
    rollout_unicycle,
    scan_points,
    sigma_from_log_variance,
    tokenize_live_scan,
)

DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "baseline_sweep_pilot5_v1.json"
DASHBOARD = REPOSITORY_ROOT / "assets" / "demo" / "live_dashboard.html"

#: Topics this process subscribes to. Subscribing cannot move the robot; the
#: list is explicit so the demonstration can state exactly what it consumes.
SCAN_TOPIC = "/scan"
ODOM_TOPIC = "/odom"
GOAL_TOPIC = "/livifuser/goal_relative"
ACTION_TOPIC = "/livifuser/cmd_vel_stamped"


def round_list(values: Any, digits: int = 3) -> list:
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


class PolicyRunner:
    """Loads one frozen checkpoint and forwards a batch-of-one window."""

    def __init__(self, checkpoint_path: str | Path, threads: int = 2) -> None:
        import torch  # imported lazily: the module map keeps torch off the ROS path

        self._torch = torch
        torch.set_num_threads(int(threads))
        from livifuser_nav.model import LiViFuserPolicy

        payload = torch.load(str(checkpoint_path), map_location="cpu")
        self.variant = str(payload["variant"])
        self.seed = payload.get("seed")
        self.config_sha256 = payload.get("config_sha256")
        self.model = LiViFuserPolicy(variant=self.variant)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def forward(self, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
        torch = self._torch
        tensors = {key: torch.from_numpy(value) for key, value in arrays.items()}
        started = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**tensors)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        mean = outputs["mean"][0].numpy().astype(np.float64)
        log_variance = outputs["log_variance"][0].numpy().astype(np.float64)
        return mean, log_variance, elapsed_ms


class FrameStore:
    """Newest dashboard frame, published to pollers under one lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: dict[str, Any] = {"tick": 0, "ready": False, "status": "starting"}

    def set(self, frame: dict[str, Any]) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> dict[str, Any]:
        with self._lock:
            return self._frame


class GoalStore:
    """Operator-settable synthetic goal, used when no goal topic is present.

    This only ever reaches the policy's input. It is not a navigation command
    and nothing subscribes to it.
    """

    def __init__(self, rho_m: float, alpha_rad: float) -> None:
        self._lock = threading.Lock()
        self._rho = float(rho_m)
        self._alpha = float(alpha_rad)

    def set(self, rho_m: float, alpha_rad: float) -> None:
        if not math.isfinite(rho_m) or rho_m < 0.0:
            raise ValueError("rho_m must be finite and non-negative")
        if not math.isfinite(alpha_rad):
            raise ValueError("alpha_rad must be finite")
        with self._lock:
            self._rho = float(rho_m)
            self._alpha = float(alpha_rad)

    def get(self) -> tuple[float, float, float]:
        with self._lock:
            return self._rho, math.sin(self._alpha), math.cos(self._alpha)


def make_handler(store: FrameStore, goals: GoalStore, dashboard: Path):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # keep the console readable
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path.startswith("/state"):
                body = json.dumps(store.get()).encode("utf-8")
                self._send(200, body, "application/json")
                return
            if self.path in ("/", "/index.html"):
                self._send(200, dashboard.read_bytes(), "text/html; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            if not self.path.startswith("/goal"):
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                goals.set(
                    float(payload["rho_m"]), math.radians(float(payload["alpha_deg"]))
                )
            except (ValueError, KeyError, TypeError) as error:
                self._send(400, str(error).encode("utf-8"), "text/plain")
                return
            self._send(200, b'{"ok":true}', "application/json")

    return Handler


class ReplaySource:
    """Steps a recorded export so the whole path can be exercised off-robot."""

    def __init__(self, export_root: str | Path) -> None:
        from livifuser_nav.learning_data import ExportRun

        self.run = ExportRun(export_root, load_rgb=False)
        self.context = {
            "calibration": self.run.manifest["calibration"],
            "run_id": self.run.run_id,
        }
        self.row = 0
        self.sequence = 0
        self.label = f"replay:{self.run.run_id}"
        self.failure_reason: str | None = None

    def is_healthy(self) -> bool:
        """A recorded export cannot stop producing rows."""

        return True

    def sample(self) -> dict[str, Any] | None:
        if self.row >= self.run.count:
            self.row = 0
        row = self.row
        self.row += 1
        self.sequence += 1
        beams = int(self.run.vectors["scan_beam_count"][row])
        return {
            "sequence": self.sequence,
            "ranges": np.asarray(self.run.scan_ranges[row][:beams], dtype=np.float64),
            "angle_increment": float(self.run.vectors["scan_angle_increment_rad"][row]),
            "goal": tuple(float(v) for v in self.run.vectors["goal"][row]),
            "robot_state": tuple(float(v) for v in self.run.vectors["robot_state"][row]),
            "operator": tuple(float(v) for v in self.run.vectors["action"][row]),
            "operator_age_s": 0.0,
            "row": row,
        }


class RosSource:
    """Subscribe-only live source. Constructs no publisher of any kind."""

    def __init__(self, use_goal_topic: bool, goals: GoalStore) -> None:
        import rclpy
        from geometry_msgs.msg import TwistStamped
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan

        self._rclpy = rclpy
        rclpy.init(args=None)
        self.node = rclpy.create_node("livifuser_live_overlay")
        self.goals = goals
        self.use_goal_topic = use_goal_topic
        self._lock = threading.Lock()
        self._scan: Any = None
        self._scan_sequence = 0
        self._odom = (0.0, 0.0)
        self._operator = (0.0, 0.0)
        self._operator_stamp = 0.0
        self._goal: tuple[float, float, float] | None = None
        self.label = "ros:live"
        self.failure_reason: str | None = None

        self.node.create_subscription(
            LaserScan, SCAN_TOPIC, self._on_scan, qos_profile_sensor_data
        )
        self.node.create_subscription(Odometry, ODOM_TOPIC, self._on_odom, 10)
        self.node.create_subscription(
            TwistStamped, ACTION_TOPIC, self._on_action, 10
        )
        if use_goal_topic:
            from livifuser_interfaces.msg import RelativeGoal

            self.node.create_subscription(RelativeGoal, GOAL_TOPIC, self._on_goal, 10)

        published = [
            name
            for name, _types in self.node.get_publisher_names_and_types_by_node(
                self.node.get_name(), self.node.get_namespace()
            )
        ]
        assert_publishers_are_inert(published)
        self.publishers = sorted(published)

        self._spin = threading.Thread(target=self._spin_forever, daemon=True)
        self._spin.start()

    def _spin_forever(self) -> None:
        # rclpy installs its own SIGINT/SIGTERM handler and shuts the context
        # down beneath us. Without recording that, this thread dies quietly, no
        # callback ever fires again, and the tick loop keeps serving a dashboard
        # that resets its window forever instead of reporting a dead source.
        try:
            self._rclpy.spin(self.node)
        except BaseException as error:  # noqa: BLE001 - recorded, then reported
            self.failure_reason = f"{type(error).__name__}: {error}".strip(": ")

    def is_healthy(self) -> bool:
        """Whether ROS callbacks can still arrive."""

        if not self._spin.is_alive():
            self.failure_reason = self.failure_reason or "the ROS spin thread stopped"
            return False
        if not self._rclpy.ok():
            self.failure_reason = self.failure_reason or "the ROS context shut down"
            return False
        return True

    def _on_scan(self, message: Any) -> None:
        with self._lock:
            self._scan = (
                np.asarray(message.ranges, dtype=np.float64),
                float(message.angle_increment),
            )
            self._scan_sequence += 1

    def _on_odom(self, message: Any) -> None:
        with self._lock:
            self._odom = (
                float(message.twist.twist.linear.x),
                float(message.twist.twist.angular.z),
            )

    def _on_action(self, message: Any) -> None:
        with self._lock:
            self._operator = (
                float(message.twist.linear.x),
                float(message.twist.angular.z),
            )
            self._operator_stamp = time.monotonic()

    def _on_goal(self, message: Any) -> None:
        with self._lock:
            self._goal = (
                float(message.rho_m),
                float(message.sin_alpha),
                float(message.cos_alpha),
            )

    def sample(self) -> dict[str, Any] | None:
        with self._lock:
            if self._scan is None:
                return None
            ranges, increment = self._scan
            goal = self._goal if (self.use_goal_topic and self._goal) else self.goals.get()
            age = time.monotonic() - self._operator_stamp if self._operator_stamp else None
            return {
                "sequence": self._scan_sequence,
                "ranges": ranges,
                "angle_increment": increment,
                "goal": goal,
                "robot_state": self._odom,
                "operator": self._operator,
                "operator_age_s": age,
                "row": None,
            }

    def shutdown(self) -> None:
        self.node.destroy_node()
        self._rclpy.shutdown()


def build_frame(
    *,
    tick: int,
    started: float,
    sample: dict[str, Any],
    tokens: Any,
    window: LiveWindow,
    runner: PolicyRunner | None,
    context: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    xs, ys = scan_points(sample["ranges"], sample["angle_increment"], context)
    goal = sample["goal"]
    frame: dict[str, Any] = {
        "tick": tick,
        "time_s": round(time.monotonic() - started, 3),
        "ready": window.ready,
        "window_depth": window.depth,
        "window_resets": window.resets,
        "beam_count": int(np.asarray(sample["ranges"]).shape[0]),
        "scan": {"x": round_list(xs, 3), "y": round_list(ys, 3)},
        "sectors": {
            "range_m": round_list(
                np.asarray(tokens.features)[:, 0] * identity["range_clip_m"], 3
            ),
            "bearing_rad": round_list(tokens.sector_bearing_rad, 4),
            "in_fov": np.asarray(tokens.in_fov, dtype=bool).tolist(),
            "validity": round_list(np.asarray(tokens.features)[:, 3], 3),
        },
        "goal": {
            "rho_m": round(float(goal[0]), 3),
            "alpha_deg": round(math.degrees(math.atan2(goal[1], goal[2])), 2),
            "xy": round_list(goal_xy(goal), 3),
        },
        "operator": {
            "linear_mps": round(float(sample["operator"][0]), 4),
            "angular_radps": round(float(sample["operator"][1]), 4),
            "age_s": (
                None if sample["operator_age_s"] is None
                else round(float(sample["operator_age_s"]), 3)
            ),
            "path": round_list(
                rollout_unicycle(
                    np.tile(np.asarray(sample["operator"], dtype=np.float64), (8, 1))
                )[:, :2],
                3,
            ),
        },
        "robot_state": {
            "linear_mps": round(float(sample["robot_state"][0]), 4),
            "angular_radps": round(float(sample["robot_state"][1]), 4),
        },
        "identity": identity,
        "status": "running",
    }
    if not (window.ready and runner is not None):
        frame["policy"] = None
        frame["status"] = "filling window"
        return frame

    mean, log_variance, elapsed_ms = runner.forward(window.arrays())
    sigma = sigma_from_log_variance(log_variance)
    frame["policy"] = {
        "mean": round_list(mean, 4),
        "sigma": round_list(sigma, 4),
        "path": round_list(rollout_unicycle(mean)[:, :2], 3),
        "path_high": round_list(rollout_unicycle(mean + sigma)[:, :2], 3),
        "path_low": round_list(rollout_unicycle(mean - sigma)[:, :2], 3),
        "inference_ms": round(elapsed_ms, 2),
    }
    frame["agreement"] = {
        key: round(value, 4)
        for key, value in agreement_error(mean, sample["operator"]).items()
    }
    return frame


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("ros", "replay"), default="ros")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--calibration-manifest",
        help=(
            "Export manifest whose calibration the live tokenizer reuses. "
            "Defaults to the replay export in replay mode; required for ROS."
        ),
    )
    parser.add_argument("--replay-export")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--goal-rho", type=float, default=1.0)
    parser.add_argument("--goal-alpha-deg", type=float, default=0.0)
    parser.add_argument(
        "--goal-source", choices=("fixed", "topic"), default="fixed",
        help="fixed uses the dashboard-settable synthetic goal; topic uses "
             f"{GOAL_TOPIC} when the goal publisher is running",
    )
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--dashboard", default=str(DASHBOARD))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    sweep_config = json.loads(Path(arguments.config).read_text("utf-8"))
    overlay_config = OverlayConfig.from_sweep_config(sweep_config)
    goals = GoalStore(arguments.goal_rho, math.radians(arguments.goal_alpha_deg))

    if arguments.source == "replay":
        if not arguments.replay_export:
            raise SystemExit("--replay-export is required with --source replay")
        source: Any = ReplaySource(arguments.replay_export)
        context = source.context
    else:
        if not arguments.calibration_manifest:
            raise SystemExit(
                "--calibration-manifest is required with --source ros; pass the "
                "manifest.json of the export the checkpoint was trained on"
            )
        context = calibration_context(arguments.calibration_manifest)
        source = RosSource(arguments.goal_source == "topic", goals)

    runner = PolicyRunner(arguments.checkpoint, threads=arguments.torch_threads)
    window = LiveWindow(overlay_config)
    identity = {
        "variant": runner.variant,
        "seed": runner.seed,
        "parameter_count": runner.parameter_count,
        "checkpoint": str(Path(arguments.checkpoint).as_posix()),
        "calibration_run_id": context.get("run_id"),
        "source": source.label,
        "context_k": overlay_config.context_k,
        "horizon_h": overlay_config.horizon_h,
        "lidar_sectors": overlay_config.lidar_sectors,
        "range_clip_m": overlay_config.lidar_range_clip_m,
        "tick_hz": round(1.0 / TICK_PERIOD_S, 2),
        "camera_used": False,
    }

    store = FrameStore()
    handler = make_handler(store, goals, Path(arguments.dashboard))
    server = ThreadingHTTPServer((arguments.host, arguments.port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"dashboard:   http://localhost:{arguments.port}/", flush=True)
    print(
        f"policy:      {runner.variant} ({runner.parameter_count} parameters, CPU)",
        flush=True,
    )
    print(f"source:      {source.label}", flush=True)
    publishers = getattr(source, "publishers", [])
    print(
        "publishers:  "
        + (", ".join(publishers) if publishers else "none")
        + " — no command topic; this process cannot move the robot",
        flush=True,
    )

    started = time.monotonic()
    tick = 0
    last_sequence = -1
    next_tick = time.monotonic()
    try:
        while True:
            next_tick += TICK_PERIOD_S
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:  # a late tick must not be silently bridged
                next_tick = time.monotonic()
            if not source.is_healthy():
                reason = source.failure_reason or "the sensor source stopped"
                store.set(
                    {
                        "tick": tick,
                        "ready": False,
                        "fatal": True,
                        "status": f"stopped — {reason}",
                    }
                )
                print(f"\nfatal: {reason}. Restart the overlay.", flush=True)
                # Hold the server up briefly so a polling dashboard shows the
                # reason rather than a bare connection error.
                time.sleep(2.0)
                return 1
            sample = source.sample()
            if sample is None:
                store.set({"tick": tick, "ready": False, "status": "waiting for /scan"})
                continue
            tick += 1
            # The window advances on scan arrival, not on wall clock. A late tick
            # is jitter and must not reset; a tick that sees no new scan is a real
            # gap and must, because bridging it would feed the policy a repeated
            # observation it never saw in training.
            fresh = sample["sequence"] != last_sequence
            last_sequence = sample["sequence"]
            tokens = tokenize_live_scan(
                sample["ranges"], sample["angle_increment"], context, overlay_config
            )
            if fresh:
                window.push(
                    timestamp_s=tick * TICK_PERIOD_S,
                    tokens=tokens,
                    goal=sample["goal"],
                    robot_state=sample["robot_state"],
                )
            else:
                window.reset()
                window.resets += 1
            store.set(
                build_frame(
                    tick=tick,
                    started=started,
                    sample=sample,
                    tokens=tokens,
                    window=window,
                    runner=runner,
                    context=context,
                    identity=identity,
                )
            )
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.shutdown()
        if hasattr(source, "shutdown"):
            source.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
