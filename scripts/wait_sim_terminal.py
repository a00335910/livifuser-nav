#!/usr/bin/env python3
"""Wait for one terminal supervisor status and preserve it as JSON.

Shutdown is driven from the main loop rather than from the subscription
callback. Calling ``rclpy.shutdown()`` inside a callback while ``rclpy.spin()``
is running invalidates the executor's wait set from under it, and the process
hangs instead of exiting. Measured on the pod: the terminal record was written
70 s into an episode and the process then sat until the 900 s wall timeout
killed it, so every episode cost roughly fourteen extra minutes of teardown --
about 250 hours across a 1,080-rollout confirmatory batch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rclpy
from livifuser_interfaces.msg import SimulationSupervisorStatus
from rclpy.node import Node


class TerminalWaiter(Node):
    def __init__(self, output: Path) -> None:
        super().__init__("livifuser_terminal_waiter")
        self.output = output
        self.finished = False
        self.wrote_record = False
        self.create_subscription(
            SimulationSupervisorStatus,
            "/livifuser/eval/supervisor_status",
            self._on_status,
            10,
        )

    def _on_status(self, message: SimulationSupervisorStatus) -> None:
        if self.finished or not message.terminal:
            return
        if self.output.exists():
            # Never overwrite a recorded outcome; stop and let the caller
            # classify the attempt.
            self.get_logger().error(f"refusing to overwrite {self.output}")
            self.finished = True
            return
        payload = {
            # 1.1.0 adds stretched_interval_count; additive, older readers
            # ignore it.
            "schema_version": "1.1.0",
            "terminal": True,
            "terminal_reason": message.terminal_reason,
            "priority_reason": message.priority_reason,
            "variant": message.variant,
            "seed": int(message.seed),
            "context_sequence": int(message.context_sequence),
            "stretched_interval_count": int(message.stretched_interval_count),
            "collision": bool(message.collision),
            "uncertainty_intervention": bool(message.uncertainty_intervention),
            "success": bool(message.success),
            "ground_truth_clearance_m": float(message.ground_truth_clearance_m),
            "ground_truth_goal_distance_m": float(message.ground_truth_goal_distance_m),
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        self.wrote_record = True
        self.finished = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rclpy.init()
    node = TerminalWaiter(args.output.resolve())
    try:
        # Poll rather than spin, so the loop -- not a callback -- decides when
        # to stop and the executor is never torn down from inside itself.
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if args.output.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
