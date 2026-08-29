#!/usr/bin/env python3
"""Capture non-secret RunPod host/runtime provenance to a new JSON file."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return f"unavailable:{type(exc).__name__}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite provenance: {output}")
    import numpy
    import PIL
    import scipy
    import torch
    import transformers

    usage = shutil.disk_usage("/workspace")
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": properties.total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "uuid_from_nvidia_smi": command_output(
                        [
                            "nvidia-smi",
                            f"--id={index}",
                            "--query-gpu=uuid",
                            "--format=csv,noheader",
                        ]
                    ),
                }
            )
    report = {
        "schema_version": "1.0.0",
        "status": "captured_before_runtime_benchmark",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "RunPod",
        "operator_encrypted_volume_ack": os.environ.get(
            "LIVIFUSER_ENCRYPTED_VOLUME_ACK"
        )
        == "YES",
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "memory": command_output(["bash", "-lc", "free -b"]),
            "kernel": platform.release(),
            "runpod_pod_id": os.environ.get("RUNPOD_POD_ID", "unavailable"),
        },
        "gpu": {
            "driver": command_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ),
            "nvidia_smi": command_output(["nvidia-smi", "-q"]),
            "devices": cuda_devices,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "transformers": transformers.__version__,
            "numpy": numpy.__version__,
            "pillow": PIL.__version__,
            "scipy": scipy.__version__,
            "ros_distro": os.environ.get("ROS_DISTRO", "unavailable"),
        },
        "workspace_storage": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "captured", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
