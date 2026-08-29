#!/usr/bin/env python3
"""Fail closed when the persistent evidence volume lacks frozen headroom."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

MINIMUM = {
    "development": 20 * 1024**3,
    "confirmatory": 800 * 1024**3,
    "rollout": 2 * 1024**3,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MINIMUM), required=True)
    parser.add_argument("--path", type=Path, default=Path("/workspace"))
    args = parser.parse_args()
    root = args.path.resolve()
    if not root.is_dir():
        raise ValueError(f"storage root is not a directory: {root}")
    usage = shutil.disk_usage(root)
    # YES means an operator confirmed encryption in the provider console.
    # UNCONFIRMED means the provider exposes no such control for this volume
    # type, so the property could not be verified. Both are recorded verbatim;
    # neither is inferred, and anything else fails closed.
    acknowledgement = os.environ.get("LIVIFUSER_ENCRYPTED_VOLUME_ACK")
    encryption_status = {
        "YES": "confirmed_by_operator",
        "UNCONFIRMED": "not_independently_confirmed",
    }.get(acknowledgement)
    if encryption_status is None:
        raise PermissionError(
            "set LIVIFUSER_ENCRYPTED_VOLUME_ACK to YES (confirmed in the provider "
            "console) or UNCONFIRMED (no such control is exposed for this volume)"
        )
    required = MINIMUM[args.mode]
    if usage.free < required:
        raise OSError(f"insufficient free storage: {usage.free} < {required}")
    report = {
        "status": "pass",
        "mode": args.mode,
        "path": str(root),
        "free_bytes": usage.free,
        "required_free_bytes": required,
        "volume_encryption_status": encryption_status,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
