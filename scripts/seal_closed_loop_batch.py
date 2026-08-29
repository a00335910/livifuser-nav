"""Seal the completed closed-loop confirmatory batch.

Sealing fixes what the analysis is entitled to read. It records, for every
attempt, the identity it belongs to, its classification, and the SHA-256 of the
records that decide that classification -- so a later reader can prove the
analysed set is the collected set, and that nothing was added, dropped or edited
between collection and reporting.

Deliberately records no rate and no per-arm outcome: sealing happens before the
analysis, and section 9 forbids a scientific aggregate until the set is sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from livifuser_nav.confirmatory_plan import (  # noqa: E402
    SCHEDULE_SHA256,
    build_plan,
    classify_attempt,
    locate_schedule,
)

SCHEMA_VERSION = "1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="evidence root")
    parser.add_argument("--output", required=True, help="seal file to write")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    if output.exists():
        print(f"refusing to overwrite an existing seal: {output}", file=sys.stderr)
        return 2

    plan = build_plan(locate_schedule(ROOT))
    by_key = {identity.key: identity for identity in plan}

    attempts = []
    counts = {"scientific": 0, "operational": 0, "absent": 0}
    for attempt_dir in sorted(root.rglob("attempt_*")):
        if not attempt_dir.is_dir():
            continue
        ordinal = int(attempt_dir.parent.name)
        seed = int(attempt_dir.parent.parent.name)
        arm = attempt_dir.parent.parent.parent.name
        key = f"{arm}/{seed}/{ordinal}"
        kind = classify_attempt(attempt_dir)
        counts[kind] = counts.get(kind, 0) + 1
        entry = {
            "identity": key,
            "attempt": attempt_dir.name,
            "classification": kind,
            "in_frozen_plan": key in by_key,
        }
        terminal = attempt_dir / "terminal.json"
        if terminal.is_file():
            entry["terminal_sha256"] = sha256_file(terminal)
            record = json.loads(terminal.read_text(encoding="utf-8"))
            entry["terminal_reason"] = record.get("terminal_reason")
            entry["context_sequence"] = record.get("context_sequence")
        clearance = attempt_dir.parent / "audit_cleared.json"
        if clearance.is_file():
            entry["audit_cleared_sha256"] = sha256_file(clearance)
        attempts.append(entry)

    outside = [a["identity"] for a in attempts if not a["in_frozen_plan"]]
    seal = {
        "schema_version": SCHEMA_VERSION,
        "sealed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schedule_sha256": SCHEDULE_SHA256,
        "plan_identities": len(plan),
        "attempts_total": len(attempts),
        "classification_counts": counts,
        "identities_with_accepted_outcome": len(
            {a["identity"] for a in attempts if a["classification"] == "scientific"}
        ),
        "attempts_outside_frozen_plan": outside,
        "attempts": attempts,
        "note": (
            "records classification and hashes only; no rate, contrast or per-arm "
            "outcome is computed here, because the analysis runs after this seal"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(seal, indent=2, sort_keys=True), encoding="utf-8", newline="\n")

    print(f"  sealed {len(attempts)} attempts -> {output}")
    print(f"  classifications      : {counts}")
    print(f"  identities accepted  : {seal['identities_with_accepted_outcome']}")
    print(f"  outside frozen plan  : {len(outside)}")
    print(f"  seal sha256          : {sha256_file(output)}")
    return 1 if outside else 0


if __name__ == "__main__":
    raise SystemExit(main())
