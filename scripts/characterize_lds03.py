#!/usr/bin/env python3
"""Characterize the physical LDS-03 observation process from recorded bags.

Offline and file-only. This script opens MCAP files directly through
``rosbag2_py``; it starts no node, joins no ROS graph, creates no publisher or
subscriber, and never contacts the robot.

It is thin by design. Every decision about whether an observation is usable
lives in :mod:`livifuser_nav.lds03_characterization`, where it is unit tested.
This file discovers inputs, decodes messages, hashes provenance, and serializes
the result.

Run on the ROS host after sourcing both setups::

    python3 scripts/characterize_lds03.py --output artifacts/lidar/lds03_characterization_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from livifuser_nav.lds03_characterization import (  # noqa: E402
    ScanRecord,
    confirm_intervals_by_zero_command,
    detect_range_quantization,
    find_motion_free_intervals,
    increment_convention_residual,
    interval_statistics,
    no_return_occupancy_by_sector,
    pooled_interval_statistics,
    robust_repeatability,
    sector_range_series,
    stable_bearing_eligibility,
    stochastic_missing_return,
)

SCHEMA_VERSION = "1.0.0"
SCAN_TOPIC = "/scan"
ODOM_TOPIC = "/odom"

#: Command evidence, preferred in this order. The stamped watchdog output carries
#: a header; raw ``/cmd_vel`` is an unstamped ``Twist``, so its only available
#: time is the bag receive time. Which one was used is recorded per bag.
STAMPED_COMMAND_TOPIC = "/livifuser/cmd_vel_stamped"
RAW_COMMAND_TOPIC = "/cmd_vel"

#: One-degree sectors. Beam spacing is at most 2*pi/397 (~0.907 deg) across every
#: observed beam count, so a one-degree sector always receives at least one beam
#: and angular binning cannot silently manufacture a missing return. Coverage is
#: nevertheless carried explicitly rather than assumed.
SECTOR_COUNT = 360

#: Documented gap thresholds. 0.15 s is the project's existing scan-gap gate;
#: the rest expose the tail.
GAP_THRESHOLDS_S = (0.15, 0.25, 0.5, 1.0)

#: Measured on this platform: reported linear velocity is *exactly* 0.0 while
#: stopped, but reported angular velocity carries noise up to about 1.4e-3 rad/s
#: (bags/stationary_pilot_2026-07-29_01) even with the wheels still. The angular
#: tolerance therefore sits above that measured noise floor and roughly two
#: orders of magnitude below the 0.40 rad/s command limit, so it cannot admit a
#: real turn. Achieved maxima are recorded per interval so the margin is
#: auditable. Command evidence is required on top of this.
STILL_LINEAR_TOLERANCE_MPS = 1e-3
STILL_ANGULAR_TOLERANCE_RADPS = 5e-3
MIN_STATIONARY_S = 5.0

#: Eligibility parameters for stationary repeatability. These are analysis
#: choices, not sensor specifications, and are echoed into the artifact.
ELIGIBILITY = {
    "min_valid_fraction": 0.98,
    "min_observations": 30,
    "min_range_m": 0.2,
    "max_range_m": 8.0,
    "max_neighbor_step_m": 0.10,
    "max_mad_m": 0.10,
    "max_half_split_drift_m": 0.02,
}

RANGE_BIN_EDGES_M = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
MIN_BIN_SAMPLES = 500

#: Recording trees that are physical robot data. Simulation output would live
#: under artifacts/simulation and is excluded by construction: this script only
#: ever looks inside `bags/`.
BAG_ROOT = Path("bags")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def git_revision(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                args, cwd=root, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    revision = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "revision": revision,
        "state": None if status is None else ("clean" if status == "" else "dirty"),
    }


def discover_scan_bags(root: Path) -> list[Path]:
    """Every recorded bag under ``root`` whose metadata declares ``/scan``."""

    found: list[Path] = []
    for metadata in sorted(root.rglob("metadata.yaml")):
        try:
            text = metadata.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if f"name: {SCAN_TOPIC}\n" in text or f"name: {SCAN_TOPIC}\r\n" in text:
            found.append(metadata.parent)
    return found


def bag_files(bag: Path) -> dict[str, Any]:
    """Hash every file that constitutes the recording."""

    entries = []
    for path in sorted(bag.iterdir()):
        if path.is_file():
            entries.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {"file_count": len(entries), "files": entries}


def read_bag(bag: Path) -> tuple[list[ScanRecord], dict[str, list], dict[str, Any]]:
    """Decode scan, odometry, and command streams. No ROS graph is involved."""

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}

    if STAMPED_COMMAND_TOPIC in types:
        command_topic, command_clock = STAMPED_COMMAND_TOPIC, "header_stamp"
    elif RAW_COMMAND_TOPIC in types:
        command_topic, command_clock = RAW_COMMAND_TOPIC, "bag_receive_time"
    else:
        command_topic, command_clock = None, "unavailable"

    wanted = [
        topic
        for topic in (SCAN_TOPIC, ODOM_TOPIC, command_topic)
        if topic is not None and topic in types
    ]
    reader.set_filter(rosbag2_py.StorageFilter(topics=wanted))

    scans: list[ScanRecord] = []
    odom: dict[str, list] = {"stamp_ns": [], "linear": [], "angular": []}
    command: dict[str, Any] = {
        "topic": command_topic,
        "clock": command_clock,
        "stamp_ns": [],
        "linear": [],
        "angular": [],
    }
    classes = {topic: get_message(types[topic]) for topic in wanted}

    while reader.has_next():
        topic, payload, receive_ns = reader.read_next()
        message = deserialize_message(payload, classes[topic])
        if topic == SCAN_TOPIC:
            stamp = message.header.stamp
            scans.append(
                ScanRecord(
                    stamp_ns=int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec),
                    angle_min=float(message.angle_min),
                    angle_max=float(message.angle_max),
                    angle_increment=float(message.angle_increment),
                    range_min=float(message.range_min),
                    range_max=float(message.range_max),
                    ranges=np.asarray(message.ranges, dtype=np.float64),
                )
            )
        elif topic == ODOM_TOPIC:
            stamp = message.header.stamp
            odom["stamp_ns"].append(int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec))
            odom["linear"].append(float(message.twist.twist.linear.x))
            odom["angular"].append(float(message.twist.twist.angular.z))
        elif topic == command_topic:
            twist = message.twist if command_clock == "header_stamp" else message
            if command_clock == "header_stamp":
                stamp = message.header.stamp
                command["stamp_ns"].append(
                    int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
                )
            else:
                command["stamp_ns"].append(int(receive_ns))
            command["linear"].append(float(twist.linear.x))
            command["angular"].append(float(twist.angular.z))
    return scans, odom, command


def geometry_summary(scans: list[ScanRecord]) -> dict[str, Any]:
    """Beam-count, angular-frame, and increment-convention evidence."""

    counts = Counter(scan.beam_count for scan in scans)
    angle_min = np.array([scan.angle_min for scan in scans], dtype=np.float64)
    angle_max = np.array([scan.angle_max for scan in scans], dtype=np.float64)
    increments = np.array([scan.angle_increment for scan in scans], dtype=np.float64)
    range_min = np.array([scan.range_min for scan in scans], dtype=np.float64)
    range_max = np.array([scan.range_max for scan in scans], dtype=np.float64)

    residuals = [
        increment_convention_residual(scan.beam_count, scan.angle_increment) for scan in scans
    ]
    relative = np.array([entry["relative_error"] for entry in residuals], dtype=np.float64)

    # Worst-case bearing disagreement if one scan's increment were reused for another.
    lowest, highest = min(counts), max(counts)
    far_index = lowest - 1
    reuse_error_rad = abs(
        far_index * (2.0 * math.pi / (lowest + 1)) - far_index * (2.0 * math.pi / (highest + 1))
    )

    return {
        "scan_count": len(scans),
        "beam_count_histogram": {str(k): int(v) for k, v in sorted(counts.items())},
        "beam_count_min": int(lowest),
        "beam_count_max": int(highest),
        "angle_min_rad": _spread(angle_min),
        "angle_max_rad": _spread(angle_max),
        "angle_increment_rad": _spread(increments),
        "declared_range_min_m": _spread(range_min),
        "declared_range_max_m": _spread(range_max),
        "increment_convention": {
            "relation": "angle_increment == 2*pi/(beam_count+1)",
            "max_relative_error": float(np.max(relative)),
            "median_relative_error": float(np.median(relative)),
            "scans_within_1e-6_relative": int(np.sum(relative <= 1e-6)),
            "verified_for_all_scans": bool(np.all(relative <= 1e-6)),
        },
        "global_bearing_table_error": {
            "worst_case_rad": float(reuse_error_rad),
            "worst_case_deg": float(math.degrees(reuse_error_rad)),
            "explanation": (
                "bearing disagreement at the far beam if the increment for "
                f"{lowest} beams were reused for {highest} beams"
            ),
        },
    }


def _spread(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "distinct_values": int(np.unique(values).size),
    }


def valid_range_support(scans: list[ScanRecord]) -> dict[str, Any]:
    """Distribution of the returns that are valid, for range-band context."""

    pooled = np.concatenate(
        [scan.ranges[np.isfinite(scan.ranges) & (scan.ranges > 0.0)] for scan in scans]
    )
    if pooled.size == 0:
        return {"sample_count": 0}
    return {
        "sample_count": int(pooled.size),
        "min_m": float(np.min(pooled)),
        "p01_m": float(np.quantile(pooled, 0.01)),
        "median_m": float(np.median(pooled)),
        "p99_m": float(np.quantile(pooled, 0.99)),
        "max_m": float(np.max(pooled)),
    }


def stationary_analysis(
    scans: list[ScanRecord],
    odom: dict[str, list],
    command: dict[str, Any],
) -> dict[str, Any]:
    """Repeatability and dropout over verified motion-free windows.

    Two independent lines of evidence must agree: odometry must report no motion
    and the recorded command stream must be exactly zero throughout. A bag whose
    name says "stationary" earns nothing here.
    """

    candidates = find_motion_free_intervals(
        odom["stamp_ns"],
        odom["linear"],
        odom["angular"],
        linear_tolerance_mps=STILL_LINEAR_TOLERANCE_MPS,
        angular_tolerance_radps=STILL_ANGULAR_TOLERANCE_RADPS,
        min_duration_s=MIN_STATIONARY_S,
    )
    intervals, command_rejected, evidence_basis = confirm_intervals_by_zero_command(
        candidates,
        command["stamp_ns"],
        command["linear"],
        command["angular"],
        command_topic_recorded=command["topic"] is not None,
    )
    reports: list[dict[str, Any]] = []
    for interval in intervals:
        window = [
            scan for scan in scans if interval.start_ns <= scan.stamp_ns < interval.end_ns
        ]
        if len(window) < ELIGIBILITY["min_observations"]:
            reports.append(
                {
                    "start_ns": interval.start_ns,
                    "duration_s": interval.duration_s,
                    "scan_count": len(window),
                    "analyzed": False,
                    "reason": "fewer scans than min_observations",
                }
            )
            continue

        values, valid, covered = sector_range_series(window, SECTOR_COUNT)
        eligibility = stable_bearing_eligibility(values, valid, covered, **ELIGIBILITY)
        eligible = eligibility["eligible"]
        repeat = robust_repeatability(
            values,
            valid,
            eligible,
            range_bin_edges_m=RANGE_BIN_EDGES_M,
            min_bin_samples=MIN_BIN_SAMPLES,
        )
        missing = stochastic_missing_return(valid, covered, eligible)
        reports.append(
            {
                "start_ns": interval.start_ns,
                "duration_s": interval.duration_s,
                "odometry_sample_count": interval.sample_count,
                "max_abs_linear_mps": interval.max_abs_linear_mps,
                "max_abs_angular_radps": interval.max_abs_angular_radps,
                "scan_count": len(window),
                "analyzed": True,
                "eligible_sector_count": int(np.sum(eligible)),
                "excluded_counts": eligibility["excluded_counts"],
                "repeatability": repeat,
                "stochastic_missing_return": missing,
            }
        )
    return {
        "command_evidence": {
            "topic": command["topic"],
            "clock": command["clock"],
            "sample_count": len(command["stamp_ns"]),
            "evidence_basis": evidence_basis,
        },
        "odometry_candidate_interval_count": len(candidates),
        "command_confirmed_interval_count": len(intervals),
        "command_rejected_intervals": command_rejected,
        "analyzed_interval_count": sum(1 for entry in reports if entry["analyzed"]),
        "intervals": reports,
    }


def pool_repeatability(bag_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-interval residual summaries into one supported statement.

    Residuals are taken about each sector's own median inside a single static
    scene, so pooling them across scenes is legitimate; pooling raw ranges would
    not be.
    """

    weighted: list[tuple[int, float, float]] = []
    total = 0
    for report in bag_reports:
        for interval in report["stationary"]["intervals"]:
            if not interval["analyzed"]:
                continue
            summary = interval["repeatability"]["overall"]
            if summary is None:
                continue
            count = int(summary["sample_count"])
            weighted.append((count, summary["robust_sigma_m"], summary["p95_abs_residual_m"]))
            total += count
    if not weighted:
        return {"sample_count": 0, "estimate": None, "note": "no analyzed stationary interval"}
    sigmas = np.array([entry[1] for entry in weighted], dtype=np.float64)
    p95s = np.array([entry[2] for entry in weighted], dtype=np.float64)
    counts = np.array([entry[0] for entry in weighted], dtype=np.float64)
    return {
        "sample_count": int(total),
        "contributing_interval_count": len(weighted),
        "estimate": {
            "sample_weighted_robust_sigma_m": float(np.sum(sigmas * counts) / np.sum(counts)),
            "interval_min_robust_sigma_m": float(np.min(sigmas)),
            "interval_max_robust_sigma_m": float(np.max(sigmas)),
            "sample_weighted_p95_abs_residual_m": float(np.sum(p95s * counts) / np.sum(counts)),
        },
        "interpretation": (
            "spread of repeated stationary observations about each sector's own median; "
            "this is repeatability, not absolute range accuracy"
        ),
    }


def build_proposed_parameters(
    geometry: dict[str, Any],
    pooled: dict[str, Any],
    bag_reports: list[dict[str, Any]],
    quantization: dict[str, Any],
) -> dict[str, Any]:
    """Simulation parameters implied by the measurements, kept clearly separate.

    Nothing here is a measurement. Each entry names the measured quantity it was
    derived from so an auditor can accept or reject the derivation independently
    of the data.
    """

    misses = []
    for report in bag_reports:
        for interval in report["stationary"]["intervals"]:
            if not interval["analyzed"]:
                continue
            estimate = interval["stochastic_missing_return"]["estimate"]
            if estimate is not None:
                misses.append(estimate["pooled_probability"])

    sigma = None
    if pooled["estimate"] is not None:
        sigma = pooled["estimate"]["sample_weighted_robust_sigma_m"]

    return {
        "status": "PROPOSED — derived from measurement, not itself measured",
        "beam_count_sampling": {
            "proposal": "sample per scan from the empirical beam-count histogram",
            "derived_from": "geometry.beam_count_histogram",
            "note": "must be resampled per scan; a fixed beam count is not faithful",
        },
        "angle_increment_rule": {
            "proposal": "angle_increment = 2*pi/(beam_count+1) for the sampled beam count",
            "derived_from": "geometry.increment_convention",
        },
        "range_quantization": {
            "proposal": (
                None
                if quantization.get("detected_step_m") is None
                else f"round simulated ranges onto a {quantization['detected_step_m']} m lattice"
            ),
            "derived_from": "measured.range_quantization",
            "caveats": [
                "the real driver reports on this lattice; omitting it makes simulated "
                "ranges strictly finer-grained than any real observation",
            ],
        },
        "range_noise": {
            "proposal": (
                None
                if sigma is None
                else (
                    f"additive zero-mean noise with sigma ~= {sigma:.4f} m, "
                    "applied before quantization"
                )
            ),
            "derived_from": "stationary_repeatability.estimate",
            "caveats": [
                "estimated at near-normal incidence on stable surfaces only",
                "range dependence is reported separately and may not be supported",
                "this is repeatability; absolute accuracy is unmeasured",
                (
                    "resolution-limited: residuals live on the reported range lattice, so "
                    "the median absolute deviation can only take lattice multiples and this "
                    "sigma cannot be resolved more finely than about one step"
                ),
                "the residual distribution is heavy-tailed; p95 far exceeds the robust sigma, "
                "so a pure Gaussian will understate rare large deviations",
            ],
        },
        "stochastic_missing_return": {
            "proposal": (
                None
                if not misses
                else f"per-beam miss probability ~= {float(np.median(misses)):.5f}"
            ),
            "derived_from": "stationary intervals, eligible sectors only",
            "caveats": [
                "measured only where a stable in-range surface was present",
                "does not describe open space, which produces no return legitimately",
                "angular structure is not modelled by this scalar",
            ],
        },
        "not_derived": [
            "absolute range accuracy (no ground-truth distance exists in this data)",
            "incidence-angle dependence (not separable from scene geometry here)",
            "reflectivity or material dependence (surface identity was not recorded)",
            "motion distortion within a scan (stationary data cannot show it)",
            "temperature or warm-up drift (not instrumented)",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bag-root", type=Path, default=BAG_ROOT, help="directory tree of recorded bags"
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON artifact to write")
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing output artifact"
    )
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        parser.error(f"{args.output} exists; refusing to overwrite derived evidence")

    started = time.perf_counter()
    bags = discover_scan_bags(args.bag_root)
    if not bags:
        parser.error(f"no bag under {args.bag_root} declares {SCAN_TOPIC}")

    bag_reports: list[dict[str, Any]] = []
    all_scans: list[ScanRecord] = []
    stamp_groups: list[list[int]] = []
    for bag in bags:
        print(f"reading {bag} ...", flush=True)
        scans, odom, command = read_bag(bag)
        if not scans:
            print(f"  no {SCAN_TOPIC} messages decoded; skipped", flush=True)
            continue
        all_scans.extend(scans)
        stamp_groups.append([scan.stamp_ns for scan in scans])
        report = {
            "bag": str(bag).replace("\\", "/"),
            "provenance": bag_files(bag),
            "geometry": geometry_summary(scans),
            "timing": interval_statistics(
                [scan.stamp_ns for scan in scans], GAP_THRESHOLDS_S
            ),
            "valid_range_support": valid_range_support(scans),
            "odometry_sample_count": len(odom["stamp_ns"]),
            "stationary": stationary_analysis(scans, odom, command),
        }
        bag_reports.append(report)
        print(
            f"  {len(scans)} scans, "
            f"{report['stationary']['analyzed_interval_count']} analyzed still interval(s)",
            flush=True,
        )

    pooled = pool_repeatability(bag_reports)
    global_geometry = geometry_summary(all_scans)
    quantization = detect_range_quantization(all_scans)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "disposition": (
            "Offline characterization of the physical LDS-03 observation process from "
            "preserved bags. No robot, ROS graph, or network access was involved."
        ),
        "code_provenance": {
            "git": git_revision(REPOSITORY_ROOT),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "analysis_parameters": {
            "sector_count": SECTOR_COUNT,
            "sector_width_deg": 360.0 / SECTOR_COUNT,
            "gap_thresholds_s": list(GAP_THRESHOLDS_S),
            "still_linear_tolerance_mps": STILL_LINEAR_TOLERANCE_MPS,
            "still_angular_tolerance_radps": STILL_ANGULAR_TOLERANCE_RADPS,
            "min_stationary_s": MIN_STATIONARY_S,
            "eligibility": dict(ELIGIBILITY),
            "range_bin_edges_m": list(RANGE_BIN_EDGES_M),
            "min_bin_samples": MIN_BIN_SAMPLES,
            "note": "analysis choices, not sensor specifications",
        },
        "inputs": {
            "bag_root": str(args.bag_root).replace("\\", "/"),
            "bag_count": len(bag_reports),
            "simulation_excluded": True,
            "simulation_exclusion_basis": (
                "only bags/ is searched; no simulated recording exists in this tree"
            ),
        },
        "measured": {
            "global_geometry": global_geometry,
            "range_quantization": quantization,
            "pooled_timing": pooled_interval_statistics(stamp_groups, GAP_THRESHOLDS_S),
            "global_no_return_occupancy": no_return_occupancy_by_sector(
                all_scans, SECTOR_COUNT
            ),
            "stationary_repeatability": pooled,
            "per_bag": bag_reports,
        },
        "proposed_simulation_parameters": build_proposed_parameters(
            global_geometry, pooled, bag_reports, quantization
        ),
        "limitations": [
            "No ground-truth distance exists in this data, so no accuracy claim is made; "
            "every range figure is repeatability about a sector's own median.",
            "Repeatability is estimated only on stable, near-normal-incidence surfaces "
            "inside the eligibility band, so it does not describe oblique or distant returns.",
            "No-return occupancy mixes open space, out-of-reach surfaces, and genuine "
            "sensor misses; only the eligible-sector estimate isolates stochastic misses.",
            "The two no-return codes (NaN and exactly 0.0) are counted separately but their "
            "physical distinction is unverified.",
            "Beam-count and increment ranges are observed bounds on the bags measured, not "
            "proven limits of the device.",
            "Reported ranges lie on a discrete lattice, so every robust spread statistic "
            "derived from them is lattice-valued; identical sigma across independent "
            "recordings reflects that resolution limit, not unusual agreement.",
            "Stationary data cannot reveal within-scan motion distortion.",
            "Bags were recorded across several sessions and scenes; scene composition, not "
            "the sensor, dominates no-return occupancy.",
        ],
    }

    # Runtime is deliberately not serialized: the artifact must be byte-identical
    # across reruns on the same inputs so determinism can be checked directly.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} in {time.perf_counter() - started:.1f} s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
