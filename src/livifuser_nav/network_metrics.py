"""Latency and delivery statistics for a network-transported ROS session.

This module holds every decision about what a network measurement means; the
collecting script only subscribes and hands samples over. That split matters
here for the same reason it does elsewhere in this project: a boundary bug in a
statistic is invisible, and statistics computed inside a callback are untestable.

Three quantities are kept distinct throughout, because conflating them produces a
flattering number rather than a useful one:

*Link latency* is what a ping measures. It is the floor, not the experience.

*Delivery latency* is arrival time minus the message's own header stamp: how
stale a sensor reading is by the time a consumer sees it. This is the figure that
decides whether closed-loop control over the link is viable, and it includes
serialisation, transport, and any queuing the middleware does.

*Delivery rate* is how much of the stream survived. A link can show low latency
on the messages that arrive while silently dropping half of them, and a latency
summary alone would call that healthy.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

NS_PER_MS = 1_000_000.0
NS_PER_S = 1_000_000_000.0


class NetworkMetricError(ValueError):
    """Raised when a measurement request is structurally impossible."""


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile over an already-sorted sequence."""

    if not sorted_values:
        return math.nan
    if not 0.0 <= fraction <= 1.0:
        raise NetworkMetricError(f"fraction must be in [0, 1], got {fraction}")
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize_ms(values_ms: Sequence[float]) -> dict[str, float | int] | None:
    """Distribution summary in milliseconds, tail included.

    The tail is reported explicitly because a median hides exactly the stalls
    that break a control loop. A p99 and a maximum are what decide whether a
    link is usable, not a mean.
    """

    finite = sorted(float(v) for v in values_ms if math.isfinite(v))
    if not finite:
        return None
    return {
        "count": len(finite),
        "min_ms": finite[0],
        "median_ms": percentile(finite, 0.5),
        "mean_ms": sum(finite) / len(finite),
        "p95_ms": percentile(finite, 0.95),
        "p99_ms": percentile(finite, 0.99),
        "max_ms": finite[-1],
    }


def jitter_summary(values_ms: Sequence[float]) -> dict[str, float | int] | None:
    """Variation between consecutive round trips, in milliseconds.

    Reported separately from latency because for teleoperation it is often the
    more important number. A link with a steady 200 ms delay is drivable — the
    operator adapts. A link averaging 40 ms that swings between 10 and 300 is
    not, because nothing about the response is predictable.

    Measured on samples in send order, so it describes how the link treated a
    stream of commands rather than how an arbitrary pair of packets compared.

    ``mean_abs_delta_ms`` is the RFC 3550 style quantity; the percentiles say
    how bad the occasional swing gets, which the mean hides.
    """

    finite = [float(v) for v in values_ms if math.isfinite(v)]
    if len(finite) < 2:
        return None
    deltas = sorted(abs(b - a) for a, b in zip(finite, finite[1:], strict=False))
    return {
        "count": len(deltas),
        "median_ms": percentile(deltas, 0.5),
        "mean_abs_delta_ms": sum(deltas) / len(deltas),
        "p95_ms": percentile(deltas, 0.95),
        "p99_ms": percentile(deltas, 0.99),
        "max_ms": deltas[-1],
    }


def reordering_summary(
    sequences_in_arrival_order: Sequence[int],
) -> dict[str, float | int]:
    """How often replies arrived out of the order they were sent.

    Worth measuring rather than assuming. Cellular links are widely expected to
    reorder, and on the runs measured here they did not reorder at all — which
    is only a usable statement because it was checked. Reordering matters for a
    command stream because a stale command applied after a newer one moves the
    robot the wrong way.

    ``max_displacement`` is how far out of position the worst single packet was.
    It does *not* distinguish one late straggler from a fully shuffled stream —
    both give the same figure when the worst packet moved the same distance.
    ``mean_displacement`` is what separates them, because a lone straggler
    leaves every other packet in place while a shuffle moves all of them.
    """

    order = [int(s) for s in sequences_in_arrival_order]
    if len(order) < 2:
        return {
            "sample_count": len(order),
            "out_of_order": 0,
            "out_of_order_fraction": 0.0,
            "max_displacement": 0,
            "mean_displacement": 0.0,
        }

    out_of_order = sum(1 for a, b in zip(order, order[1:], strict=False) if b < a)
    ranks = {seq: index for index, seq in enumerate(sorted(order))}
    displacements = [abs(ranks[seq] - index) for index, seq in enumerate(order)]
    return {
        "sample_count": len(order),
        "out_of_order": out_of_order,
        "out_of_order_fraction": out_of_order / (len(order) - 1),
        "max_displacement": max(displacements),
        "mean_displacement": sum(displacements) / len(displacements),
    }


@dataclass(frozen=True, slots=True)
class Spike:
    """A contiguous run of samples above a threshold."""

    start_s: float
    end_s: float
    peak_ms: float
    sample_count: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def find_spikes(
    times_s: Sequence[float],
    values_ms: Sequence[float],
    threshold_ms: float,
) -> list[Spike]:
    """Locate excursions above ``threshold_ms`` in time, not just count them.

    Knowing that 2% of samples exceeded a threshold is far less useful than
    knowing they arrived as three separate two-second stalls, because the first
    is tolerable jitter and the second is three lost control windows.
    """

    if len(times_s) != len(values_ms):
        raise NetworkMetricError("times and values must have equal length")

    spikes: list[Spike] = []
    start_index: int | None = None
    for index in range(len(values_ms) + 1):
        above = index < len(values_ms) and math.isfinite(values_ms[index]) and (
            values_ms[index] > threshold_ms
        )
        if above and start_index is None:
            start_index = index
        elif not above and start_index is not None:
            stop = index - 1
            window = values_ms[start_index : stop + 1]
            spikes.append(
                Spike(
                    start_s=float(times_s[start_index]),
                    end_s=float(times_s[stop]),
                    peak_ms=float(max(window)),
                    sample_count=stop - start_index + 1,
                )
            )
            start_index = None
    return spikes


def delivery_summary(
    arrival_ns: Sequence[int],
    header_ns: Sequence[int],
    *,
    expected_hz: float | None,
    gap_thresholds_ms: Sequence[float],
    spike_threshold_ms: float,
) -> dict[str, object]:
    """Full picture for one topic: staleness, cadence, losses and spikes.

    ``expected_hz`` enables a delivered-fraction estimate. Without it the summary
    still reports achieved rate, but cannot say what proportion of the stream was
    lost, and reports that honestly rather than assuming none was.
    """

    if len(arrival_ns) != len(header_ns):
        raise NetworkMetricError("arrival and header sequences must have equal length")
    if len(arrival_ns) < 2:
        return {
            "sample_count": len(arrival_ns),
            "insufficient_samples": True,
            "delivery_latency": None,
            "inter_arrival": None,
            "achieved_hz": None,
            "delivered_fraction": None,
            "gaps_above_threshold_ms": {f"{t:g}": 0 for t in gap_thresholds_ms},
            "spikes": [],
        }

    order = sorted(range(len(arrival_ns)), key=lambda i: arrival_ns[i])
    arrivals = [int(arrival_ns[i]) for i in order]
    headers = [int(header_ns[i]) for i in order]

    latency_ms = [(a - h) / NS_PER_MS for a, h in zip(arrivals, headers, strict=True)]
    intervals_ms = [
        (later - earlier) / NS_PER_MS
        for earlier, later in zip(arrivals, arrivals[1:], strict=False)
    ]

    span_s = (arrivals[-1] - arrivals[0]) / NS_PER_S
    achieved_hz = (len(arrivals) - 1) / span_s if span_s > 0 else None

    delivered_fraction = None
    if expected_hz and span_s > 0:
        expected_count = expected_hz * span_s
        if expected_count > 0:
            delivered_fraction = min(1.0, (len(arrivals) - 1) / expected_count)

    base_s = arrivals[0] / NS_PER_S
    times_s = [(a / NS_PER_S) - base_s for a in arrivals]
    spikes = find_spikes(times_s, latency_ms, spike_threshold_ms)

    return {
        "sample_count": len(arrivals),
        "duration_s": span_s,
        "delivery_latency": summarize_ms(latency_ms),
        "inter_arrival": summarize_ms(intervals_ms),
        "achieved_hz": achieved_hz,
        "expected_hz": expected_hz,
        "delivered_fraction": delivered_fraction,
        "gaps_above_threshold_ms": {
            f"{threshold:g}": sum(1 for v in intervals_ms if v > threshold)
            for threshold in gap_thresholds_ms
        },
        "spike_threshold_ms": spike_threshold_ms,
        "spike_count": len(spikes),
        "spike_total_duration_s": sum(s.duration_s for s in spikes),
        "spikes": [
            {
                "start_s": round(s.start_s, 3),
                "end_s": round(s.end_s, 3),
                "duration_s": round(s.duration_s, 3),
                "peak_ms": round(s.peak_ms, 3),
                "sample_count": s.sample_count,
            }
            for s in spikes[:200]
        ],
        "spikes_truncated": len(spikes) > 200,
    }


#: Matches the RTT field of a ping reply on Linux, macOS and Windows.
_PING_RTT = re.compile(r"time[=<]\s*([0-9.]+)\s*ms", re.IGNORECASE)


def parse_ping_rtt_ms(line: str) -> float | None:
    """Extract a round-trip time from one line of ping output.

    Returns ``None`` for any line that is not a reply, including timeouts and
    summary lines, so a lost packet is never silently recorded as a fast one.
    """

    if "unreachable" in line.lower() or "timed out" in line.lower():
        return None
    match = _PING_RTT.search(line)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


@dataclass
class PingSession:
    """Accumulates ping replies and losses over a measurement window."""

    target: str
    rtt_ms: list[float] = field(default_factory=list)
    sent: int = 0
    lost: int = 0

    def record_line(self, line: str) -> None:
        rtt = parse_ping_rtt_ms(line)
        if rtt is not None:
            self.rtt_ms.append(rtt)
            self.sent += 1
        elif "unreachable" in line.lower() or "timed out" in line.lower():
            self.sent += 1
            self.lost += 1

    def summary(self, *, spike_threshold_ms: float) -> dict[str, object]:
        above = sum(1 for v in self.rtt_ms if v > spike_threshold_ms)
        return {
            "target": self.target,
            "replies": len(self.rtt_ms),
            "sent": self.sent,
            "lost": self.lost,
            "loss_fraction": (self.lost / self.sent) if self.sent else None,
            "rtt": summarize_ms(self.rtt_ms),
            "spike_threshold_ms": spike_threshold_ms,
            "replies_above_threshold": above,
        }


def command_round_trip(
    intent_stamp_ns: Sequence[int],
    final_stamp_ns: Sequence[int],
    *,
    match_window_ms: float = 500.0,
) -> dict[str, object]:
    """Latency from operator intent to the corresponding gated final command.

    Each final command is matched to the most recent intent at or before it, so a
    future intent can never be credited with having caused an earlier command.
    Unmatched commands are counted rather than dropped, because a large unmatched
    count is itself the finding.
    """

    intents = sorted(int(v) for v in intent_stamp_ns)
    finals = sorted(int(v) for v in final_stamp_ns)
    if not intents or not finals:
        return {"matched": 0, "unmatched": len(finals), "round_trip": None}

    window_ns = match_window_ms * NS_PER_MS
    deltas_ms: list[float] = []
    unmatched = 0
    cursor = 0
    for final in finals:
        while cursor + 1 < len(intents) and intents[cursor + 1] <= final:
            cursor += 1
        candidate = intents[cursor]
        if candidate <= final and (final - candidate) <= window_ns:
            deltas_ms.append((final - candidate) / NS_PER_MS)
        else:
            unmatched += 1

    return {
        "matched": len(deltas_ms),
        "unmatched": unmatched,
        "match_window_ms": match_window_ms,
        "round_trip": summarize_ms(deltas_ms),
    }


#: Control-quality bands for remote operation, in milliseconds. These are
#: conventional teleoperation thresholds rather than measurements, and are
#: labelled as such wherever they are reported.
CONTROL_BANDS_MS: tuple[tuple[str, float, float], ...] = (
    ("responsive", 0.0, 100.0),
    ("noticeable_lag", 100.0, 300.0),
    ("degraded", 300.0, 1000.0),
    ("unusable", 1000.0, math.inf),
)


def control_quality_bands(
    values_ms: Sequence[float],
    bands: Sequence[tuple[str, float, float]] = CONTROL_BANDS_MS,
) -> dict[str, object]:
    """Fraction of samples in each remote-operation quality band.

    A median tells you nothing about whether a link is drivable. What matters is
    how much of the session sat in each regime: an average of 90 ms made up of
    half at 20 ms and half at 160 ms is a very different experience from a steady
    90 ms, and only the banded view distinguishes them.

    The band edges are conventional teleoperation rules of thumb, not something
    measured here, and are reported alongside the result so they can be argued
    with.
    """

    finite = [float(v) for v in values_ms if math.isfinite(v)]
    if not finite:
        return {"sample_count": 0, "bands": None}

    counts = {}
    for name, low, high in bands:
        counts[name] = sum(1 for v in finite if low <= v < high)
    total = len(finite)
    return {
        "sample_count": total,
        # An open-ended upper edge is emitted as null: JSON has no infinity, and
        # serialising it as a large number would invent a bound that is not there.
        "band_edges_ms": [
            [name, low, (high if math.isfinite(high) else None)] for name, low, high in bands
        ],
        "edges_are": "conventional teleoperation thresholds, not measured here",
        "counts": counts,
        "fractions": {name: counts[name] / total for name in counts},
    }


def outage_windows(
    times_s: Sequence[float],
    *,
    expected_interval_s: float | None = None,
    missing_multiple: float = 3.0,
) -> dict[str, object]:
    """Periods where samples stopped arriving altogether.

    Distinct from high latency, and more serious. A slow link still delivers; a
    link that stops delivering leaves a remotely driven robot executing its last
    command with nobody watching, which is why the duration of the longest
    outage matters more than the average.

    ``expected_interval_s`` defaults to the *observed* median interval rather
    than an assumed one. Assuming it caused a real failure: a probe believed to
    run at 5 Hz actually ran at 1 Hz on one platform, so every ordinary gap
    exceeded the threshold and a clean run was reported as 595 outages covering
    its entire duration. Deriving the baseline from the data cannot drift away
    from the probe that produced it.
    """

    ordered = sorted(float(t) for t in times_s)
    if len(ordered) < 2:
        return {"sample_count": len(ordered), "outages": [], "longest_s": None}

    intervals = sorted(b - a for a, b in zip(ordered, ordered[1:], strict=False))
    observed_median = intervals[len(intervals) // 2]
    if expected_interval_s is None:
        expected_interval_s = observed_median

    limit = expected_interval_s * missing_multiple
    outages = []
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        gap = later - earlier
        if gap > limit:
            outages.append(
                {"start_s": round(earlier, 3), "duration_s": round(gap, 3)}
            )
    return {
        "sample_count": len(ordered),
        "expected_interval_s": expected_interval_s,
        "observed_median_interval_s": observed_median,
        "observed_rate_hz": (1.0 / observed_median) if observed_median > 0 else None,
        "outage_threshold_s": limit,
        "outage_count": len(outages),
        "longest_s": max((o["duration_s"] for o in outages), default=None),
        "total_outage_s": sum(o["duration_s"] for o in outages),
        "outages": outages[:200],
    }


def correlate(
    a_times_s: Sequence[float],
    a_values: Sequence[float],
    b_times_s: Sequence[float],
    b_values: Sequence[float],
    *,
    window_s: float = 1.0,
) -> dict[str, object]:
    """Pair two unevenly sampled series by nearest timestamp and correlate them.

    Intended for asking whether latency spikes coincide with signal drops. The
    two probes run at different rates and are not synchronised, so pairing is by
    nearest timestamp within a window rather than by index; unpaired samples are
    counted rather than dropped silently.
    """

    if len(a_times_s) != len(a_values) or len(b_times_s) != len(b_values):
        raise NetworkMetricError("each series needs matching times and values")

    b_sorted = sorted(zip(b_times_s, b_values, strict=True))
    b_t = [t for t, _ in b_sorted]
    pairs: list[tuple[float, float]] = []
    unpaired = 0

    for t, value in zip(a_times_s, a_values, strict=True):
        if not b_t:
            break
        lo, hi = 0, len(b_t) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if b_t[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        best = min(
            (c for c in {max(lo - 1, 0), lo, min(lo + 1, len(b_t) - 1)}),
            key=lambda i: abs(b_t[i] - t),
        )
        if abs(b_t[best] - t) <= window_s:
            pairs.append((float(value), float(b_sorted[best][1])))
        else:
            unpaired += 1

    if len(pairs) < 3:
        return {"paired": len(pairs), "unpaired": unpaired, "pearson_r": None}

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(pairs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    r = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None

    return {
        "paired": n,
        "unpaired": unpaired,
        "window_s": window_s,
        "pearson_r": r,
        "note": "association only; this does not establish that one caused the other",
    }


def attribute_loss(
    sent_seq: Sequence[int],
    robot_received_seq: Sequence[int],
    operator_replied_seq: Sequence[int],
) -> dict[str, object]:
    """Separate uplink loss from downlink loss using both sides' records.

    From the operator alone, a probe that never returns is simply "lost", and
    that is the wrong granularity. If the robot logged the probe but no reply
    came back, the command *arrived and was acted on* while the operator saw
    nothing. If the robot never logged it, the command never arrived at all.
    Those demand different responses — the first needs better feedback, the
    second needs the robot to stop on its own — so they are counted separately.

    Sequence numbers not present in ``sent_seq`` are reported rather than
    ignored, since they indicate the two logs are from different runs.
    """

    sent = set(int(s) for s in sent_seq)
    received = set(int(s) for s in robot_received_seq)
    replied = set(int(s) for s in operator_replied_seq)

    unexpected = (received | replied) - sent
    received &= sent
    replied &= sent

    uplink_lost = sent - received
    downlink_lost = received - replied
    complete = replied

    total = len(sent)
    return {
        "sent": total,
        "arrived_at_robot": len(received),
        "returned_to_operator": len(complete),
        "uplink_lost": len(uplink_lost),
        "downlink_lost": len(downlink_lost),
        "uplink_loss_fraction": (len(uplink_lost) / total) if total else None,
        "downlink_loss_fraction": (
            (len(downlink_lost) / len(received)) if received else None
        ),
        "unexpected_sequences": len(unexpected),
        "interpretation": {
            "uplink_lost": "command never reached the robot; it did not act",
            "downlink_lost": "robot received and acted, but the operator saw no reply",
        },
    }


def clock_offset_estimate(
    t_send: float, t_robot_recv: float, t_robot_send: float, t_final: float
) -> dict[str, float | None]:
    """Round-trip time, and one-way legs given an estimated clock offset.

    Round-trip is exact: both ends of it are read from the operator's own clock,
    so no synchronisation is required and it is always trustworthy.

    The one-way split is not. It needs the two machines to agree on time, and
    the offset here is estimated the way NTP does it, by assuming the path is
    symmetric. On a cellular link that assumption is often wrong — uplink and
    downlink are rarely equal — so the split is reported as an estimate and the
    round-trip should be preferred whenever it can answer the question.
    """

    round_trip = (t_final - t_send) - (t_robot_send - t_robot_recv)
    offset = ((t_robot_recv - t_send) + (t_robot_send - t_final)) / 2.0
    return {
        "round_trip_ms": round_trip * 1000.0,
        "robot_processing_ms": (t_robot_send - t_robot_recv) * 1000.0,
        "estimated_clock_offset_ms": offset * 1000.0,
        "uplink_ms_estimate": (t_robot_recv - t_send - offset) * 1000.0,
        "downlink_ms_estimate": (t_final - t_robot_send + offset) * 1000.0,
        "one_way_caveat": "assumes a symmetric path; round_trip_ms needs no such assumption",
    }


def split_by_state(
    sample_times_s: Sequence[float],
    sample_values: Sequence[float],
    state_times_s: Sequence[float],
    state_labels: Sequence[str],
) -> dict[str, object]:
    """Group samples by whichever state was in force when each was taken.

    State is treated as a step function held from each observation until the
    next, which is the correct reading for something polled periodically: the
    connection was in that mode from when it was seen until it was seen to
    change, not only at the instant of the poll.

    This exists because a session can silently be a mixture. A tunnelled link
    that falls back to a relay partway through produces one latency
    distribution that is really two, and an unsplit summary describes neither.
    """

    if len(sample_times_s) != len(sample_values):
        raise NetworkMetricError("sample times and values must have equal length")
    if len(state_times_s) != len(state_labels):
        raise NetworkMetricError("state times and labels must have equal length")

    states = sorted(zip(state_times_s, state_labels, strict=True))
    if not states:
        return {"states": {}, "unassigned": len(sample_values)}

    grouped: dict[str, list[float]] = {}
    unassigned = 0
    for t, value in zip(sample_times_s, sample_values, strict=True):
        if t < states[0][0]:
            unassigned += 1
            continue
        lo, hi = 0, len(states) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if states[mid][0] <= t:
                lo = mid
            else:
                hi = mid - 1
        grouped.setdefault(states[lo][1], []).append(float(value))

    return {
        "states": {
            label: {
                "sample_count": len(values),
                "summary": summarize_ms(values),
                "control_bands": control_quality_bands(values),
            }
            for label, values in sorted(grouped.items())
        },
        "unassigned": unassigned,
        "note": "state held from each observation until the next",
    }


def network_cost(
    robot_local: dict[str, object],
    host_observed: dict[str, object],
    *,
    keys: Sequence[str] = ("median_ms", "p95_ms", "p99_ms", "max_ms"),
) -> dict[str, object]:
    """Latency attributable to transport, by subtracting a robot-local baseline.

    A measurement taken on the host includes latency the sensor pipeline already
    had before any network was involved. On this platform that baseline is large
    and asymmetric — the camera is tens of milliseconds behind its own header
    stamp on the robot itself, while the scanner is close to instantaneous — so
    attributing a host-side figure wholly to the link would overstate the network
    cost badly, and differently for each sensor.

    Subtraction is only meaningful for the same topic measured under the same
    conditions. Delivered fraction is reported as a ratio rather than a
    difference, because losing half of an already-degraded stream is a
    multiplicative effect, not an additive one.
    """

    def latency(summary: dict[str, object], key: str) -> float | None:
        block = summary.get("delivery_latency") if summary else None
        if isinstance(block, dict) and block.get(key) is not None:
            return float(block[key])
        return None

    rows = []
    for key in keys:
        base, observed = latency(robot_local, key), latency(host_observed, key)
        rows.append(
            {
                "measure": key,
                "robot_local_ms": base,
                "host_observed_ms": observed,
                "attributable_to_transport_ms": (
                    observed - base if (base is not None and observed is not None) else None
                ),
            }
        )

    def fraction(summary: dict[str, object]) -> float | None:
        value = summary.get("delivered_fraction") if summary else None
        return float(value) if value is not None else None

    base_fraction = fraction(robot_local)
    host_fraction = fraction(host_observed)
    surviving = None
    if base_fraction and host_fraction is not None and base_fraction > 0:
        surviving = host_fraction / base_fraction

    return {
        "rows": rows,
        "delivered_fraction": {
            "robot_local": base_fraction,
            "host_observed": host_fraction,
            "surviving_ratio": surviving,
        },
        "interpretation": (
            "difference is latency added between the robot and the host; it "
            "excludes the sensor pipeline's own delay, which the baseline "
            "already accounts for"
        ),
    }


def compare_sessions(
    label_a: str,
    summary_a: dict[str, object],
    label_b: str,
    summary_b: dict[str, object],
    *,
    keys: Sequence[str] = ("median_ms", "p95_ms", "p99_ms", "max_ms"),
) -> dict[str, object]:
    """Side-by-side comparison of two measurement sessions.

    Produces the table Appendix P needs. Missing values are reported as ``None``
    rather than zero, so an unmeasured configuration cannot be mistaken for a
    fast one.
    """

    def pick(summary: dict[str, object], key: str) -> float | None:
        latency = summary.get("delivery_latency") if summary else None
        if isinstance(latency, dict):
            value = latency.get(key)
            return float(value) if value is not None else None
        return None

    rows = []
    for key in keys:
        a, b = pick(summary_a, key), pick(summary_b, key)
        rows.append(
            {
                "measure": key,
                label_a: a,
                label_b: b,
                "difference_ms": (b - a) if (a is not None and b is not None) else None,
            }
        )
    return {"labels": [label_a, label_b], "rows": rows}
