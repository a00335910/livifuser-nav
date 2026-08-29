"""Raw-sample analysis of the remote-operation link recordings.

Reads the per-sample ``.jsonl`` directly rather than the derived
``.summary.json``, so every number printed here is recomputed from source. This
exists because the derived summaries answered the wrong question in one place:
they treated ``c3_run1`` as a two-condition run (relayed, then direct) and
reported a single relayed median of 93 ms. The raw samples show three regimes,
not two, and the middle one is invisible to Tailscale's own status output. See
the segment-boundary constants below.

Adds analyses the summaries do not carry:

* distribution shape at teleoperation-relevant thresholds, rather than
  percentiles alone;
* consecutive-loss burst length -- 17 scattered losses and 17 consecutive ones
  are very different events for a moving robot;
* the control gap, the longest interval with no acknowledged command, which is
  the quantity an operator actually experiences;
* blind distance, the control gap expressed as centimetres travelled;
* stationarity across the run, which is what caught the C3 regime change;
* a lattice test for discrete quantisation of the round-trip distribution.

Usage (from the repository root, where ``artifacts/network/`` lives)::

    python3 scripts/analyse_link_raw.py
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys

ART = pathlib.Path("artifacts/network")

# TurtleBot3 Burger maximum linear velocity, used to turn milliseconds into a
# distance the robot covers before the operator can see the consequence.
TB3_MAX_SPEED_MS = 0.22

# Segment boundaries.
#
# c2_run2: the operator ended the run early on a falling robot battery and
# instructed that the final 10 s be excluded. Every sample above 1000 ms in that
# recording falls inside that window, and whether they were the link or the
# robot's power state cannot be separated from this data.
C2_END_S = 418.0

# c3_run1: a within-run A/B that turned out to have three regimes, not two.
# force_derp.sh blocked udp/41641 to remove the direct path; Tailscale fell back
# to the DERP relay and did not re-punch a direct path until t=336.5 s. But all
# four probes -- UDP echo, ICMP, TCP connect and SSH -- also step together at
# t=236.2 s, from ~94 ms to ~55 ms, while Tailscale reports no path change at
# all: same relay (lhr), same empty CurAddr, before and after. The cause is
# below the level Tailscale exposes and is not identifiable from this data. It
# is kept as its own segment rather than averaged away, because averaging the
# two produces a relay cost that is true of neither.
C3_SLOW_END_S = 236.2
C3_DERP_END_S = 336.5

THRESHOLDS = (50, 100, 150, 200, 300, 500, 1000)

# Spacing of the discrete modes seen in the c3 slow segment. Tested, not assumed.
LATTICE_MS = 15.5
LATTICE_TOLERANCE_MS = 1.5


def load(name):
    with (ART / f"{name}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def pct(sorted_values, q):
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def describe(values):
    s = sorted(values)
    if not s:
        return None
    return {
        "n": len(s),
        "min": s[0],
        "p50": pct(s, 0.50),
        "p90": pct(s, 0.90),
        "p95": pct(s, 0.95),
        "p99": pct(s, 0.99),
        "p999": pct(s, 0.999),
        "max": s[-1],
        "mean": statistics.fmean(s),
        "stdev": statistics.stdev(s) if len(s) > 1 else 0.0,
    }


def jitter(values):
    """Consecutive absolute differences, in the spirit of RFC 3550."""
    deltas = [abs(b - a) for a, b in zip(values, values[1:], strict=False)]
    return describe(deltas) if deltas else None


def burst_runs(missing):
    """Group missing sequence numbers into consecutive runs.

    Scattered loss and burst loss are different failures. Seventeen packets lost
    one at a time is a lossy link; seventeen consecutive is 1.7 s during which
    the robot received nothing at all.
    """
    runs = []
    for seq in sorted(missing):
        if runs and seq == runs[-1][-1] + 1:
            runs[-1].append(seq)
        else:
            runs.append([seq])
    return runs


def lattice_fit(values):
    """Test whether the distribution sits on a discrete lattice.

    Returns the best-fitting phase offset and the fraction of samples within
    tolerance of it, alongside the fraction expected by chance and the span in
    lattice cells. The span matters: a distribution narrower than about three
    cells can be captured by a single cell, so a high hit rate there means
    nothing. Callers must check ``valid`` before reading ``hit_fraction``.
    """
    s = sorted(values)
    if len(s) < 50:
        return None
    chance = min(1.0, 2 * LATTICE_TOLERANCE_MS / LATTICE_MS)
    best_hits, best_offset = 0, 0.0
    for step in range(int(LATTICE_MS * 10)):
        offset = step / 10.0
        hits = 0
        for value in s:
            phase = (value - offset) % LATTICE_MS
            if min(phase, LATTICE_MS - phase) <= LATTICE_TOLERANCE_MS:
                hits += 1
        if hits > best_hits:
            best_hits, best_offset = hits, offset
    span_cells = (pct(s, 0.99) - pct(s, 0.01)) / LATTICE_MS
    return {
        "hit_fraction": best_hits / len(s),
        "chance_fraction": chance,
        "offset_ms": best_offset,
        "span_cells": span_cells,
        "valid": span_cells >= 3.0,
    }


def collect(name, t_min, t_max):
    """Pull every probe series out of one recording, within a time window."""
    sends, replies = {}, {}
    icmp, ssh, tcp, modes = [], [], [], []

    for row in load(name):
        t, probe = row["t"], row["probe"]
        in_window = t_min <= t <= t_max
        if probe == "udp_send" and not row.get("failed") and in_window:
            sends[row["seq"]] = t
        elif probe == "udp_reply":
            # Keyed by sequence, filtered later against the sends in window, so
            # a reply arriving just past the boundary still counts its packet.
            replies[row["seq"]] = (t, row["round_trip_ms"])
        elif probe == "icmp" and row.get("rtt_ms") is not None and in_window:
            icmp.append((t, row["rtt_ms"]))
        elif probe == "ssh" and not row.get("failed") and in_window:
            ssh.append((t, row["round_trip_ms"]))
        elif probe == "tcp" and not row.get("failed") and in_window:
            tcp.append((t, row["connect_ms"]))
        elif probe == "tailscale":
            modes.append((t, row["mode"], (row.get("peer") or {}).get("cur_addr")))

    return sends, replies, icmp, ssh, tcp, modes


def robot_receipts(name):
    """Robot-side receipt log: the other half of loss attribution.

    Without this, a probe that never returns says only that something failed.
    With it, uplink loss (the robot never got the command) separates from
    downlink loss (it acted and the operator saw nothing).
    """
    path = ART / f"{name}_pi.echo.jsonl"
    if not path.exists():
        return set(), []
    arrived, gaps = set(), []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            arrived.add(row["seq"])
            if row.get("since_previous_recv_s") is not None:
                gaps.append(row["since_previous_recv_s"] * 1000.0)
    return arrived, gaps


def analyse(label, name, t_min=-1.0, t_max=float("inf"), note=None):
    sends, replies, icmp, ssh, tcp, modes = collect(name, t_min, t_max)
    arrived, gaps = robot_receipts(name)

    sent_seqs = set(sends)
    replied = {s for s in replies if s in sent_seqs}
    arrived_here = arrived & sent_seqs
    lost_uplink = sent_seqs - arrived_here if arrived else set()
    lost_downlink = arrived_here - replied if arrived else set()
    lost_any = sent_seqs - replied

    rtts = [replies[s][1] for s in sorted(replied)]
    by_time = sorted((replies[s][0], replies[s][1]) for s in replied)
    stats = describe(rtts)

    print("=" * 78)
    print(f"{label}   [{name}]")
    if note:
        print(f"  {note}")
    print("=" * 78)
    lo = min(sends.values(), default=0.0)
    hi = max(sends.values(), default=0.0)
    print(f"window   t = {lo:.1f} .. {hi:.1f} s   ({hi - lo:.1f} s)\n")

    print("--- UDP echo round trip (ms) ---")
    print(f"  n={stats['n']}  min={stats['min']:.2f}  p50={stats['p50']:.2f}  "
          f"p90={stats['p90']:.2f}  p95={stats['p95']:.2f}  p99={stats['p99']:.2f}  "
          f"p99.9={stats['p999']:.2f}  max={stats['max']:.2f}")
    print(f"  mean={stats['mean']:.2f}  stdev={stats['stdev']:.2f}  "
          f"p95-p50={stats['p95'] - stats['p50']:.2f}")
    jit = jitter(rtts)
    if jit:
        print(f"  jitter |delta|  p50={jit['p50']:.2f}  p95={jit['p95']:.2f}  "
              f"p99={jit['p99']:.2f}  max={jit['max']:.2f}")
    print()

    print("--- Time above threshold ---")
    for threshold in THRESHOLDS:
        over = sum(1 for v in rtts if v > threshold)
        frac = 100.0 * over / len(rtts)
        print(f"  > {threshold:>5} ms : {over:>5} / {len(rtts)} = {frac:7.3f}%  "
              + "#" * min(40, int(frac * 0.4)))
    print()

    print("--- Loss and attribution ---")
    print(f"  sent                  {len(sent_seqs)}")
    if arrived:
        print(f"  arrived at robot      {len(arrived_here)}  (uplink lost {len(lost_uplink)})")
        print(f"  returned to operator  {len(replied)}  (downlink lost {len(lost_downlink)})")
    else:
        print("  (no robot-side receipt log; loss cannot be attributed)")
    print(f"  unreturned            {len(lost_any)} = "
          f"{100.0 * len(lost_any) / max(1, len(sent_seqs)):.3f}%")
    for run in burst_runs(lost_any)[:6]:
        print(f"    seq {run[0]}..{run[-1]}  {len(run)} consecutive  "
              f"at t={sends[run[0]]:.1f} s  = {len(run) * 0.1:.1f} s of commands")
    print()

    print("--- Control gap: longest interval with no acknowledged command ---")
    if by_time:
        received = sorted(t + rtt / 1000.0 for t, rtt in by_time)
        pairs = zip(received, received[1:], strict=False)
        gaps_s = sorted(((b - a, a) for a, b in pairs), reverse=True)
        for gap, at in gaps_s[:5]:
            print(f"  {gap * 1000:9.1f} ms at t={at:7.1f} s  -> uncommanded {gap:5.2f} s, "
                  f"{gap * TB3_MAX_SPEED_MS * 100:6.1f} cm at 0.22 m/s")
        median_gap = statistics.median(g for g, _ in gaps_s)
        print(f"  median inter-arrival {median_gap * 1000:.1f} ms (nominal 100 ms)")
    print()

    print("--- Blind distance at TB3 Burger top speed ---")
    for name_q in ("p50", "p95", "p99", "max"):
        value = stats[name_q]
        print(f"  {name_q:>4} {value:9.2f} ms -> {value / 1000 * TB3_MAX_SPEED_MS * 100:6.2f} cm")
    print()

    print("--- Stationarity across the window ---")
    if by_time:
        n = len(by_time)
        for i in range(3):
            chunk = by_time[i * n // 3:(i + 1) * n // 3]
            values = sorted(v for _, v in chunk)
            print(f"  third {i + 1}  t={chunk[0][0]:6.1f}-{chunk[-1][0]:6.1f} s  "
                  f"n={len(values):5}  p50={pct(values, 0.5):8.2f}  "
                  f"p95={pct(values, 0.95):8.2f}  max={values[-1]:9.2f}")
    print()

    print("--- Cross-probe agreement ---")
    for probe_name, series in (("icmp", icmp), ("ssh", ssh), ("tcp connect", tcp)):
        if series:
            values = sorted(v for _, v in series)
            print(f"  {probe_name:<12} n={len(values):5}  p50={pct(values, 0.5):8.2f}  "
                  f"p95={pct(values, 0.95):8.2f}  max={values[-1]:9.2f}")
    if icmp:
        icmp_median = statistics.median(v for _, v in icmp)
        print(f"  udp p50 - icmp p50 = {stats['p50'] - icmp_median:.2f} ms "
              "(echo and userspace cost above the raw link)")
    print()

    fit = lattice_fit(rtts)
    if fit:
        print(f"--- Quantisation ({LATTICE_MS} ms lattice) ---")
        if fit["valid"]:
            print(f"  {fit['hit_fraction'] * 100:.1f}% of samples within "
                  f"+/-{LATTICE_TOLERANCE_MS} ms of a lattice point "
                  f"(chance {fit['chance_fraction'] * 100:.1f}%), offset {fit['offset_ms']:.1f} ms")
        else:
            print(f"  not testable: distribution spans {fit['span_cells']:.1f} lattice cells, "
                  "under the 3-cell minimum, so a single cell would capture it trivially")
        print()

    if gaps:
        # A stall in the agent shows as a long silence followed by a burst of
        # near-zero gaps: it stopped calling recvfrom, packets queued in the
        # socket buffer, then all arrived at once. Nominal spacing is 100 ms.
        silences = [v for v in gaps if v > 500]
        bursts = sum(1 for v in gaps if v < 5)
        print("--- Robot agent health ---")
        print(f"  inter-receive gap  p50={statistics.median(gaps):.2f}  max={max(gaps):.2f} ms")
        print(f"  silences > 500 ms  {len(silences)}"
              + (f"  {[f'{v:.0f}' for v in silences[:6]]}" if silences else ""))
        print(f"  burst arrivals     {bursts} ({100.0 * bursts / len(gaps):.2f}% of receipts)")
        print()

    if modes:
        changes = [(t, m, addr) for i, (t, m, addr) in enumerate(modes)
                   if i == 0 or modes[i - 1][1] != m]
        print("--- Tailscale underlay path ---")
        for t, mode, addr in changes:
            print(f"  t={t:7.1f} s  {mode:<12} cur_addr={addr or '(none)'}")
        print()

    return {"label": label, "stats": stats, "jitter": jit, "sent": len(sent_seqs),
            "lost": len(lost_any), "uplink": len(lost_uplink),
            "downlink": len(lost_downlink)}


def changepoint_scan(name, window_s=10):
    """Print a rolling median so regime changes are visible rather than averaged.

    This is what found the c3 step at 236 s. Run it on any new recording before
    trusting a single median for the whole run.
    """
    rows = load(name)
    series = sorted((r["t"], r["round_trip_ms"]) for r in rows if r["probe"] == "udp_reply")
    if not series:
        return
    print(f"--- rolling median, {name}, {window_s} s bins ---")
    previous = None
    end = int(series[-1][0]) + 1
    for start in range(0, end, window_s):
        window = [v for t, v in series if start <= t < start + window_s]
        if not window:
            continue
        median = statistics.median(window)
        flag = ""
        if previous is not None and abs(median - previous) / previous > 0.15:
            flag = "   <== step"
        print(f"  {start:4d}-{start + window_s:4d} s  n={len(window):4d}  p50={median:8.2f}{flag}")
        previous = median
    print()


def main():
    results = [
        analyse("C1  WiFi / LAN", "c1_run3"),
        analyse("C2  cellular, direct", "c2_run2", t_max=C2_END_S,
                note="final 10 s excluded by operator instruction (falling battery)"),
        analyse("C3  relayed, slow regime", "c3_run1", t_max=C3_SLOW_END_S,
                note="DERP; Tailscale reports the same path here as in the next segment"),
        analyse("C3  relayed, fast regime", "c3_run1",
                t_min=C3_SLOW_END_S, t_max=C3_DERP_END_S,
                note="still DERP by Tailscale's own status; cause of the step unidentified"),
        analyse("C3  direct, same run", "c3_run1", t_min=C3_DERP_END_S,
                note="after Tailscale re-punched a direct path"),
    ]

    for name in ("c1_run3", "c2_run2", "c3_run1"):
        changepoint_scan(name, window_s=20)

    print("=" * 78)
    print("SIDE BY SIDE")
    print("=" * 78)
    print(f"{'':<24}" + "".join(f"{r['label'][:16]:>18}" for r in results))
    rows = (
        ("udp p50 (ms)", lambda r: f"{r['stats']['p50']:.2f}"),
        ("udp p95 (ms)", lambda r: f"{r['stats']['p95']:.2f}"),
        ("udp p99 (ms)", lambda r: f"{r['stats']['p99']:.2f}"),
        ("udp max (ms)", lambda r: f"{r['stats']['max']:.1f}"),
        ("stdev (ms)", lambda r: f"{r['stats']['stdev']:.2f}"),
        ("jitter p50 (ms)", lambda r: f"{r['jitter']['p50']:.2f}"),
        ("samples", lambda r: str(r["sent"])),
        ("unreturned", lambda r: str(r["lost"])),
        ("uplink lost", lambda r: str(r["uplink"])),
        ("downlink lost", lambda r: str(r["downlink"])),
    )
    for title, fn in rows:
        print(f"{title:<24}" + "".join(f"{fn(r):>18}" for r in results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
