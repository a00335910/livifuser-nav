"""Tests for network session latency and delivery statistics."""

from __future__ import annotations

import math
import unittest

from livifuser_nav.network_metrics import (
    NetworkMetricError,
    PingSession,
    attribute_loss,
    clock_offset_estimate,
    command_round_trip,
    compare_sessions,
    control_quality_bands,
    correlate,
    delivery_summary,
    find_spikes,
    jitter_summary,
    network_cost,
    outage_windows,
    parse_ping_rtt_ms,
    percentile,
    reordering_summary,
    split_by_state,
    summarize_ms,
)

MS = 1_000_000


class PercentileTests(unittest.TestCase):
    def test_interpolates_between_samples(self) -> None:
        self.assertAlmostEqual(percentile([0.0, 10.0], 0.5), 5.0)
        self.assertAlmostEqual(percentile([0.0, 10.0, 20.0], 0.5), 10.0)

    def test_empty_is_nan_not_zero(self) -> None:
        self.assertTrue(math.isnan(percentile([], 0.5)))

    def test_rejects_out_of_range_fraction(self) -> None:
        with self.assertRaisesRegex(NetworkMetricError, "fraction must be"):
            percentile([1.0, 2.0], 1.5)


class SummaryTests(unittest.TestCase):
    def test_reports_tail_not_only_centre(self) -> None:
        values = [10.0] * 99 + [500.0]
        s = summarize_ms(values)
        self.assertAlmostEqual(s["median_ms"], 10.0)
        self.assertAlmostEqual(s["max_ms"], 500.0)
        self.assertGreater(s["p99_ms"], 10.0)

    def test_empty_returns_none_not_zeroes(self) -> None:
        self.assertIsNone(summarize_ms([]))

    def test_non_finite_values_are_excluded(self) -> None:
        s = summarize_ms([1.0, float("nan"), 3.0, float("inf")])
        self.assertEqual(s["count"], 2)


class SpikeTests(unittest.TestCase):
    def test_groups_contiguous_excursions_into_episodes(self) -> None:
        """Three separate stalls must not read as scattered jitter."""

        times = [i * 0.1 for i in range(30)]
        values = [10.0] * 30
        for i in (5, 6, 7):
            values[i] = 300.0
        for i in (20, 21):
            values[i] = 250.0
        spikes = find_spikes(times, values, threshold_ms=100.0)
        self.assertEqual(len(spikes), 2)
        self.assertEqual(spikes[0].sample_count, 3)
        self.assertAlmostEqual(spikes[0].peak_ms, 300.0)
        self.assertEqual(spikes[1].sample_count, 2)

    def test_spike_at_end_of_series_is_closed(self) -> None:
        spikes = find_spikes([0.0, 0.1, 0.2], [10.0, 10.0, 900.0], threshold_ms=100.0)
        self.assertEqual(len(spikes), 1)
        self.assertAlmostEqual(spikes[0].peak_ms, 900.0)

    def test_no_excursion_yields_no_spikes(self) -> None:
        self.assertEqual(find_spikes([0.0, 1.0], [5.0, 6.0], threshold_ms=100.0), [])

    def test_mismatched_lengths_rejected(self) -> None:
        with self.assertRaisesRegex(NetworkMetricError, "equal length"):
            find_spikes([0.0], [1.0, 2.0], threshold_ms=1.0)


class DeliverySummaryTests(unittest.TestCase):
    @staticmethod
    def stream(count: int, period_ms: float, latency_ms: float):
        header = [int(i * period_ms * MS) for i in range(count)]
        arrival = [h + int(latency_ms * MS) for h in header]
        return arrival, header

    def test_clean_stream_reports_expected_rate_and_latency(self) -> None:
        arrival, header = self.stream(100, 100.0, 12.0)
        s = delivery_summary(
            arrival,
            header,
            expected_hz=10.0,
            gap_thresholds_ms=[150.0],
            spike_threshold_ms=100.0,
        )
        self.assertAlmostEqual(s["delivery_latency"]["median_ms"], 12.0, places=3)
        self.assertAlmostEqual(s["achieved_hz"], 10.0, places=3)
        self.assertAlmostEqual(s["delivered_fraction"], 1.0, places=3)
        self.assertEqual(s["gaps_above_threshold_ms"]["150"], 0)
        self.assertEqual(s["spike_count"], 0)

    def test_dropped_messages_reduce_delivered_fraction(self) -> None:
        """Half the stream lost must not read as a healthy link."""

        arrival, header = self.stream(100, 100.0, 12.0)
        arrival, header = arrival[::2], header[::2]
        s = delivery_summary(
            arrival,
            header,
            expected_hz=10.0,
            gap_thresholds_ms=[150.0],
            spike_threshold_ms=100.0,
        )
        self.assertLess(s["delivered_fraction"], 0.55)
        self.assertAlmostEqual(s["achieved_hz"], 5.0, places=2)
        # Latency alone would still have looked perfect.
        self.assertAlmostEqual(s["delivery_latency"]["median_ms"], 12.0, places=3)

    def test_unknown_expected_rate_reports_none_not_full_delivery(self) -> None:
        arrival, header = self.stream(20, 100.0, 5.0)
        s = delivery_summary(
            arrival, header, expected_hz=None, gap_thresholds_ms=[150.0], spike_threshold_ms=50.0
        )
        self.assertIsNone(s["delivered_fraction"])
        self.assertIsNotNone(s["achieved_hz"])

    def test_gap_is_counted_against_threshold(self) -> None:
        arrival, header = self.stream(10, 100.0, 5.0)
        arrival = arrival[:5] + [a + 400 * MS for a in arrival[5:]]
        s = delivery_summary(
            arrival, header, expected_hz=10.0, gap_thresholds_ms=[150.0], spike_threshold_ms=1e9
        )
        self.assertEqual(s["gaps_above_threshold_ms"]["150"], 1)

    def test_single_sample_is_reported_as_insufficient(self) -> None:
        s = delivery_summary(
            [0], [0], expected_hz=10.0, gap_thresholds_ms=[150.0], spike_threshold_ms=100.0
        )
        self.assertTrue(s["insufficient_samples"])
        self.assertIsNone(s["delivery_latency"])

    def test_out_of_order_arrivals_are_sorted(self) -> None:
        arrival, header = self.stream(10, 100.0, 5.0)
        s = delivery_summary(
            list(reversed(arrival)),
            list(reversed(header)),
            expected_hz=10.0,
            gap_thresholds_ms=[150.0],
            spike_threshold_ms=100.0,
        )
        self.assertAlmostEqual(s["achieved_hz"], 10.0, places=3)

    def test_mismatched_lengths_rejected(self) -> None:
        with self.assertRaisesRegex(NetworkMetricError, "equal length"):
            delivery_summary(
                [0, 1], [0], expected_hz=None, gap_thresholds_ms=[], spike_threshold_ms=1.0
            )


class PingParsingTests(unittest.TestCase):
    def test_parses_linux_reply(self) -> None:
        line = "64 bytes from 192.168.0.33: icmp_seq=1 ttl=64 time=12.4 ms"
        self.assertAlmostEqual(parse_ping_rtt_ms(line), 12.4)

    def test_parses_windows_reply(self) -> None:
        line = "Reply from 192.168.0.33: bytes=32 time=13ms TTL=64"
        self.assertAlmostEqual(parse_ping_rtt_ms(line), 13.0)

    def test_timeout_is_not_a_fast_reply(self) -> None:
        self.assertIsNone(parse_ping_rtt_ms("Request timed out."))
        self.assertIsNone(parse_ping_rtt_ms("From 10.0.0.1 Destination Host Unreachable"))

    def test_summary_lines_are_ignored(self) -> None:
        self.assertIsNone(parse_ping_rtt_ms("5 packets transmitted, 5 received, 0% packet loss"))

    def test_session_counts_losses_separately(self) -> None:
        session = PingSession(target="192.168.0.33")
        for line in (
            "64 bytes from x: icmp_seq=1 ttl=64 time=10.0 ms",
            "Request timed out.",
            "64 bytes from x: icmp_seq=3 ttl=64 time=300.0 ms",
        ):
            session.record_line(line)
        summary = session.summary(spike_threshold_ms=100.0)
        self.assertEqual(summary["replies"], 2)
        self.assertEqual(summary["lost"], 1)
        self.assertAlmostEqual(summary["loss_fraction"], 1 / 3)
        self.assertEqual(summary["replies_above_threshold"], 1)


class CommandRoundTripTests(unittest.TestCase):
    def test_matches_each_command_to_the_preceding_intent(self) -> None:
        intents = [0, 100 * MS, 200 * MS]
        finals = [30 * MS, 130 * MS, 230 * MS]
        result = command_round_trip(intents, finals)
        self.assertEqual(result["matched"], 3)
        self.assertAlmostEqual(result["round_trip"]["median_ms"], 30.0, places=3)

    def test_future_intent_is_never_credited(self) -> None:
        """A command cannot have been caused by an intent that came after it."""

        result = command_round_trip([500 * MS], [100 * MS])
        self.assertEqual(result["matched"], 0)
        self.assertEqual(result["unmatched"], 1)

    def test_intent_outside_window_is_unmatched(self) -> None:
        result = command_round_trip([0], [900 * MS], match_window_ms=500.0)
        self.assertEqual(result["matched"], 0)
        self.assertEqual(result["unmatched"], 1)

    def test_empty_input_is_handled(self) -> None:
        self.assertIsNone(command_round_trip([], [])["round_trip"])


class ControlBandTests(unittest.TestCase):
    def test_bimodal_link_is_distinguished_from_a_steady_one(self) -> None:
        """Same mean, very different to drive."""

        steady = control_quality_bands([90.0] * 100)
        bimodal = control_quality_bands([20.0] * 50 + [160.0] * 50)
        self.assertAlmostEqual(steady["fractions"]["responsive"], 1.0)
        self.assertAlmostEqual(bimodal["fractions"]["responsive"], 0.5)
        self.assertAlmostEqual(bimodal["fractions"]["noticeable_lag"], 0.5)

    def test_covers_the_full_range_including_unusable(self) -> None:
        result = control_quality_bands([50.0, 200.0, 500.0, 5000.0])
        self.assertEqual(result["counts"]["responsive"], 1)
        self.assertEqual(result["counts"]["noticeable_lag"], 1)
        self.assertEqual(result["counts"]["degraded"], 1)
        self.assertEqual(result["counts"]["unusable"], 1)

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(control_quality_bands([])["bands"])


class OutageTests(unittest.TestCase):
    def test_finds_gaps_where_samples_stopped(self) -> None:
        times = [i * 0.2 for i in range(10)] + [8.0, 8.2, 8.4]
        result = outage_windows(times, expected_interval_s=0.2)
        self.assertEqual(result["outage_count"], 1)
        self.assertAlmostEqual(result["longest_s"], 6.2, places=3)

    def test_steady_sampling_has_no_outage(self) -> None:
        times = [i * 0.2 for i in range(50)]
        self.assertEqual(outage_windows(times, expected_interval_s=0.2)["outage_count"], 0)

    def test_single_sample_is_handled(self) -> None:
        self.assertIsNone(outage_windows([1.0], expected_interval_s=0.2)["longest_s"])

    def test_baseline_defaults_to_the_observed_rate(self) -> None:
        """The bug this guards: a 1 Hz probe assumed to be 5 Hz read as one long outage."""

        times = [float(i) for i in range(60)]  # steady 1 Hz, no outage at all
        assumed = outage_windows(times, expected_interval_s=0.2)
        derived = outage_windows(times)
        self.assertEqual(assumed["outage_count"], 59)
        self.assertEqual(derived["outage_count"], 0)
        self.assertAlmostEqual(derived["observed_rate_hz"], 1.0)

    def test_derived_baseline_still_finds_a_genuine_outage(self) -> None:
        times = [float(i) for i in range(20)] + [40.0, 41.0, 42.0]
        result = outage_windows(times)
        self.assertEqual(result["outage_count"], 1)
        self.assertAlmostEqual(result["longest_s"], 21.0)


class CorrelationTests(unittest.TestCase):
    def test_detects_latency_rising_as_signal_falls(self) -> None:
        t = [float(i) for i in range(40)]
        signal = [-70.0 + i for i in range(40)]
        latency = [200.0 - 2.0 * i for i in range(40)]
        result = correlate(t, latency, t, signal)
        self.assertEqual(result["paired"], 40)
        self.assertLess(result["pearson_r"], -0.95)

    def test_unsynchronised_series_pair_by_nearest_timestamp(self) -> None:
        a_t = [0.0, 1.0, 2.0]
        b_t = [0.1, 1.05, 2.2]
        result = correlate(a_t, [1.0, 2.0, 3.0], b_t, [1.0, 2.0, 3.0], window_s=0.5)
        self.assertEqual(result["paired"], 3)

    def test_samples_outside_window_are_counted_not_dropped(self) -> None:
        result = correlate([0.0, 50.0], [1.0, 2.0], [0.1], [1.0], window_s=0.5)
        self.assertEqual(result["unpaired"], 1)

    def test_too_few_pairs_reports_no_correlation(self) -> None:
        self.assertIsNone(correlate([0.0], [1.0], [0.0], [1.0])["pearson_r"])

    def test_mismatched_series_rejected(self) -> None:
        with self.assertRaisesRegex(NetworkMetricError, "matching times and values"):
            correlate([0.0, 1.0], [1.0], [0.0], [1.0])


class LossAttributionTests(unittest.TestCase):
    def test_separates_uplink_from_downlink_loss(self) -> None:
        """The distinction the operator alone cannot make."""

        sent = list(range(10))
        robot_received = [0, 1, 2, 3, 4, 5, 6]  # 7,8,9 never arrived
        operator_replied = [0, 1, 2, 3]  # 4,5,6 arrived but no reply came back
        result = attribute_loss(sent, robot_received, operator_replied)
        self.assertEqual(result["uplink_lost"], 3)
        self.assertEqual(result["downlink_lost"], 3)
        self.assertEqual(result["returned_to_operator"], 4)
        self.assertAlmostEqual(result["uplink_loss_fraction"], 0.3)
        self.assertAlmostEqual(result["downlink_loss_fraction"], 3 / 7)

    def test_clean_link_reports_no_loss_either_way(self) -> None:
        sent = list(range(5))
        result = attribute_loss(sent, sent, sent)
        self.assertEqual(result["uplink_lost"], 0)
        self.assertEqual(result["downlink_lost"], 0)

    def test_all_uplink_loss_when_robot_saw_nothing(self) -> None:
        result = attribute_loss([0, 1, 2], [], [])
        self.assertEqual(result["uplink_lost"], 3)
        self.assertEqual(result["downlink_lost"], 0)
        self.assertIsNone(result["downlink_loss_fraction"])

    def test_logs_from_different_runs_are_flagged(self) -> None:
        result = attribute_loss([0, 1], [0, 1, 99], [0])
        self.assertEqual(result["unexpected_sequences"], 1)
        self.assertEqual(result["arrived_at_robot"], 2)


class ClockOffsetTests(unittest.TestCase):
    def test_round_trip_is_exact_without_clock_sync(self) -> None:
        """A 500 ms clock offset must not disturb the round trip."""

        skew = 0.5
        # Symmetric path: 40 ms each way, 5 ms robot processing.
        t_send, t_final = 0.0, 0.085
        t_robot_recv, t_robot_send = 0.040 + skew, 0.045 + skew
        result = clock_offset_estimate(t_send, t_robot_recv, t_robot_send, t_final)
        self.assertAlmostEqual(result["round_trip_ms"], 80.0, places=6)
        self.assertAlmostEqual(result["robot_processing_ms"], 5.0, places=6)
        self.assertAlmostEqual(result["estimated_clock_offset_ms"], skew * 1000.0, places=6)

    def test_synced_clocks_split_the_legs(self) -> None:
        # 30 ms each way, 2 ms processing, clocks agreeing.
        result = clock_offset_estimate(0.0, 0.030, 0.032, 0.062)
        self.assertAlmostEqual(result["estimated_clock_offset_ms"], 0.0, places=6)
        self.assertAlmostEqual(result["uplink_ms_estimate"], 30.0, places=3)
        self.assertAlmostEqual(result["downlink_ms_estimate"], 30.0, places=3)

    def test_asymmetric_path_biases_the_one_way_split(self) -> None:
        """The documented failure mode, pinned so it cannot be forgotten.

        Clocks agree exactly, but uplink is 30 ms and downlink 68 ms. The
        symmetric-path assumption splits the difference, so each leg is wrong by
        half the asymmetry while the round trip stays exact. On cellular, where
        uplink and downlink genuinely differ, this is the normal case.
        """

        result = clock_offset_estimate(0.0, 0.030, 0.032, 0.100)
        self.assertAlmostEqual(result["round_trip_ms"], 98.0, places=6)
        self.assertAlmostEqual(result["uplink_ms_estimate"], 49.0, places=3)
        self.assertAlmostEqual(result["downlink_ms_estimate"], 49.0, places=3)
        # True legs were 30 and 68; the estimate landed on their mean.
        self.assertAlmostEqual(
            result["uplink_ms_estimate"], (30.0 + 68.0) / 2, places=3
        )

    def test_one_way_split_is_labelled_as_an_assumption(self) -> None:
        result = clock_offset_estimate(0.0, 0.03, 0.031, 0.09)
        self.assertIn("symmetric", result["one_way_caveat"])


class SplitByStateTests(unittest.TestCase):
    def test_a_session_that_changed_mode_is_reported_as_two_distributions(self) -> None:
        """A relay fallback partway through must not average into one number."""

        times = [float(i) for i in range(20)]
        values = [20.0] * 10 + [180.0] * 10
        state_t = [0.0, 10.0]
        state_l = ["direct", "derp:lhr"]
        result = split_by_state(times, values, state_t, state_l)
        self.assertEqual(set(result["states"]), {"direct", "derp:lhr"})
        self.assertAlmostEqual(result["states"]["direct"]["summary"]["median_ms"], 20.0)
        self.assertAlmostEqual(result["states"]["derp:lhr"]["summary"]["median_ms"], 180.0)

    def test_state_is_held_until_the_next_observation(self) -> None:
        """Polling is periodic; the mode applies between polls, not only at them."""

        result = split_by_state([0.5, 1.5, 2.5], [10.0, 11.0, 12.0], [0.0], ["direct"])
        self.assertEqual(result["states"]["direct"]["sample_count"], 3)

    def test_samples_before_any_state_observation_are_unassigned(self) -> None:
        result = split_by_state([0.0, 5.0], [1.0, 2.0], [2.0], ["direct"])
        self.assertEqual(result["unassigned"], 1)
        self.assertEqual(result["states"]["direct"]["sample_count"], 1)

    def test_no_state_observations_leaves_everything_unassigned(self) -> None:
        result = split_by_state([0.0, 1.0], [1.0, 2.0], [], [])
        self.assertEqual(result["unassigned"], 2)
        self.assertEqual(result["states"], {})

    def test_mismatched_lengths_rejected(self) -> None:
        with self.assertRaisesRegex(NetworkMetricError, "sample times and values"):
            split_by_state([0.0], [1.0, 2.0], [0.0], ["x"])
        with self.assertRaisesRegex(NetworkMetricError, "state times and labels"):
            split_by_state([0.0], [1.0], [0.0, 1.0], ["x"])


class NetworkCostTests(unittest.TestCase):
    def test_subtracts_the_robot_local_baseline(self) -> None:
        """The camera is already tens of ms stale before any network is involved."""

        robot = {"delivery_latency": {"median_ms": 44.8, "p95_ms": 48.0}}
        host = {"delivery_latency": {"median_ms": 217.0, "p95_ms": 260.0}}
        cost = network_cost(robot, host, keys=("median_ms", "p95_ms"))
        median = next(r for r in cost["rows"] if r["measure"] == "median_ms")
        self.assertAlmostEqual(median["attributable_to_transport_ms"], 172.2, places=3)

    def test_low_baseline_sensor_attributes_almost_everything_to_transport(self) -> None:
        robot = {"delivery_latency": {"median_ms": 1.4}}
        host = {"delivery_latency": {"median_ms": 90.0}}
        cost = network_cost(robot, host, keys=("median_ms",))
        self.assertAlmostEqual(cost["rows"][0]["attributable_to_transport_ms"], 88.6, places=3)

    def test_missing_baseline_reports_none_not_full_attribution(self) -> None:
        host = {"delivery_latency": {"median_ms": 90.0}}
        cost = network_cost({}, host, keys=("median_ms",))
        self.assertIsNone(cost["rows"][0]["attributable_to_transport_ms"])
        self.assertAlmostEqual(cost["rows"][0]["host_observed_ms"], 90.0)

    def test_delivered_fraction_is_a_ratio_not_a_difference(self) -> None:
        robot = {"delivery_latency": {"median_ms": 1.0}, "delivered_fraction": 1.0}
        host = {"delivery_latency": {"median_ms": 2.0}, "delivered_fraction": 0.5}
        cost = network_cost(robot, host, keys=("median_ms",))
        self.assertAlmostEqual(cost["delivered_fraction"]["surviving_ratio"], 0.5)


class ComparisonTests(unittest.TestCase):
    def test_builds_side_by_side_rows(self) -> None:
        a = {
            "delivery_latency": {
                "median_ms": 10.0, "p95_ms": 20.0, "p99_ms": 25.0, "max_ms": 30.0
            }
        }
        b = {
            "delivery_latency": {
                "median_ms": 40.0, "p95_ms": 90.0, "p99_ms": 150.0, "max_ms": 900.0
            }
        }
        table = compare_sessions("router", a, "direct", b)
        self.assertEqual(table["labels"], ["router", "direct"])
        median_row = next(r for r in table["rows"] if r["measure"] == "median_ms")
        self.assertAlmostEqual(median_row["difference_ms"], 30.0)

    def test_missing_session_reports_none_not_zero(self) -> None:
        a = {"delivery_latency": {"median_ms": 10.0}}
        table = compare_sessions("router", a, "direct", {})
        row = next(r for r in table["rows"] if r["measure"] == "median_ms")
        self.assertIsNone(row["direct"])
        self.assertIsNone(row["difference_ms"])


if __name__ == "__main__":
    unittest.main()


class TestJitterSummary(unittest.TestCase):
    """Jitter is reported because for teleoperation it often matters more than
    latency: a steady 200 ms link is drivable, a 40 ms link swinging to 300 is not."""

    def test_a_perfectly_steady_link_has_no_jitter(self):
        self.assertEqual(jitter_summary([20.0] * 10)["max_ms"], 0.0)

    def test_jitter_measures_consecutive_change_not_spread(self):
        """A slow drift and a wild alternation can share a min and a max while
        being completely different links. Only consecutive deltas separate them."""

        drift = jitter_summary([10.0, 11.0, 12.0, 13.0, 14.0])
        alternating = jitter_summary([10.0, 14.0, 10.0, 14.0, 10.0])
        self.assertEqual(drift["max_ms"], 1.0)
        self.assertEqual(alternating["max_ms"], 4.0)

    def test_delta_count_is_one_fewer_than_samples(self):
        self.assertEqual(jitter_summary([1.0, 2.0, 3.0])["count"], 2)

    def test_too_few_samples_returns_none_rather_than_zero(self):
        """Zero jitter and no measurement are different claims."""

        self.assertIsNone(jitter_summary([]))
        self.assertIsNone(jitter_summary([5.0]))

    def test_non_finite_samples_are_dropped(self):
        self.assertEqual(jitter_summary([1.0, float("nan"), 2.0])["count"], 1)


class TestReorderingSummary(unittest.TestCase):
    def test_in_order_arrival_reports_none(self):
        result = reordering_summary([0, 1, 2, 3, 4])
        self.assertEqual(result["out_of_order"], 0)
        self.assertEqual(result["max_displacement"], 0)

    def test_a_single_straggler_is_detected(self):
        result = reordering_summary([0, 1, 3, 4, 2])
        self.assertEqual(result["out_of_order"], 1)
        self.assertEqual(result["max_displacement"], 2)

    def test_mean_displacement_separates_a_straggler_from_a_shuffled_stream(self):
        """One late straggler is survivable; a shuffled command stream is not.

        max_displacement does NOT tell them apart -- pinned below, because the
        docstring originally claimed it did. Both streams move their worst
        packet the same distance. Only the mean, which the straggler leaves
        near zero for every other packet, separates them.
        """

        straggler = reordering_summary([1, 2, 3, 4, 0])
        shuffled = reordering_summary([4, 3, 2, 1, 0])
        self.assertEqual(straggler["out_of_order"], 1)
        self.assertEqual(shuffled["out_of_order"], 4)
        self.assertEqual(straggler["max_displacement"], shuffled["max_displacement"])
        self.assertLess(straggler["mean_displacement"], shuffled["mean_displacement"])

    def test_gaps_from_lost_packets_are_not_counted_as_reordering(self):
        """Loss is already measured elsewhere; counting it twice would overstate
        reordering on exactly the runs that lost packets."""

        result = reordering_summary([0, 1, 5, 6, 9])
        self.assertEqual(result["out_of_order"], 0)

    def test_empty_and_single_inputs_do_not_divide_by_zero(self):
        self.assertEqual(reordering_summary([])["out_of_order_fraction"], 0.0)
        self.assertEqual(reordering_summary([7])["out_of_order_fraction"], 0.0)
