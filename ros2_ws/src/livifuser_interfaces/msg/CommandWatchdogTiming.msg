# Command-watchdog timing observability, carried separately from
# CommandWatchdogStatus so that message keeps the wire layout already present in
# recorded bags. Published on /livifuser/command_watchdog_timing.
#
# The header stamp is the command timestamp taken inside the critical callback,
# so a timing sample joins to the status sample with the identical stamp.
#
# -1.0 marks "not applicable": the first tick has no predecessor, and the
# shutdown zero is published outside the timer.

std_msgs/Header header

# Monotonic interval between consecutive command-timer entries. At 10 Hz a
# healthy value is ~0.1; the 2026-07-30 keyboard bag implies values of 1.17,
# 2.28 and 3.60 during its three command outages.
float32 tick_interval_s

# Wall duration of the /cmd_vel publish call alone. A RELIABLE publisher whose
# subscriber is slow can block here, which no amount of threading prevents.
float32 command_publish_duration_s

# Age of the cached /cmd_vel publisher picture at decision time. Exceeding
# graph_max_age_ms forces zero output.
float32 graph_cache_age_s

# Cumulative diagnostic samples dropped because the queue was full. Any nonzero
# value invalidates a recording: the exporter's 150 ms zero-order hold can carry
# the previous action across a single missing 10 Hz sample without rejecting it,
# so a drop corrupts labels silently rather than visibly.
uint32 diagnostic_drops
