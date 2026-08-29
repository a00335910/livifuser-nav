#!/bin/sh
# Force Tailscale onto a DERP relay for a bounded window, then restore.
#
# Combination 3 asks what the relay costs. There is no supported "always relay"
# switch: `tailscale debug force-prefer-derp` only picks which region is home,
# not whether to relay at all. What does work is removing the direct path.
# Direct WireGuard runs over UDP/41641; DERP runs over TCP/443 to the relay. So
# dropping that UDP port kills direct and Tailscale falls back on its own.
#
# The revert is unconditional and time-based, and that is not optional here.
# The robot has no WiFi configuration any more, so Tailscale over the modem is
# the only way in. If the fallback does not happen, there is no session left to
# type a fix into and the rule must expire by itself.
#
# Run a SHORT window first (120 s) and confirm `tailscale ping` reports a DERP
# path before committing a full recording to it.
#
# Usage:  sudo nohup /tmp/force_derp.sh 600 >/tmp/force_derp.log 2>&1 &

WINDOW="${1:-600}"
PORT=41641

apply() {
    iptables -I INPUT  -p udp --dport "$PORT" -j DROP 2>/dev/null
    iptables -I OUTPUT -p udp --sport "$PORT" -j DROP 2>/dev/null
    iptables -I OUTPUT -p udp --dport "$PORT" -j DROP 2>/dev/null
}

revert() {
    iptables -D INPUT  -p udp --dport "$PORT" -j DROP 2>/dev/null
    iptables -D OUTPUT -p udp --sport "$PORT" -j DROP 2>/dev/null
    iptables -D OUTPUT -p udp --dport "$PORT" -j DROP 2>/dev/null
}

apply
echo "direct path blocked at $(date -Is); reverting in ${WINDOW}s"

# Nudge Tailscale to re-evaluate rather than wait for its own timer.
tailscale debug break-derp-conns 2>/dev/null
tailscale debug rebind 2>/dev/null
tailscale debug restun 2>/dev/null

sleep "$WINDOW"

revert
tailscale debug rebind 2>/dev/null
tailscale debug restun 2>/dev/null
echo "direct path restored at $(date -Is)"
