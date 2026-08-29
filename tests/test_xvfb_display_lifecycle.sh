#!/usr/bin/env bash
# The X display range must survive more episodes than it has slots.
#
# The confirmatory batch stopped after 110 episodes because the process-group
# teardown kills Xvfb, Xvfb never removes its own lock, and the scan treated any
# existing lock as occupied. All 110 slots filled with debris and every later
# episode died at the "no free X display number" guard. These checks exercise
# the two fixes without starting a simulator.
set -uo pipefail
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0
check() { if [[ "$2" == "$3" ]]; then echo "ok   - $1"; else echo "FAIL - $1 (want $3, got $2)"; fail=1; fi; }

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/run_live_sim_development_episode.sh"

# The scan must reclaim a lock whose pid is gone, and must not touch a live one.
scan() {
  local dir="$1" out=""
  for candidate in $(seq 90 92); do
    local lock="$dir/.X${candidate}-lock"
    if [[ ! -e "$lock" ]]; then out=":${candidate}"; break; fi
    local lock_pid; lock_pid="$(tr -dc '0-9' < "$lock" 2>/dev/null)"
    if [[ -z "$lock_pid" ]] || ! kill -0 "$lock_pid" 2>/dev/null; then
      rm -f "$lock"; out=":${candidate}"; break
    fi
  done
  echo "$out"
}

echo "99999999" > "$TMP/.X90-lock"          # dead pid
check "a lock naming a dead pid is reclaimed" "$(scan "$TMP")" ":90"
check "the reclaimed lock file is removed" "$([[ -e "$TMP/.X90-lock" ]] && echo present || echo gone)" "gone"

echo "$$" > "$TMP/.X90-lock"                # this shell: alive
echo "99999999" > "$TMP/.X91-lock"
check "a live lock is skipped, not stolen" "$(scan "$TMP")" ":91"
check "the live lock survives" "$([[ -e "$TMP/.X90-lock" ]] && echo present || echo gone)" "present"

printf '' > "$TMP/.X92-lock"                # empty/corrupt
rm -f "$TMP/.X90-lock" "$TMP/.X91-lock"
echo "$$" > "$TMP/.X90-lock"; echo "$$" > "$TMP/.X91-lock"
check "an unreadable lock is treated as debris" "$(scan "$TMP")" ":92"

# The script itself must carry both halves of the fix.
grep -q 'kill -0 "$lock_pid"' "$SCRIPT"
check "the script reclaims stale locks" "$?" "0"
grep -q 'rm -f "/tmp/.X${XVFB_DISPLAY#:}-lock"' "$SCRIPT"
check "teardown removes the episode's own lock" "$?" "0"

exit "$fail"
