#!/usr/bin/env bash
# Excluded-development Gate 6 smoke. Confirmatory identities are forbidden.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
W=artifacts/simulation/development_worlds
OUT="${GATE6_OUT:-/workspace/livifuser/evidence/gate6}"
export LIVIFUSER_ENCRYPTED_VOLUME_ACK="${LIVIFUSER_ENCRYPTED_VOLUME_ACK:-UNCONFIRMED}"
export SCIENTIFIC_DEADLINE_SEC="${SCIENTIFIC_DEADLINE_SEC:-20.0}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"
SUMMARY="$OUT/summary.jsonl"
: > "$SUMMARY"

run_one() {
  local name="$1" sdf="$2" json="$3" lidar="$4" obs="$5" variant="$6" seed="$7"
  local dest="$OUT/$name"
  echo "=== $name ==="
  set +e
  bash scripts/run_live_sim_development_episode.sh \
    "$dest" "$sdf" "$json" "$lidar" "$obs" "$variant" "$seed"
  local status=$?
  set -e
  local reason="none"
  if [[ -f "$dest/terminal.json" ]]; then
    reason="$(python3 -c "import json; print(json.load(open('$dest/terminal.json'))['terminal_reason'])")"
  elif [[ -f "$dest/operational_failure.json" ]]; then
    reason="operational_failure"
  fi
  printf '{"name":"%s","exit":%d,"terminal_reason":"%s"}\n' "$name" "$status" "$reason" | tee -a "$SUMMARY"
}

# Two topologies × C0, C1, C3b, C4. C1/C4 are visual world variants with C0 LiDAR.
# C3b is the analytic LiDAR condition on the C0 world.
run_one straight_c0 \
  $W/dev_straight_corridor_000.sdf $W/dev_straight_corridor_000.json \
  C0 71000001 full 20260805
run_one straight_c1 \
  $W/dev_straight_corridor_000.C1.sdf $W/dev_straight_corridor_000.C1.json \
  C0 71000011 full 20260805
run_one straight_c3b \
  $W/dev_straight_corridor_000.sdf $W/dev_straight_corridor_000.json \
  C3b 71000021 full 20260805
run_one straight_c4 \
  $W/dev_straight_corridor_000.C4.sdf $W/dev_straight_corridor_000.C4.json \
  C0 71000031 full 20260805
run_one dogleg_c0 \
  $W/dev_dogleg_corridor_001.sdf $W/dev_dogleg_corridor_001.json \
  C0 71000002 full 20260805
run_one dogleg_c1 \
  $W/dev_dogleg_corridor_001.C1.sdf $W/dev_dogleg_corridor_001.C1.json \
  C0 71000012 full 20260805
run_one dogleg_c3b \
  $W/dev_dogleg_corridor_001.sdf $W/dev_dogleg_corridor_001.json \
  C3b 71000022 full 20260805
run_one dogleg_c4 \
  $W/dev_dogleg_corridor_001.C4.sdf $W/dev_dogleg_corridor_001.C4.json \
  C0 71000032 full 20260805

python3 - "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
print("=== gate6 matrix ===")
for row in rows:
    print(f"{row['name']:16} exit={row['exit']}  {row['terminal_reason']}")
print("distinct terminals:", sorted({row["terminal_reason"] for row in rows}))
PY
