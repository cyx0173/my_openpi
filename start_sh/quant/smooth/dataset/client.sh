#!/usr/bin/env bash
set -uo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

CLIENT_ROOT="${CLIENT_ROOT:-/home/chengyuxuan/openpi/experiments/selector_dataset}"
SERVER_ROOT="${SERVER_ROOT:-/share/chengyuxuan-local/openpi/selector/recovery_data_selector}"

TASK_JSON="${1:-$CLIENT_ROOT/task_schedule.json}"

# 如果你已经把 clean 版覆盖到 experiments/mode6_data_selector.py，这行可以不改。
# 否则建议用新的 clean 文件名。
CLIENT="${CLIENT:-experiments/mode6_data_selector.py}"

LOG_DIR="$CLIENT_ROOT/logs/client"
mkdir -p "$LOG_DIR"

SLOTS=(${SLOTS_OVERRIDE:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15})
BASE_PORT="${BASE_PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

is_case_done() {
  local case_id="$1"
  local schedule="$2"

  [ "$FORCE_RERUN" = "1" ] && return 1

  python - "$CLIENT_ROOT" "$SERVER_ROOT" "$case_id" "$schedule" <<'PY'
import json
import sys
from pathlib import Path

client_root = Path(sys.argv[1])
server_root = Path(sys.argv[2])
case_id = sys.argv[3]
schedule_path = Path(sys.argv[4])

client_case_root = client_root / "cases" / case_id
server_case_root = server_root / "cases" / case_id
summary_path = client_case_root / "state_summary.json"

if not summary_path.exists() or summary_path.stat().st_size <= 0:
    sys.exit(1)

try:
    summary = json.load(open(summary_path, "r", encoding="utf-8"))
except Exception:
    sys.exit(1)

# 只跳过完整成功收集的 case。
if summary.get("success") is not True:
    sys.exit(1)

try:
    schedule = json.load(open(schedule_path, "r", encoding="utf-8"))
except Exception:
    sys.exit(1)

rows = schedule.get("selector_chunk_schedule") or schedule.get("chunk_schedule") or []
if not rows:
    sys.exit(1)

last_chunk = summary.get("last_chunk")
if last_chunk is None:
    sys.exit(1)
last_chunk = int(last_chunk)

needed_chunks = [int(r["chunk_idx"]) for r in rows if int(r["chunk_idx"]) <= last_chunk]
if not needed_chunks:
    sys.exit(1)

# 现在只检查 server-side mid 文件，不再检查 client-side state。
for k in needed_chunks:
    required = [
        server_case_root / "mid" / "w4a4" / f"chunk{k:04d}.npz",
        server_case_root / "mid" / "w4a8" / f"chunk{k:04d}.npz",
        server_case_root / "mid" / "w4a16" / f"chunk{k:04d}.npz",
    ]
    for p in required:
        if not p.exists() or p.stat().st_size <= 0:
            sys.exit(1)

sys.exit(0)
PY
}

run_case() {
  local slot="$1"
  local id="$2"
  local case_id="$3"
  local schedule="$4"

  local port=$((BASE_PORT + slot))
  local log="$LOG_DIR/id${id}_${case_id}.log"

  echo "[LAUNCH] id=${id} case=${case_id} slot=${slot} port=${port}"

  (
    uv run python "$CLIENT" "$schedule" \
      --port "$port"
  ) > "$log" 2>&1 &

  local pid=$!
  PIDS[$slot]="$pid"
  CASES[$slot]="$case_id"
  IDS[$slot]="$id"
}

wait_one() {
  local done_pid=""
  local status=0

  set +e
  wait -n -p done_pid
  status=$?
  set -u

  local done_slot=""
  for slot in "${SLOTS[@]}"; do
    if [ "${PIDS[$slot]:-}" = "$done_pid" ]; then
      done_slot="$slot"
      break
    fi
  done

  if [ -z "$done_slot" ]; then
    echo "[WARN] finished unknown pid=${done_pid:-none} rc=$status"
    active=$((active - 1))
    return 0
  fi

  local case_id="${CASES[$done_slot]}"
  local id="${IDS[$done_slot]}"

  if [ "$status" -eq 0 ]; then
    echo "[DONE] id=$id case=$case_id slot=$done_slot pid=$done_pid"
    finished=$((finished + 1))
  else
    echo "[FAIL] id=$id case=$case_id slot=$done_slot pid=$done_pid rc=$status log=$LOG_DIR/id${id}_${case_id}.log"
    failed=$((failed + 1))
  fi

  unset PIDS[$done_slot]
  unset CASES[$done_slot]
  unset IDS[$done_slot]
  FREE_SLOTS+=("$done_slot")
  active=$((active - 1))
}

declare -a PIDS
declare -a CASES
declare -a IDS
declare -a FREE_SLOTS

for slot in "${SLOTS[@]}"; do
  FREE_SLOTS+=("$slot")
done

active=0
finished=0
failed=0
skipped=0
launched=0

if [ ! -f "$TASK_JSON" ]; then
  echo "[ERROR] task json not found: $TASK_JSON" >&2
  exit 1
fi

if [ ! -f "$CLIENT" ]; then
  echo "[ERROR] client not found: $CLIENT" >&2
  exit 1
fi

echo "[START] client_root=$CLIENT_ROOT"
echo "[START] server_root=$SERVER_ROOT"
echo "[START] task_json=$TASK_JSON"
echo "[START] client=$CLIENT"
echo "[START] host=$HOST base_port=$BASE_PORT slots=${SLOTS[*]}"
echo "[START] log_dir=$LOG_DIR"

while IFS=$'\t' read -r id case_id schedule; do
  [ -f "$schedule" ] || continue

  if is_case_done "$case_id" "$schedule"; then
    echo "[SKIP] id=${id} case=${case_id} already complete"
    skipped=$((skipped + 1))
    continue
  fi

  while [ "${#FREE_SLOTS[@]}" -eq 0 ]; do
    wait_one
  done

  slot="${FREE_SLOTS[0]}"
  FREE_SLOTS=("${FREE_SLOTS[@]:1}")

  run_case "$slot" "$id" "$case_id" "$schedule"
  active=$((active + 1))
  launched=$((launched + 1))
done < <(
  python - "$TASK_JSON" <<'PY'
import json
import re
import sys

task_json = sys.argv[1]
tasks = json.load(open(task_json, "r", encoding="utf-8"))

def key_fn(x):
    m = re.search(r"task(\d+)_ep(\d+)", x["case_id"])
    if not m:
        return (999, 999999, x["case_id"])
    return (int(m.group(1)), int(m.group(2)), x["case_id"])

tasks = sorted(tasks, key=key_fn)

for new_id, x in enumerate(tasks):
    print(f'{new_id}\t{x["case_id"]}\t{x["schedule"]}')
PY
)

while [ "$active" -gt 0 ]; do
  wait_one
done

echo "[SUMMARY] launched=$launched finished=$finished skipped=$skipped failed=$failed"