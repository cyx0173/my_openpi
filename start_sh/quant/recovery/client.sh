#!/usr/bin/env bash
set -uo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

DATA_ROOT="/home/chengyuxuan/openpi/experiments/smooth/mode7"

INDEX_JSON="$DATA_ROOT/data.json"
CLIENT="experiments/mode7_recovery.py"

OUT_DIR="$DATA_ROOT/recovery_schedules"
CACHE_DIR="$DATA_ROOT/recovery_restore_cache"
LOG_DIR="$DATA_ROOT/logs/recovery_clients"

mkdir -p "$LOG_DIR"

START_ID="${1:-0}"
END_ID="${2:-7}"

NUM_WORKERS=8
BASE_PORT=8000
PORT_OFFSET=8
POLICY_HOST="0.0.0.0"

MASTER_LOG="$LOG_DIR/master_${START_ID}_${END_ID}.log"
PID_FILE="$LOG_DIR/master_${START_ID}_${END_ID}.pid"

# 默认后台运行。
# 前台调试：
#   RECOVERY_FOREGROUND=1 bash start_sh/quant/recovery/client.sh 0 7
if [ "${RECOVERY_FOREGROUND:-0}" != "1" ] && [ "${RECOVERY_DAEMONIZED:-0}" != "1" ]; then
  echo "[DAEMON] starting recovery clients in background"
  echo "[DAEMON] range=${START_ID}-${END_ID}"
  echo "[DAEMON] master_log=${MASTER_LOG}"
  echo "[DAEMON] pid_file=${PID_FILE}"

  export RECOVERY_DAEMONIZED=1
  nohup setsid bash "$0" "$START_ID" "$END_ID" > "$MASTER_LOG" 2>&1 < /dev/null &
  echo "$!" > "$PID_FILE"

  echo "[DAEMON] pid=$(cat "$PID_FILE")"
  echo "[DAEMON] use: tail -f $MASTER_LOG"
  exit 0
fi

if [ ! -f "$INDEX_JSON" ]; then
  echo "[ERROR] data.json not found: $INDEX_JSON"
  exit 1
fi

declare -a slot_pids
declare -a slot_jobs
declare -a slot_cases

cleanup() {
  echo "[STOP] stopping current recovery clients..."
  for slot in $(seq 0 $((NUM_WORKERS - 1))); do
    pid="${slot_pids[$slot]:-}"
    if [ -n "$pid" ]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  exit 130
}

trap cleanup INT TERM

get_job_path() {
  local job_id="$1"

  python -c '
import json, sys
index_path = sys.argv[1]
job_id = int(sys.argv[2])

with open(index_path, "r", encoding="utf-8") as f:
    data = json.load(f)

p = None

m = data.get("id_to_job_path")
if isinstance(m, dict):
    p = m.get(str(job_id))

if not p:
    for row in data.get("jobs", []):
        if int(row.get("id", -1)) == job_id:
            p = row.get("job_path")
            break

if not p:
    raise SystemExit(f"[ERROR] cannot find job_id={job_id} in {index_path}")

print(p)
' "$INDEX_JSON" "$job_id"
}

get_case_id() {
  local job_path="$1"

  python -c '
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get("case_id", "unknown_case"))
' "$job_path"
}

already_done() {
  local case_id="$1"
  local case_dir="$OUT_DIR/$case_id"
  local schedule_path="$case_dir/recovery_schedule.json"
  local video_path="$case_dir/final_path.mp4"
  local action_path="$case_dir/final_actions.json"

  if [ ! -f "$schedule_path" ]; then
    return 1
  fi

  python - "$schedule_path" "$video_path" "$action_path" <<'PY'
import json
import sys
from pathlib import Path

schedule = Path(sys.argv[1])
video = Path(sys.argv[2])
actions = Path(sys.argv[3])

try:
    s = json.load(open(schedule, "r", encoding="utf-8"))
except Exception:
    raise SystemExit(1)

# 失败 case 没有最终成功路径，不需要 video/actions。
if s.get("final_success") is not True:
    raise SystemExit(0)

# 成功 case 必须同时有 video 和 action。
if video.exists() and actions.exists():
    raise SystemExit(0)

raise SystemExit(1)
PY
}

active_count() {
  local n=0
  for slot in $(seq 0 $((NUM_WORKERS - 1))); do
    if [ -n "${slot_pids[$slot]:-}" ]; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

active_pids() {
  for slot in $(seq 0 $((NUM_WORKERS - 1))); do
    if [ -n "${slot_pids[$slot]:-}" ]; then
      echo "${slot_pids[$slot]}"
    fi
  done
}

launch_job_in_slot() {
  local job_id="$1"
  local slot="$2"

  local port_a4=$((BASE_PORT + slot))
  local port_a8=$((BASE_PORT + slot + PORT_OFFSET))
  local port_a16=$((BASE_PORT + slot + PORT_OFFSET * 2))

  local job_path
  if ! job_path="$(get_job_path "$job_id")"; then
    echo "[ERROR] cannot resolve job_id=${job_id}"
    return 2
  fi

  if [ -z "$job_path" ] || [ ! -f "$job_path" ]; then
    echo "[ERROR] recovery_job not found for job_id=${job_id}: $job_path"
    return 2
  fi

  local case_id
  case_id="$(get_case_id "$job_path")"

  if already_done "$case_id"; then
    echo "[SKIP] job_id=${job_id} case=${case_id} already done"
    return 2
  fi

  local log_file="$LOG_DIR/job_${job_id}_slot_${slot}_${case_id}.log"

  echo "[LAUNCH] job_id=${job_id} slot=${slot} case=${case_id} A4=${port_a4} A8=${port_a8} A16=${port_a16}"
  echo "[LOG] $log_file"

  uv run python "$CLIENT" \
    --job "$job_path" \
    --out "$OUT_DIR" \
    --policy-host "$POLICY_HOST" \
    --port-a4 "$port_a4" \
    --port-a8 "$port_a8" \
    --port-a16 "$port_a16" \
    --save-trial-summary \
    > "$log_file" 2>&1 &

  local pid="$!"

  slot_pids[$slot]="$pid"
  slot_jobs[$slot]="$job_id"
  slot_cases[$slot]="$case_id"

  return 0
}

fill_slot() {
  local slot="$1"

  while [ "$job_id" -le "$END_ID" ]; do
    local cur_job_id="$job_id"
    job_id=$((job_id + 1))

    if launch_job_in_slot "$cur_job_id" "$slot"; then
      return 0
    fi
  done

  return 1
}

echo "[START] recovery worker-pool client"
echo "[RANGE] ${START_ID}-${END_ID}"
echo "[WORKERS] ${NUM_WORKERS}"
echo "[INDEX] $INDEX_JSON"
echo "[OUT] $OUT_DIR"
echo "[CACHE] $CACHE_DIR"
echo "[MASTER_LOG] $MASTER_LOG"

job_id="$START_ID"

# 先填满 8 个 worker
for slot in $(seq 0 $((NUM_WORKERS - 1))); do
  fill_slot "$slot" || true
done

# worker pool:
# 哪个 slot 结束，就立刻补下一个 job
while [ "$(active_count)" -gt 0 ]; do
  mapfile -t pids_now < <(active_pids)

  done_pid=""

  wait -n -p done_pid "${pids_now[@]}"
  rc="$?"

  finished_slot=""

  for slot in $(seq 0 $((NUM_WORKERS - 1))); do
    if [ "${slot_pids[$slot]:-}" = "${done_pid:-}" ]; then
      finished_slot="$slot"
      break
    fi
  done

  if [ -z "$finished_slot" ]; then
    echo "[WARN] finished pid=${done_pid:-unknown} but cannot find slot"
    continue
  fi

  old_job="${slot_jobs[$finished_slot]:-unknown}"
  old_case="${slot_cases[$finished_slot]:-unknown}"

  echo "[FINISH] slot=${finished_slot} pid=${done_pid} job_id=${old_job} case=${old_case} rc=${rc}"

  unset "slot_pids[$finished_slot]"
  unset "slot_jobs[$finished_slot]"
  unset "slot_cases[$finished_slot]"

  fill_slot "$finished_slot" || true
done

echo "[DONE] all recovery clients finished"