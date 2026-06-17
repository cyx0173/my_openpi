#!/usr/bin/env bash
set -uo pipefail

cd /home/chengyuxuan/openpi

TASK_JSON="${1:-/home/chengyuxuan/openpi/experiments/smooth/mode7/data.json}"
CLIENT="${CLIENT:-experiments/mode7_recovery.py}"
RUN_ROOT="${RUN_ROOT:-/home/chengyuxuan/openpi/experiments/smooth/mode7}"

OUT_DIR="$RUN_ROOT/recovery_schedules"
LOG_DIR="$RUN_ROOT/logs/client"
MASTER_LOG="$LOG_DIR/master.log"
PID_FILE="$LOG_DIR/master.pid"

# One client uses one serve. 32 serves means 32 slots by default:
# slot 0 -> BASE_PORT, slot 1 -> BASE_PORT+1, ...
NUM_SLOTS="${NUM_SLOTS:-40}"
BASE_PORT="${BASE_PORT:-8000}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

# 默认后台运行；前台调试：
#   RECOVERY_FOREGROUND=1 bash start_sh/quant/smooth/greedy/client.sh
if [ "${RECOVERY_FOREGROUND:-0}" != "1" ] && [ "${RECOVERY_DAEMONIZED:-0}" != "1" ]; then
  SCRIPT_PATH="$(readlink -f "$0")"

  if [ -f "$PID_FILE" ]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "${old_pid:-}" ] && kill -0 "$old_pid" 2>/dev/null; then
      echo "[DAEMON] already running pid=$old_pid"
      echo "[DAEMON] log=$MASTER_LOG"
      exit 0
    fi
  fi

  nohup env RECOVERY_DAEMONIZED=1 \
    CLIENT="$CLIENT" \
    RUN_ROOT="$RUN_ROOT" \
    FORCE_RERUN="${FORCE_RERUN:-0}" \
    SAVE_VIDEO="${SAVE_VIDEO:-1}" \
    NUM_SLOTS="$NUM_SLOTS" \
    BASE_PORT="$BASE_PORT" \
    MAX_STEPS="${MAX_STEPS:-520}" \
    BASE_SEED="${BASE_SEED:-0}" \
    STAGNATION_CHUNKS="${STAGNATION_CHUNKS:-10}" \
    bash "$SCRIPT_PATH" "$TASK_JSON" \
    > "$MASTER_LOG" 2>&1 &

  pid=$!
  echo "$pid" > "$PID_FILE"

  echo "[DAEMON] started pid=$pid"
  echo "[DAEMON] pid_file=$PID_FILE"
  echo "[DAEMON] log=$MASTER_LOG"
  exit 0
fi

source start_sh/reset_path.sh

echo "$$" > "$PID_FILE"

TOTAL="$(
python - "$TASK_JSON" <<'PY'
import json, sys
obj = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if isinstance(obj, list):
    tasks = obj
elif isinstance(obj, dict):
    tasks = obj.get("jobs") or obj.get("tasks") or obj.get("data") or []
else:
    tasks = []
print(len(tasks))
PY
)"

case_id_of() {
  local idx="$1"
  python - "$TASK_JSON" "$idx" <<'PY'
import json, re, sys
obj = json.load(open(sys.argv[1], "r", encoding="utf-8"))
idx = int(sys.argv[2])
tasks = obj if isinstance(obj, list) else (obj.get("jobs") or obj.get("tasks") or obj.get("data") or [])
x = tasks[idx]

if "case_id" in x:
    print(str(x["case_id"]))
    raise SystemExit

task_id = x.get("task_id")
ep = x.get("episode_idx", x.get("ep_id", x.get("episode_id")))
if task_id is not None and ep is not None:
    print(f"task{int(task_id):02d}_ep{int(ep):03d}")
    raise SystemExit

m = re.search(r"task(\d+)_ep(\d+)", json.dumps(x, ensure_ascii=False))
if m:
    print(f"task{int(m.group(1)):02d}_ep{int(m.group(2)):03d}")
    raise SystemExit

raise SystemExit(f"cannot infer case_id from index={idx}: {x}")
PY
}

is_done() {
  local case="$1"
  [ "${FORCE_RERUN:-0}" = "1" ] && return 1

  python - "$OUT_DIR/$case/recovery_schedule.json" <<'PY'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists() or p.stat().st_size == 0:
    sys.exit(1)

try:
    j = json.load(open(p, "r", encoding="utf-8"))
except Exception:
    sys.exit(1)

ok = j.get("final_success") is True

sys.exit(0 if ok else 1)
PY
}

launch_slot() {
  local slot="$1"
  local idx="$2"

  local case
  case="$(case_id_of "$idx")" || return 1

  if is_done "$case"; then
    echo "[SKIP] id=$idx case=$case"
    return 2
  fi

  local port=$((BASE_PORT + slot))
  local video_arg="--save-video"
  [ "${SAVE_VIDEO:-1}" = "0" ] && video_arg="--no-save-video"

  echo "[LAUNCH] id=$idx case=$case slot=$slot port=$port"

  (
    uv run python "$CLIENT" \
      --job <(python - "$TASK_JSON" "$idx" <<'PY'
import json, sys
obj = json.load(open(sys.argv[1], "r", encoding="utf-8"))
idx = int(sys.argv[2])
tasks = obj if isinstance(obj, list) else (obj.get("jobs") or obj.get("tasks") or obj.get("data") or [])
print(json.dumps(tasks[idx], ensure_ascii=False))
PY
) \
      --out "$OUT_DIR" \
      --policy-host 127.0.0.1 \
      --policy-port "$port" \
  ) > "$LOG_DIR/${case}.log" 2>&1 &

  local pid=$!
  PIDS[$slot]="$pid"
  CASES[$slot]="$case"
  IDS[$slot]="$idx"
  return 0
}

next_idx=0
active=0
finished=0
failed=0
skipped=0

declare -a PIDS
declare -a CASES
declare -a IDS

# 先填满 NUM_SLOTS 个 slot。
for ((slot=0; slot<NUM_SLOTS; slot++)); do
  while [ "$next_idx" -lt "$TOTAL" ]; do
    launch_slot "$slot" "$next_idx"
    rc=$?
    next_idx=$((next_idx + 1))

    if [ "$rc" -eq 0 ]; then
      active=$((active + 1))
      break
    elif [ "$rc" -eq 2 ]; then
      skipped=$((skipped + 1))
      continue
    else
      failed=$((failed + 1))
      continue
    fi
  done
done

# 动态分配：哪个 pid 完成，就把同一个 slot 立刻补上下一个任务。
while [ "$active" -gt 0 ]; do
  done_pid=""
  wait -n -p done_pid
  status=$?

  done_slot=""
  for ((slot=0; slot<NUM_SLOTS; slot++)); do
    if [ "${PIDS[$slot]:-}" = "$done_pid" ]; then
      done_slot="$slot"
      break
    fi
  done

  if [ -z "$done_slot" ]; then
    echo "[WARN] finished unknown pid=$done_pid status=$status"
    active=$((active - 1))
    continue
  fi

  case="${CASES[$done_slot]}"
  idx="${IDS[$done_slot]}"

  finished=$((finished + 1))
  active=$((active - 1))

  if [ "$status" -eq 0 ]; then
    echo "[DONE] id=$idx case=$case slot=$done_slot pid=$done_pid"
  else
    failed=$((failed + 1))
    echo "[FAIL] id=$idx case=$case slot=$done_slot pid=$done_pid rc=$status"
  fi

  unset PIDS[$done_slot]
  unset CASES[$done_slot]
  unset IDS[$done_slot]

  while [ "$next_idx" -lt "$TOTAL" ]; do
    launch_slot "$done_slot" "$next_idx"
    rc=$?
    next_idx=$((next_idx + 1))

    if [ "$rc" -eq 0 ]; then
      active=$((active + 1))
      break
    elif [ "$rc" -eq 2 ]; then
      skipped=$((skipped + 1))
      continue
    else
      failed=$((failed + 1))
      continue
    fi
  done
done

