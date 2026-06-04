#!/usr/bin/env bash
set -uo pipefail

OPENPI_ROOT="/home/chengyuxuan/openpi"
cd "$OPENPI_ROOT"

TASK_JSON="${1:-/home/chengyuxuan/openpi/experiments/smooth/mode7/task.json}"

CLIENT="experiments/mode7_recovery.py"

RUN_ROOT="$HOME/openpi/experiments/smooth/recovery/mode7"
JOB_DIR="$RUN_ROOT/jobs"
OUT_DIR="$RUN_ROOT/recovery_schedules"
LOG_DIR="$RUN_ROOT/logs/client"

mkdir -p "$JOB_DIR" "$OUT_DIR" "$LOG_DIR"

MASTER_LOG="$LOG_DIR/master.log"
PID_FILE="$LOG_DIR/master.pid"

# 默认后台运行。
# 前台调试：
#   RECOVERY_FOREGROUND=1 bash start_sh/quant/recovery/run_mode7_recovery_daemon.sh
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

  echo "[DAEMON] starting mode7 recovery runner..."
  nohup env RECOVERY_DAEMONIZED=1 bash "$SCRIPT_PATH" "$TASK_JSON" \
    > "$MASTER_LOG" 2>&1 &

  pid=$!
  echo "$pid" > "$PID_FILE"

  echo "[DAEMON] started pid=$pid"
  echo "[DAEMON] pid_file=$PID_FILE"
  echo "[DAEMON] log=$MASTER_LOG"
  exit 0
fi

source start_sh/reset_path.sh

echo "[MASTER] started at $(date)"
echo "[MASTER] pid=$$"
echo "[MASTER] task_json=$TASK_JSON"
echo "[MASTER] run_root=$RUN_ROOT"
echo "$$" > "$PID_FILE"

export OPENPI_RECOVERY_ROOT="$RUN_ROOT"
export OPENPI_RECOVERY_SCHEDULE_ROOT="$OUT_DIR"

SLOTS=(0 1 2 3 4 5 6 7)

is_case_done() {
  local case_id="$1"

  python - "$OUT_DIR" "$case_id" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
case_id = sys.argv[2]
case_root = out_dir / case_id

candidates = [
    case_root / "recovery_schedule.json",
    case_root / "schedule.json",
    case_root / "recovery_schedule_recovered.json",
]

schedule_path = None
for p in candidates:
    if p.exists() and p.stat().st_size > 0:
        schedule_path = p
        break

if schedule_path is None:
    sys.exit(1)

try:
    j = json.load(open(schedule_path, "r", encoding="utf-8"))
except Exception:
    sys.exit(1)

success = (
    j.get("final_success") is True
    or j.get("success") is True
    or (j.get("best_result") or {}).get("success") is True
)

sys.exit(0 if success else 1)
PY
}

run_case() {
  local slot="$1"
  local id="$2"
  local case_id="$3"
  local job_json="$4"

  local port_a4=$((8000 + slot))
  local port_a8=$((8008 + slot))
  local port_a16=$((8016 + slot))

  echo "[LAUNCH] id=${id} case=${case_id} slot=${slot} ports=${port_a4},${port_a8},${port_a16}"

  uv run python "$CLIENT" \
    --job "$job_json" \
    --out "$OUT_DIR" \
    --policy-host "127.0.0.1" \
    --port-a4 "$port_a4" \
    --port-a8 "$port_a8" \
    --port-a16 "$port_a16" \
    --suffix-probe-every 4 \
    --save-trial-summary \
    --save-final-actions \
    --save-video \
    > "$LOG_DIR/id${id}_${case_id}.log" 2>&1 &
}

slot_i=0

while IFS=$'\t' read -r id case_id job_json; do
  [ -f "$job_json" ] || continue

  if is_case_done "$case_id"; then
    echo "[SKIP] id=${id} case=${case_id} already successful"
    continue
  fi

  run_case "${SLOTS[$slot_i]}" "$id" "$case_id" "$job_json"

  slot_i=$((slot_i + 1))

  if [ "$slot_i" -ge "${#SLOTS[@]}" ]; then
    wait || echo "[WARN] one or more jobs in this batch failed; continue resume loop"
    slot_i=0
  fi
done < <(
  python - "$TASK_JSON" "$JOB_DIR" <<'PY'
import json
import re
import sys
from pathlib import Path

task_json = Path(sys.argv[1])
job_dir = Path(sys.argv[2])
job_dir.mkdir(parents=True, exist_ok=True)

obj = json.load(open(task_json, "r", encoding="utf-8"))

if isinstance(obj, list):
    tasks = obj
elif isinstance(obj, dict):
    tasks = obj.get("jobs") or obj.get("tasks") or obj.get("data") or []
else:
    tasks = []

def get_case_id(x):
    if "case_id" in x:
        return str(x["case_id"])

    task_id = x.get("task_id")
    ep = x.get("episode_idx", x.get("ep_id", x.get("episode_id")))

    if task_id is not None and ep is not None:
        return f"task{int(task_id):02d}_ep{int(ep):03d}"

    text = json.dumps(x, ensure_ascii=False)
    m = re.search(r"task(\d+)_ep(\d+)", text)
    if m:
        return f"task{int(m.group(1)):02d}_ep{int(m.group(2)):03d}"

    raise KeyError(f"Cannot infer case_id from job: {x}")

def key_fn(x):
    cid = get_case_id(x)
    m = re.search(r"task(\d+)_ep(\d+)", cid)
    if not m:
        return (999, 999999, cid)
    return (int(m.group(1)), int(m.group(2)), cid)

tasks = sorted(tasks, key=key_fn)

for new_id, x in enumerate(tasks):
    x = dict(x)
    case_id = get_case_id(x)
    x.setdefault("case_id", case_id)

    safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)
    job_path = job_dir / f"id{new_id:04d}_{safe_case}.json"

    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(x, f, indent=2, ensure_ascii=False)

    print(f"{new_id}\t{case_id}\t{job_path}")
PY
)

wait || echo "[WARN] one or more final jobs failed"

echo "[DONE] all mode7 recovery jobs finished at $(date)"