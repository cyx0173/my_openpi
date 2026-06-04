#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

ROOT="/share/chengyuxuan-local/openpi/recovery_data_selector"

TASK_JSON="${1:-$ROOT/task_schedule.json}"

CLIENT="experiments/mode6_data_selector.py"
LOG_DIR="/home/chengyuxuan/openpi/experiments/selector_dataset/logs/client"
mkdir -p "$LOG_DIR"

export OPENPI_SELECTOR_DATASET_ROOT="$ROOT"

SLOTS=(0 1 2 3 4 5 6 7)

is_case_done() {
  local case_id="$1"
  local schedule="$2"

  python - "$ROOT" "$case_id" "$schedule" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
case_id = sys.argv[2]
schedule_path = Path(sys.argv[3])

case_root = root / "cases" / case_id

# case 目录不存在：没完成
if not case_root.exists():
    sys.exit(1)

# summary 不存在：大概率半成品
if not (case_root / "state_summary.json").exists():
    sys.exit(1)

try:
    schedule = json.load(open(schedule_path, "r", encoding="utf-8"))
except Exception:
    sys.exit(1)

rows = schedule.get("chunk_schedule") or schedule.get("selector_chunk_schedule") or []
if not rows:
    sys.exit(1)

for row in rows:
    k = int(row["chunk_idx"])

    required = [
        case_root / "state" / f"chunk{k:04d}_state.npz",
        case_root / "mid" / "w4a4" / f"chunk{k:04d}.npz",
        case_root / "mid" / "w4a8" / f"chunk{k:04d}.npz",
        case_root / "mid" / "w4a16" / f"chunk{k:04d}.npz",
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

  local port_a4=$((8000 + slot))
  local port_a8=$((8008 + slot))
  local port_a16=$((8016 + slot))
  local ports="${port_a4},${port_a8},${port_a16}"

  echo "[LAUNCH] id=${id} case=${case_id} slot=${slot} ports=${ports}"

  uv run python "$CLIENT" "$schedule" --ports "$ports" \
    > "$LOG_DIR/id${id}_${case_id}.log" 2>&1 &
}

slot_i=0

while IFS=$'\t' read -r id case_id schedule; do
  [ -f "$schedule" ] || continue

  # 关键：先检查是否已经完成。
  # 完成则跳过，不占用 slot。
  if is_case_done "$case_id" "$schedule"; then
    echo "[SKIP] id=${id} case=${case_id} already complete"
    continue
  fi

  # 没完成才分配任务
  run_case "${SLOTS[$slot_i]}" "$id" "$case_id" "$schedule"

  slot_i=$((slot_i + 1))

  if [ "$slot_i" -ge "${#SLOTS[@]}" ]; then
    wait
    slot_i=0
  fi
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

wait

echo "[DONE] all mode6 selector collection jobs finished"