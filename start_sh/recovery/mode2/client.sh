#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BANK_BASE_DIR="/home/chengyuxuan/openpi/recovery/mode1"
OUT_BASE_DIR="/home/chengyuxuan/openpi/recovery/mode2"

LOG_DIR="$OUT_BASE_DIR/logs/client"
mkdir -p "$LOG_DIR"

run_one_task() {
  local port_w4a8="$1"
  local tag="$2"
  local task_id="$3"
  local port_w4a16="$4"

  local output_name="oracle_labels_${tag}.jsonl"

  {
    echo "===== ${tag}: recovery label sweep ====="
    echo "Task id: ${task_id}"
    echo "W4A8 port: ${port_w4a8}"
    echo "W4A16 port: ${port_w4a16}"
    echo "Bank base dir: ${BANK_BASE_DIR}"
    echo "Output base dir: ${OUT_BASE_DIR}"
    echo "Output: ${output_name}"
    echo "Log time: $(date)"
    echo

    rm -f "$OUT_BASE_DIR/$output_name"

    uv run python recovery/mode2_recover.py \
      --args.host 0.0.0.0 \
      --args.port-w4a8 "$port_w4a8" \
      --args.port-w4a16 "$port_w4a16" \
      --args.task-ids "$task_id" \
      --args.episode-start 0 \
      --args.episode-end 10 \
      --args.bank-base-dir "$BANK_BASE_DIR" \
      --args.base-dir "$OUT_BASE_DIR" \
      --args.output-name "$output_name"

    echo
    echo "===== ${tag}: finished at $(date) ====="
  } > "$LOG_DIR/${tag}.log" 2>&1 &
}

run_one_task 8000 task8 8 8001
# run_one_task 8002 task9 9 8003

ps -ef | grep "recovery/mode2_recover.py" | grep -v grep || true