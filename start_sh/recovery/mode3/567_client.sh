#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/recovery/mode3/567_accrute"
LOG_DIR="$BASE_DIR/logs/client"
mkdir -p "$LOG_DIR"

run_one_task() {
  local port="$1"
  local tag="$2"

  {
    echo "===== ${tag}: build action bank ====="
    echo "Port: ${port}"
    echo "Base dir: ${BASE_DIR}/${tag}"
    echo "Log time: $(date)"
    echo

    uv run python recovery/mode3_accrute.py \
      --args.port-w4a4 "$port" \
      --args.task-ids "9" \
      --args.episode-start 0 \
      --args.episode-end 10 \
      --args.base-dir "$BASE_DIR/$tag"

    echo
    echo "===== ${tag}: finished at $(date) ====="
  } > "$LOG_DIR/${tag}.log" 2>&1 &
}

run_one_task 8005 w4a4
run_one_task 8006 w4a8
run_one_task 8007 w4a16

echo "Launched w4a4/w4a8/w4a16 clients in parallel."
echo "Logs:"
echo "  $LOG_DIR/w4a4.log"
echo "  $LOG_DIR/w4a8.log"
echo "  $LOG_DIR/w4a16.log"

ps -ef | grep "recovery/mode3_data.py" | grep -v grep || true