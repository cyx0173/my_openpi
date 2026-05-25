#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/experiments/mode1"
LOG_DIR="$BASE_DIR/logs/client"
mkdir -p "$LOG_DIR"

run_one_task() {
  local port="$1"
  local task_name="$2"
  local task_id="$3"
  {

    uv run python experiments/mode1_baseline.py \
      --args.port-w4a4 "$port" \
      --args.task_ids "$task_id" \
      --args.episode-start 0 \
      --args.episode-end 10 \
      --args.base-dir "$BASE_DIR/$task_name" \

  } > "$LOG_DIR/${task_name}.log" 2>&1 &
}

run_one_task 8000 "turn on the stove and put the moka pot on it" 2
run_one_task 8001 "put the yellow and white mug in the microwave and close it" 9
# cmd + / to toggle comments
# run_one_task 8005 w4a4
# run_one_task 8006 w4a8
# run_one_task 8007 w4a16

ps -ef | grep "experiments/mode1_baseline.py" | grep -v grep || true