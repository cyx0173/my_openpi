#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/experiments/baseline/320steps"
LOG_DIR="$BASE_DIR/logs/client"
mkdir -p "$LOG_DIR"

run_one_task() {
  local port="$1"
  local tag="$2"
  local task_id="$3"
  local task_name="$4"

  {

    uv run python experiments/mode1_baseline.py \
      --args.port-w4a4 "$port" \
      --args.task_ids "$task_id" \
      --args.episode-start 0 \
      --args.episode-end 20 \
      --args.base-dir "$BASE_DIR/$tag"

  } > "$LOG_DIR/${tag}_${task_name}.log" 2>&1 &
}

run_one_task 8002 w4a4 8 "put_both_moka_pots_on_the_stove"
run_one_task 8003 w4a8 8 "put_both_moka_pots_on_the_stove"
run_one_task 8004 w4a16 8 "put_both_moka_pots_on_the_stove"
# cmd +
run_one_task 8005 w4a4 9 "put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
run_one_task 8006 w4a8 9 "put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
run_one_task 8007 w4a16 9 "put_the_yellow_and_white_mug_in_the_microwave_and_close_it"

ps -ef | grep "experiments/mode1_baseline.py" | grep -v grep || true