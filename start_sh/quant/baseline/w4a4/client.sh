#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

ROOT="/home/chengyuxuan/openpi/experiments/baseline/trace_2"
LOG_DIR="$ROOT/logs/client"
mkdir -p "$LOG_DIR"

get_task_name() {
  case "$1" in
    "0") echo "put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket" ;;
    "1") echo "put_both_the_cream_cheese_box_and_the_butter_in_the_basket" ;;
    "2") echo "turn_on_the_stove_and_put_the_moka_pot_on_it" ;;
    "3") echo "put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it" ;;
    "4") echo "put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate" ;;
    "5") echo "pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy" ;;
    "6") echo "put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate" ;;
    "7") echo "put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket" ;;
    "8") echo "put_both_moka_pots_on_the_stove" ;;
    "9") echo "put_the_yellow_and_white_mug_in_the_microwave_and_close_it" ;;
    *) echo "unknown_task" ;;
  esac
}

run_trace() {
  local port="$1"
  local task_id="$2"

  local precision="w4a4"
  local task_name
  task_name="$(get_task_name "$task_id")"

  local out_dir="$ROOT/action_chunk/$task_name/$precision"
  local log_file="$LOG_DIR/${precision}/task${task_id}.log"
  mkdir -p "$LOG_DIR/${precision}"
  mkdir -p "$out_dir"

  uv run python experiments/mode3_collector.py \
    --args.port-w4a4 "$port" \
    --args.precision "$precision" \
    --args.task-ids "$task_id" \
    --args.episode-start 0 \
    --args.episode-end 50 \
    --args.base-dir "$out_dir" \
    --args.skip-existing \
    > "$log_file" 2>&1 &

  echo "$!"
}

pids=()

for task_id in 0 1 2 3 4 5 6 7 8 9; do
  port=$((8000 + task_id))
  pids+=("$(run_trace "$port" "$task_id")")
done

ps -ef | grep "experiments/mode3_collector.py" | grep -v grep || true

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

exit "$fail"