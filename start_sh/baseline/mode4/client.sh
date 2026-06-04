#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/experiments/mode5/trace"
LOG_DIR="$BASE_DIR/logs/client"
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

run_one_precision() {
  local port="$1"
  local tag="$2"
  local task_id="$3"

  local task_name
  task_name="$(get_task_name "$task_id")"

  local out_dir="$BASE_DIR/$task_name/$tag"
  local log_file="$LOG_DIR/${tag}_${task_name}.log"

  mkdir -p "$out_dir"

  {
    uv run python experiments/mode3_collector.py \
      --args.port-w4a4 "$port" \
      --args.precision "$tag" \
      --args.task-ids "$task_id" \
      --args.episode-start 0 \
      --args.episode-end 50 \
      --args.base-dir "$out_dir" \
      --args.skip-existing
  } > "$log_file" 2>&1 &

  echo "$!"
}

pids=()

# pids+=("$(run_one_precision 8026 w4a4 0)")
# pids+=("$(run_one_precision 8027 w4a4 1)")
# pids+=("$(run_one_precision 8018 w4a4 2)")
# pids+=("$(run_one_precision 8019 w4a4 3)")
# pids+=("$(run_one_precision 8020 w4a4 4)")
# pids+=("$(run_one_precision 8021 w4a4 5)")
# pids+=("$(run_one_precision 8022 w4a4 6)")
# pids+=("$(run_one_precision 8023 w4a4 7)")
# pids+=("$(run_one_precision 8024 w4a4 8)")
# pids+=("$(run_one_precision 8025 w4a4 9)")

# pids+=("$(run_one_precision 8000 w4a16 0)")
# pids+=("$(run_one_precision 8001 w4a16 1)")
# pids+=("$(run_one_precision 8002 w4a16 2)")
# pids+=("$(run_one_precision 8003 w4a16 3)")
# pids+=("$(run_one_precision 8004 w4a16 4)")
# pids+=("$(run_one_precision 8005 w4a16 5)")
# pids+=("$(run_one_precision 8006 w4a16 6)")
# pids+=("$(run_one_precision 8007 w4a16 7)")
pids+=("$(run_one_precision 8008 w4a16 8)")
pids+=("$(run_one_precision 8009 w4a16 9)")
ps -ef | grep "experiments/mode3_collector.py" | grep -v grep || true

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

exit "$fail"