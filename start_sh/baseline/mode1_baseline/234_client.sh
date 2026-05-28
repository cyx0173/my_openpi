#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi

BASE_DIR="/home/chengyuxuan/openpi/experiments/mode1/libero_spatial"
LOG_DIR="$BASE_DIR/logs/client"
mkdir -p "$LOG_DIR"
# libero_spatial
get_task_name() {
    case "$1" in
        "0") TASK_NAME="put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket" ;;
        "1") TASK_NAME="put_both_the_cream_cheese_box_and_the_butter_in_the_basket" ;;
        "2") TASK_NAME="turn_on_the_stove_and_put_the_moka_pot_on_it" ;;
        "3") TASK_NAME="put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it" ;;
        "4") TASK_NAME="put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate" ;;
        "5") TASK_NAME="pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy" ;;
        "6") TASK_NAME="put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate" ;;
        "7") TASK_NAME="put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket" ;;
        "8") TASK_NAME="put_both_moka_pots_on_the_stove" ;;
        "9") TASK_NAME="put_the_yellow_and_white_mug_in_the_microwave_and_close_it" ;;
        *) TASK_NAME="unknown_task" ;;
    esac
    echo "$TASK_NAME"
}
# get_task_name() {
#     case "$1" in
#         "0") TASK_NAME="put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket" ;;
#         "1") TASK_NAME="put_both_the_cream_cheese_box_and_the_butter_in_the_basket" ;;
#         "2") TASK_NAME="turn_on_the_stove_and_put_the_moka_pot_on_it" ;;
#         "3") TASK_NAME="put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it" ;;
#         "4") TASK_NAME="put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate" ;;
#         "5") TASK_NAME="pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy" ;;
#         "6") TASK_NAME="put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate" ;;
#         "7") TASK_NAME="put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket" ;;
#         "8") TASK_NAME="put_both_moka_pots_on_the_stove" ;;
#         "9") TASK_NAME="put_the_yellow_and_white_mug_in_the_microwave_and_close_it" ;;
#         *) TASK_NAME="unknown_task" ;;
#     esac
# }

# CURRENT_ID="8"
# get_task_name "$CURRENT_ID"

run_one_precision() {
  local port="$1"
  local tag="$2"
  local task_id="$3"
  local task_name="$4"

  local task_tag
  task_tag=$(printf "task%02d" "$task_id")

  {
    uv run python experiments/mode1_baseline.py \
      --args.port-w4a4 "$port" \
      --args.task_ids "$task_id" \
      --args.episode-start 0 \
      --args.episode-end 10 \
      --args.base-dir "$BASE_DIR/$task_name/$tag"
  } >> "$LOG_DIR/${tag}_${task_name}.log" 2>&1 &

  echo $!
}
# 顺次执行 9, 7, 5, 3, 1 任务（如果想正序，改成 1 3 5 7 9 即可）
for CURRENT_ID in 9 7 5 3 1; do

    get_task_name "$CURRENT_ID"

    run_one_precision 8002 w4a4   "$CURRENT_ID" "$TASK_NAME"
    run_one_precision 8003 w4a8   "$CURRENT_ID" "$TASK_NAME"
    run_one_precision 8004 w4a16  "$CURRENT_ID" "$TASK_NAME"

    wait
    
    echo "Task $CURRENT_ID 运行完毕！准备切入下一个任务。"
done


ps -ef | grep "experiments/mode1_baseline.py" | grep -v grep || true