#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

ROOT="/home/chengyuxuan/openpi/experiments/selector_test_data"
LOG_DIR="$ROOT/logs/client"
mkdir -p "$LOG_DIR"

# 默认精度目录名。也可以外部覆盖：
#   PRECISION=w4a4 bash xxx.sh
PRECISION="${PRECISION:-w4a4}"

# 端口池：8040 - 8071，共 32 个 port
PORT_START=8040
PORT_END=8071

# episode 配置
EPISODE_START=0
EPISODE_END=50      # 左闭右开：跑 0-49
CHUNK_SIZE=5        # 每个 job 跑 5 个 eps

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
  local episode_start="$3"
  local episode_end="$4"

  local task_name
  task_name="$(get_task_name "$task_id")"

  local out_dir="$ROOT/action_chunk/$task_name/$PRECISION"
  local log_file="$LOG_DIR/task${task_id}_ep${episode_start}_${episode_end}_port${port}.log"

  mkdir -p "$out_dir"

  uv run python experiments/mode1_baseline.py \
    --args.port_w4a4 "$port" \
    --args.task-ids "$task_id" \
    --args.episode-start "$episode_start" \
    --args.episode-end "$episode_end" \
    --args.base_dir "$out_dir" \
    > "$log_file" 2>&1 &

  echo "$!"
}

# 生成端口池
free_ports=()
for ((p=PORT_START; p<=PORT_END; p++)); do
  free_ports+=("$p")
done

# 生成任务队列：task0 ep0-5, task0 ep5-10, ...
job_task_ids=()
job_ep_starts=()
job_ep_ends=()

for task_id in 0 1 2 3 4 5 6 7 8 9; do
  for ((ep=EPISODE_START; ep<EPISODE_END; ep+=CHUNK_SIZE)); do
    ep_end=$((ep + CHUNK_SIZE))
    if (( ep_end > EPISODE_END )); then
      ep_end="$EPISODE_END"
    fi

    job_task_ids+=("$task_id")
    job_ep_starts+=("$ep")
    job_ep_ends+=("$ep_end")
  done
done

total_jobs="${#job_task_ids[@]}"
next_job_idx=0
active_count=0
failed_count=0

declare -A PID_TO_PORT
declare -A PID_TO_META

cleanup() {
  echo
  echo "[CLEANUP] killing active clients..."
  for pid in "${!PID_TO_PORT[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

launch_next_job_on_port() {
  local port="$1"

  if (( next_job_idx >= total_jobs )); then
    free_ports+=("$port")
    return
  fi

  local task_id="${job_task_ids[$next_job_idx]}"
  local ep_start="${job_ep_starts[$next_job_idx]}"
  local ep_end="${job_ep_ends[$next_job_idx]}"

  local pid
  pid="$(run_trace "$port" "$task_id" "$ep_start" "$ep_end")"

  PID_TO_PORT["$pid"]="$port"
  PID_TO_META["$pid"]="task=${task_id} eps=${ep_start}-$((ep_end - 1)) port=${port}"

  echo "[LAUNCH] pid=${pid} ${PID_TO_META[$pid]}"

  next_job_idx=$((next_job_idx + 1))
  active_count=$((active_count + 1))
}

# 先把 32 个端口尽量填满
while (( ${#free_ports[@]} > 0 && next_job_idx < total_jobs )); do
  port="${free_ports[0]}"
  free_ports=("${free_ports[@]:1}")
  launch_next_job_on_port "$port"
done

# 动态调度：谁先结束，谁立刻拿下一个 job
while (( active_count > 0 )); do
  sleep 2

  for pid in "${!PID_TO_PORT[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi

    port="${PID_TO_PORT[$pid]}"
    meta="${PID_TO_META[$pid]}"

    set +e
    wait "$pid"
    status="$?"
    set -e

    unset PID_TO_PORT["$pid"]
    unset PID_TO_META["$pid"]
    active_count=$((active_count - 1))

    if (( status == 0 )); then
      echo "[DONE] pid=${pid} ${meta}"
    else
      echo "[FAIL] pid=${pid} status=${status} ${meta}"
      failed_count=$((failed_count + 1))
    fi

    # 这个 port 立刻继续拿下一个 job
    launch_next_job_on_port "$port"
  done
done

echo
echo "[ALL DONE] total_jobs=${total_jobs} failed_jobs=${failed_count}"
echo "[CHECK]"
ps -ef | grep "experiments/mode1_baseline.py" | grep -v grep || true