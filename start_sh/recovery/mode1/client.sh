 #!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/recovery/mode1"
LOG_DIR="$BASE_DIR/logs/client"
mkdir -p "$LOG_DIR"

run_one_task() {
  local task_id="$1"
  local task_name="$2"
  local port="$3"  # 新增：接收第三个参数作为端口号

  {
    echo "===== task ${task_id} ${task_name}: build W4A4 action bank ====="

    uv run python recovery/mode1_data.py \
      --args.port-w4a4 "$port" \
      --args.task-ids "$task_id" \
      --args.episode-start 0 \
      --args.episode-end 10 \
      --args.base-dir "$BASE_DIR"

    echo "===== task ${task_id} ${task_name}: recovery label sweep ====="
  } > "$LOG_DIR/task${task_id}_${task_name}.log" 2>&1 &
}

# 修改：在调用时传入对应的端口号 8000, 8001, 8002
run_one_task 2 "turn_on_stove_and_put_moka_pot_on_it" 8000
run_one_task 8 "put_both_moka_pots_on_stove" 8001
run_one_task 9 "put_yellow_white_mug_in_microwave_and_close_it" 8002

echo "Launched task2/task8/task9 in parallel."
echo "Logs:"
echo "  $LOG_DIR/task2_turn_on_stove_and_put_moka_pot_on_it.log"
echo "  $LOG_DIR/task8_put_both_moka_pots_on_stove.log"
echo "  $LOG_DIR/task9_put_yellow_white_mug_in_microwave_and_close_it.log"

ps -ef | grep "recovery/mode1_data.py" | grep -v grep || true