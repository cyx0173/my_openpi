#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi

BASE_DIR="/home/chengyuxuan/openpi/active_quant/mode1_data_select"
LOG_DIR="$BASE_DIR/logs/client"
mkdir -p "$LOG_DIR"

CLIENT_SCRIPT="active_quant/mode1_data_select.py"

TASK_SUITE_NAME="libero_10"
REPLAN_STEPS=5
SEED=7

# ------------------------------------------------------------
# LIBERO_10 task split.
#
# 注意：代码里的 task_id 是 0-based。
#
# 人类说的任务 1,3,5,7,9  -> task_id 0,2,4,6,8
# 人类说的任务 2,4,6,8,10 -> task_id 1,3,5,7,9
# ------------------------------------------------------------
TASK_IDS_GROUP_A="0,2,4,6,8"
TASK_IDS_GROUP_B="1,3,5,7,9"

NUM_TRIALS_PER_TASK=1
EPISODE_START=0
EPISODE_END=-1

DEBUG_NOISE_BASE_SEED=123
DEBUG_NOISE_HORIZON=10
DEBUG_NOISE_DIM=32

# Group A ports: GPU 0-3, ports 8000-8003
PORT_A_FP16=8000
PORT_A_W4A4=8001
PORT_A_W4A8=8002
PORT_A_W4A16=8003

# Group B ports: GPU 4-7, ports 8010-8013
PORT_B_FP16=8010
PORT_B_W4A4=8011
PORT_B_W4A8=8012
PORT_B_W4A16=8013

WORKER_ID_A=0
WORKER_ID_B=1

echo "[INFO] Start LIBERO_10 checkpoint-bank collection by task split"
echo "[INFO] BASE_DIR=${BASE_DIR}"
echo "[INFO] CLIENT_SCRIPT=${CLIENT_SCRIPT}"
echo "[INFO] Group A tasks: ${TASK_IDS_GROUP_A}  # human tasks 1,3,5,7,9"
echo "[INFO] Group B tasks: ${TASK_IDS_GROUP_B}  # human tasks 2,4,6,8,10"
echo "[INFO] num_trials_per_task=${NUM_TRIALS_PER_TASK}"
echo "[INFO] debug_noise_base_seed=${DEBUG_NOISE_BASE_SEED}"
echo

echo "[INFO] Launch Group A client: EGL=0, ports 8000-8003"

uv run python "$CLIENT_SCRIPT" \
  --args.host 127.0.0.1 \
  --args.port-fp16 "$PORT_A_FP16" \
  --args.port-w4a4 "$PORT_A_W4A4" \
  --args.port-w4a8 "$PORT_A_W4A8" \
  --args.port-w4a16 "$PORT_A_W4A16" \
  --args.task-suite-name "$TASK_SUITE_NAME" \
  --args.task-ids "$TASK_IDS_GROUP_A" \
  --args.episode-start "$EPISODE_START" \
  --args.episode-end "$EPISODE_END" \
  --args.num-trials-per-task "$NUM_TRIALS_PER_TASK" \
  --args.worker-id "$WORKER_ID_A" \
  --args.seed "$SEED" \
  --args.replan-steps "$REPLAN_STEPS" \
  --args.use-debug-noise \
  --args.debug-noise-base-seed "$DEBUG_NOISE_BASE_SEED" \
  --args.debug-noise-horizon "$DEBUG_NOISE_HORIZON" \
  --args.debug-noise-dim "$DEBUG_NOISE_DIM" \
  --args.base-dir "$BASE_DIR" \
  > "$LOG_DIR/client_libero10_groupA_tasks_13579.log" 2>&1 &

PID_A=$!

echo "[INFO] Launch Group B client: EGL=4, ports 8010-8013"

uv run python "$CLIENT_SCRIPT" \
  --args.host 127.0.0.1 \
  --args.port-fp16 "$PORT_B_FP16" \
  --args.port-w4a4 "$PORT_B_W4A4" \
  --args.port-w4a8 "$PORT_B_W4A8" \
  --args.port-w4a16 "$PORT_B_W4A16" \
  --args.task-suite-name "$TASK_SUITE_NAME" \
  --args.task-ids "$TASK_IDS_GROUP_B" \
  --args.episode-start "$EPISODE_START" \
  --args.episode-end "$EPISODE_END" \
  --args.num-trials-per-task "$NUM_TRIALS_PER_TASK" \
  --args.worker-id "$WORKER_ID_B" \
  --args.seed "$SEED" \
  --args.replan-steps "$REPLAN_STEPS" \
  --args.use-debug-noise \
  --args.debug-noise-base-seed "$DEBUG_NOISE_BASE_SEED" \
  --args.debug-noise-horizon "$DEBUG_NOISE_HORIZON" \
  --args.debug-noise-dim "$DEBUG_NOISE_DIM" \
  --args.base-dir "$BASE_DIR" \
  > "$LOG_DIR/client_libero10_groupB_tasks_246810.log" 2>&1 &

PID_B=$!

echo
echo "[INFO] Clients launched in background."
echo "[INFO] PIDs:"
echo "  Group A client: $PID_A"
echo "  Group B client: $PID_B"
echo
echo "[INFO] Logs:"
echo "  tail -f $LOG_DIR/client_libero10_groupA_tasks_13579.log"
echo "  tail -f $LOG_DIR/client_libero10_groupB_tasks_246810.log"
echo
echo "[INFO] Output dirs:"
echo "  $BASE_DIR/checkpoint_bank/data"
echo "  $BASE_DIR/checkpoint_bank/obs"
echo "  $BASE_DIR/checkpoint_bank/noise"
echo "  $BASE_DIR/checkpoint_bank/videos"
echo
echo "[INFO] Summary files:"
echo "  $BASE_DIR/checkpoint_bank/data/chunk_bank_summary_w0.json"
echo "  $BASE_DIR/checkpoint_bank/data/chunk_bank_summary_w1.json"