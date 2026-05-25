#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/recovery/mode1"
LOG_DIR="$BASE_DIR/logs/serve"
mkdir -p "$LOG_DIR"

LAYOUT="naive_vlm_action_selective"

start_server() {
  local port="$1"
  local gpu="$2"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  OPENPI_QUANT_MODE=1 \
  OPENPI_DUQUANT_LAYOUT="${LAYOUT}" \
  python scripts/serve_policy.py --port "${port}" --quantize \
    > "${LOG_DIR}/w4a4_port${port}_gpu${gpu}.log" 2>&1 &
}

start_server 8000 0
start_server 8001 1
start_server 8002 2

ps -ef | grep "scripts/serve_policy.py" | grep -v grep