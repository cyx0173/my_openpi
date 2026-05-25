#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/experiments/mode1"
LOG_DIR="$BASE_DIR/logs/serve"
mkdir -p "$LOG_DIR"

start_server() {
  local port="$1"
  local gpu="$2"
  local mode="$3"  # 新增：接收第三个参数作为 quant_mode
  local tag="$4"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  OPENPI_QUANT_MODE="${mode}" \
  python scripts/serve_policy.py --port "${port}" --quantize \
    > "${LOG_DIR}/device_${gpu}.log" 2>&1 &
}

#start_server 8002 2 1 w4a4
#start_server 8003 3 2 w4a8
#start_server 8004 4 3 w4a16
#320steps
start_server 8000 0 1 w4a4
start_server 8001 1 1 w4a4
ps -ef | grep "scripts/serve_policy.py" | grep -v grep || true