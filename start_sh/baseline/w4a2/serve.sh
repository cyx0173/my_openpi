#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/experiments/baseline/w4a2"
LOG_DIR="$BASE_DIR/logs/serve"
mkdir -p "$LOG_DIR"

start_server() {
  local port="$1"
  local gpu="$2"
  local tag="$3"
  local mode="1"  # 固定 quant_mode 为 w4a2

  CUDA_VISIBLE_DEVICES="${gpu}" \
  OPENPI_QUANT_MODE="${mode}" \
  python scripts/serve_policy.py --port "${port}" --quantize \
    > "${LOG_DIR}/${tag}.log" 2>&1 &
}

start_server 8001 1 "device1"
ps -ef | grep "scripts/serve_policy.py" | grep -v grep || true