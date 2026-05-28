#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/experiments/baseline/fp16"
LOG_DIR="$BASE_DIR/logs/serve"
mkdir -p "$LOG_DIR"

start_server() {
  local port="$1"
  local gpu="$2"
  local tag="$3"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  python scripts/serve_policy.py --port "${port}" \
    > "${LOG_DIR}/${tag}.log" 2>&1 &
}

start_server 8000 0 "device0"
ps -ef | grep "scripts/serve_policy.py" | grep -v grep || true