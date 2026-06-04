#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/experiments/smooth/mode7"
LOG_DIR="$BASE_DIR/logs/serve"
mkdir -p "$LOG_DIR"

start_server() {
  local port="$1"
  local gpu="$2"
  local mode="$3"
  local tag="$4"
  mkdir -p "$LOG_DIR/tag_${tag}"
  echo "[START] port=${port} gpu=${gpu} mode=${mode} tag=${tag}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  OPENPI_QUANT_BACKEND=smoothvla \
  OPENPI_QUANT_MODE="${mode}" \
  python scripts/serve_policy.py \
    --port "${port}" \
    --quantize \
    > "${LOG_DIR}/tag_${tag}/mode${mode}_port${port}.log" 2>&1 &

  echo $!
}

start_server 8030 6 "1" "w4a4"
start_server 8031 7 "2" "w4a8"
start_server 8032 1 "3" "w4a16"

# start_server 8027 3 "1" "w4a4"
# start_server 8028 4 "2" "w4a8"
# start_server 8029 5 "3" "w4a16"


ps -ef | grep "scripts/serve_policy.py" | grep -v grep || true

#lsof -ti:8000 | xargs -r kill