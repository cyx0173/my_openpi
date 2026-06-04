#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/experiments/smooth/baseline"
LOG_DIR="$BASE_DIR/logs/serve"
mkdir -p "$LOG_DIR"

start_server() {
  local port="$1"
  local gpu="$2"
  local mode="3"
  local tag="w4a16"
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

start_server 8020 0 
start_server 8021 1 
start_server 8022 2 
start_server 8023 3 

start_server 8024 4 
start_server 8025 5 
start_server 8026 6 
start_server 8027 7

start_server 8028 2
start_server 8029 3
# start_server 8010 0 
# start_server 8011 1 
# start_server 8012 2 
# start_server 8013 3 

# start_server 8014 4 
# start_server 8015 5 
# start_server 8016 6 
# start_server 8017 7

# start_server 8018 4
# start_server 8019 3
ps -ef | grep "scripts/serve_policy.py" | grep -v grep || true

#lsof -ti:8000 | xargs -r kill