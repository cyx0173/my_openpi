#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/experiments/selector_dataset"
LOG_DIR="$BASE_DIR/logs/serve"
mkdir -p "$LOG_DIR"

start_server() {
  local port="$1"
  local gpu="$2"
  local tag="$3"
  mkdir -p "$LOG_DIR/tag_${tag}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  OPENPI_QUANT_BACKEND=smoothvla \
  python scripts/serve_policy.py \
    --port "${port}" \
    --quantize \
    > "${LOG_DIR}/tag_${tag}/port${port}.log" 2>&1 &

  echo $!
}
for idx in $(seq 0 7); do
# for idx in $(seq 8 15); do
  port=$((8000 + idx))
  gpu=$((idx % 8))
  start_server "$port" "$gpu" "w4a16"
done

ps -ef | grep "scripts/serve_policy.py" | grep -v grep || true

#lsof -ti:8000 | xargs -r kill