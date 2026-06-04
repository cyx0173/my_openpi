#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/experiments/smooth/recovery"
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
for port in 6 7  ; do
  port_1=$((8000 + port ))
  port_2=$((8000 + port + 8))
  port_3=$((8000 + port + 16))
  start_server "$port_1" "$port" "1" "w4a4"
  start_server "$port_2" "$port" "2" "w4a8"
  start_server "$port_3" "$port" "3" "w4a16"

done

ps -ef | grep "scripts/serve_policy.py" | grep -v grep || true

#lsof -ti:8000 | xargs -r kill