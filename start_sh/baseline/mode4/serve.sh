#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/experiments/mode5"
LOG_DIR="$BASE_DIR/logs/serve"
mkdir -p "$LOG_DIR"

start_server() {
  local port="$1"
  local gpu="$2"
  local mode="$3"
  local tag="$4"

  echo "[START] port=${port} gpu=${gpu} mode=${mode} tag=${tag}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  OPENPI_QUANT_MODE="${mode}" \
  python scripts/serve_policy.py \
    --port "${port}" \
    --quantize \
    > "${LOG_DIR}/port${port}_${tag}_mode${mode}.log" 2>&1 &

  echo $!
}

# start_server 8000 0 1 "w4a4"
# start_server 8001 1 2 "w4a8"


# start_server 8002 2 3 "w4a16"
# start_server 8003 3 3 "w4a16"

# start_server 8000 0 3 "w4a16"
# start_server 8001 1 3 "w4a16"

# start_server 8002 2 3 "w4a16"
# start_server 8003 3 3 "w4a16"

# start_server 8004 4 4 "w4a16"
# start_server 8005 5 3 "w4a16"

# start_server 8006 6 3 "w4a16"
# start_server 8007 7 3 "w4a16"

start_server 8008 0 3 "w4a16"

start_server 8009 1 3 "w4a16"


ps -ef | grep "scripts/serve_policy.py" | grep -v grep || true

#lsof -ti:8000 | xargs -r kill