#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/experiments/selector_test_data"
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
for idx in $(seq 64 71); do
  port=$((8000 + idx))
  gpu=$((idx % 8))
  start_server "$port" "$gpu" "w4a16"
done

# start_server 8033 1 "w4a16"
# start_server 8035 3 "w4a16"
# start_server 8036 4 "w4a16"
# start_server 8039 7 "w4a16"
ps -ef | grep "scripts/serve_policy.py" | grep -v grep || true

#lsof -ti:8000 | xargs -r kill