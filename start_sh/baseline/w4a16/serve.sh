#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/lab_track/check"
LOG_DIR="$BASE_DIR/logs/serve"
mkdir -p "$LOG_DIR"

start_server() {
  local port="$1"
  local gpu="$2"
  local mode="$3"  # 新增：接收第三个参数作为 quant_mode
  local tag="$4"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  OPENPI_DUQUANT_ABLATION_BACKEND=fused \
  OPENPI_DUQUANT_ABLATION_ACT=native \
  OPENPI_QUANT_MODE="${mode}" \
  python scripts/serve_policy.py --port "${port}" --quantize \
    > "${LOG_DIR}/${tag}.log" 2>&1 &
}

start_server 8000 0 1 "W4A4_fused_native"
start_server 8001 1 2 "W4A8_fused_native"
start_server 8002 2 3 "W4A16_fused_native"

OPENPI_QUANT_MODE=3 \
OPENPI_DUQUANT_ABLATION_BACKEND=reference \
OPENPI_DUQUANT_ABLATION_ACT=native \
CUDA_VISIBLE_DEVICES=3 \
python scripts/serve_policy.py \
  --port 8003 \
  --quantize \
  > "${LOG_DIR}/W4A16_reference_native.log" 2>&1 &
#420steps
# start_server 8005 5 1 w4a4
# start_server 8006 6 2 w4a8
# start_server 8007 7 3 w4a16
ps -ef | grep "scripts/serve_policy.py" | grep -v grep || true