#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/recovery/mode3/234_accrute"
LOG_DIR="$BASE_DIR/logs/serve"
mkdir -p "$LOG_DIR"

LAYOUT="naive_vlm_action_selective"

start_server() {
  local port="$1"
  local gpu="$2"
  local mode="$3"  # 新增：接收第三个参数作为 quant_mode

  CUDA_VISIBLE_DEVICES="${gpu}" \
  OPENPI_QUANT_MODE="${mode}" \
  OPENPI_DUQUANT_STAGED=1 \
  OPENPI_DUQUANT_LAYOUT="${LAYOUT}" \
  OPENPI_DISABLE_ATM_OHB=1 \
  python scripts/serve_policy.py --port "${port}" --quantize \
    > "${LOG_DIR}/w4a4_port${port}_gpu${gpu}_mode${mode}.log" 2>&1 &
}

# 依次传入：端口号、GPU ID、QUANT_MODE
# 对应关系：8002 -> mode 1 | 8003 -> mode 2 | 8004 -> mode 3
start_server 8002 2 1
start_server 8003 3 2
start_server 8004 4 3

echo "Launched servers on ports 8002 (mode1), 8003 (mode2), 8004 (mode3)."

ps -ef | grep "scripts/serve_policy.py" | grep -v grep || true