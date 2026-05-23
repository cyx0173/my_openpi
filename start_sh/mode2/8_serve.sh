#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi

BASE_DIR="/home/chengyuxuan/openpi/active_quant/mode2_recovery_select"
LOG_DIR="$BASE_DIR/logs/serve"
mkdir -p "$LOG_DIR"

echo "[INFO] Kill old ports 8000-8007"
for p in 8000 8001 8002 8003 8004 8005 8006 8007; do
  fuser -k "${p}/tcp" || true
done

echo "[INFO] Start 8 clean FP16 servers"

for i in 0 1 2 3 4 5 6 7; do
  port=$((8000 + i))
  echo "[INFO] Start clean FP16 server gpu=$i port=$port"

  env \
    -u OPENPI_QUANT_MODE \
    -u OPENPI_DUQUANT_STAGED \
    -u OPENPI_DUQUANT_INCLUDE \
    -u OPENPI_DUQUANT_EXCLUDE \
    -u OPENPI_DUQUANT_WBITS \
    -u OPENPI_DUQUANT_ABITS \
    -u OPENPI_DUQUANT_BLOCK \
    -u OPENPI_DUQUANT_BLOCK_OUT \
    -u OPENPI_DUQUANT_PERM \
    -u OPENPI_ATM_ENABLE \
    -u OPENPI_ATM_ALPHA_PATH \
    -u OPENPI_ATM_SCOPE \
    -u OPENPI_ATM_ALPHA_ONES \
    -u OPENPI_ATM_ALPHA_HALF \
    -u OPENPI_ATM_CAPTURE_TAG \
    -u OPENPI_ATM_CAPTURE_PATH \
    -u OPENPI_OHB_ENABLE \
    -u OPENPI_OHB_BETA_PATH \
    -u OPENPI_OHB_SCOPE \
    -u OPENPI_OHB_BETA_ONES \
    -u OPENPI_OHB_BETA_CONSTANT \
    -u OPENPI_OHB_CAPTURE_TAG \
    -u OPENPI_OHB_CAPTURE_PATH \
    CUDA_VISIBLE_DEVICES="$i" python scripts/serve_policy.py --port "$port" \
    > "$LOG_DIR/fp16_gpu${i}_port${port}.log" 2>&1 &
done

echo
echo "[INFO] 8 clean FP16 servers launched."
echo "[INFO] Logs:"
for i in 0 1 2 3 4 5 6 7; do
  port=$((8000 + i))
  echo "  tail -f $LOG_DIR/fp16_gpu${i}_port${port}.log"
done