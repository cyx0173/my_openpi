#!/usr/bin/env bash
set -euo pipefail

OPENPI_DIR="/home/chengyuxuan/openpi/lab_track/final/logs"
ATM_DIR="/home/chengyuxuan/openpi/lab_track/atm_1"

ATM_ALPHA_JSON="$ATM_DIR/pi05_atm_alpha_w4a8.json"
OHB_BETA_JSON="$ATM_DIR/pi05_ohb_beta_w4a8_atm.json"

cd /home/chengyuxuan/openpi
mkdir -p "$OPENPI_DIR"

echo "[INFO] Kill old servers on ports 8000/8001/8002/8003/8004"
fuser -k 8000/tcp || true
fuser -k 8001/tcp || true
fuser -k 8002/tcp || true
fuser -k 8003/tcp || true
fuser -k 8004/tcp || true

echo "[INFO] Clean old logs"
rm -f "$OPENPI_DIR/fp16.log"
rm -f "$OPENPI_DIR/w4a8.log"
rm -f "$OPENPI_DIR/w4a8_atm.log"
rm -f "$OPENPI_DIR/w4a8_atm_ohb.log"
rm -f "$OPENPI_DIR/fp16_ohb.log"

echo "[INFO] Start FP16 baseline server on port 8000"

env \
-u OPENPI_QUANT_MODE \
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
CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py --port 8000 \
> "$OPENPI_DIR/fp16.log" 2>&1 &

PID_FP16=$!


echo "[INFO] Start W4A8 pure quant server on port 8001"

env \
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
OPENPI_QUANT_MODE=2 \
CUDA_VISIBLE_DEVICES=1 python scripts/serve_policy.py --port 8001 --quantize \
> "$OPENPI_DIR/w4a8.log" 2>&1 &

PID_W4A8=$!


echo "[INFO] Start W4A8 + ATM server on port 8002"

env \
-u OPENPI_ATM_CAPTURE_TAG \
-u OPENPI_ATM_CAPTURE_PATH \
-u OPENPI_ATM_ALPHA_ONES \
-u OPENPI_ATM_ALPHA_HALF \
-u OPENPI_OHB_ENABLE \
-u OPENPI_OHB_BETA_PATH \
-u OPENPI_OHB_SCOPE \
-u OPENPI_OHB_BETA_ONES \
-u OPENPI_OHB_BETA_CONSTANT \
-u OPENPI_OHB_CAPTURE_TAG \
-u OPENPI_OHB_CAPTURE_PATH \
OPENPI_QUANT_MODE=2 \
OPENPI_ATM_ENABLE=1 \
OPENPI_ATM_SCOPE=gemma_expert \
OPENPI_ATM_ALPHA_PATH="$ATM_ALPHA_JSON" \
CUDA_VISIBLE_DEVICES=2 python scripts/serve_policy.py --port 8002 --quantize \
> "$OPENPI_DIR/w4a8_atm.log" 2>&1 &

PID_W4A8_ATM=$!


echo "[INFO] Start W4A8 + ATM + OHB server on port 8003"

env \
-u OPENPI_ATM_CAPTURE_TAG \
-u OPENPI_ATM_CAPTURE_PATH \
-u OPENPI_ATM_ALPHA_ONES \
-u OPENPI_ATM_ALPHA_HALF \
-u OPENPI_OHB_CAPTURE_TAG \
-u OPENPI_OHB_CAPTURE_PATH \
-u OPENPI_OHB_BETA_ONES \
-u OPENPI_OHB_BETA_CONSTANT \
OPENPI_QUANT_MODE=2 \
OPENPI_ATM_ENABLE=1 \
OPENPI_ATM_SCOPE=gemma_expert \
OPENPI_ATM_ALPHA_PATH="$ATM_ALPHA_JSON" \
OPENPI_OHB_ENABLE=1 \
OPENPI_OHB_SCOPE=gemma_expert \
OPENPI_OHB_BETA_PATH="$OHB_BETA_JSON" \
CUDA_VISIBLE_DEVICES=3 python scripts/serve_policy.py --port 8003 --quantize \
> "$OPENPI_DIR/w4a8_atm_ohb.log" 2>&1 &

PID_W4A8_ATM_OHB=$!


env \
-u OPENPI_ATM_ENABLE \
-u OPENPI_ATM_ALPHA_PATH \
-u OPENPI_ATM_SCOPE \
-u OPENPI_ATM_ALPHA_ONES \
-u OPENPI_ATM_ALPHA_HALF \
-u OPENPI_ATM_CAPTURE_TAG \
-u OPENPI_ATM_CAPTURE_PATH \
-u OPENPI_OHB_CAPTURE_TAG \
-u OPENPI_OHB_CAPTURE_PATH \
-u OPENPI_OHB_BETA_ONES \
-u OPENPI_OHB_BETA_CONSTANT \
OPENPI_QUANT_MODE=2 \
OPENPI_OHB_ENABLE=1 \
OPENPI_OHB_SCOPE=gemma_expert \
OPENPI_OHB_BETA_PATH="$OHB_BETA_JSON" \
CUDA_VISIBLE_DEVICES=4 python scripts/serve_policy.py --port 8004 --quantize \
> "$OPENPI_DIR/w4a8_ohb.log" 2>&1 &

PID_W4A8_OHB=$!



echo "[INFO] Servers launched."
echo "[INFO] PIDs:"
echo "  FP16             : $PID_FP16"
echo "  W4A8             : $PID_W4A8"
echo "  W4A8 + ATM       : $PID_W4A8_ATM"
echo "  W4A8 + ATM + OHB : $PID_W4A8_ATM_OHB"
echo "  W4A8 + OHB       : $PID_W4A8_OHB"

echo ""
echo "[INFO] Shutdown:"
echo "  fuser -k 8000/tcp 8001/tcp 8002/tcp 8003/tcp 8004/tcp"
echo "  or: pkill -f scripts/serve_policy.py"