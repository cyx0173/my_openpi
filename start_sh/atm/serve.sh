#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi

OUT_DIR="/home/chengyuxuan/openpi/lab_track/ohb_2"
ATM_DIR="/home/chengyuxuan/openpi/lab_track/atm_1"

mkdir -p "$OUT_DIR"
mkdir -p "$ATM_DIR"

echo "[INFO] Kill old servers on 8000/8001"
fuser -k 8000/tcp || true
fuser -k 8001/tcp || true

echo "[INFO] Clean old paired ATM capture files"
rm -f "$ATM_DIR/atm_pair_fp16.jsonl"
rm -f "$ATM_DIR/atm_pair_w4a8.jsonl"

echo "[INFO] Start FP16 teacher + ATM capture on port 8000"

env \
-u OPENPI_ATM_ENABLE \
-u OPENPI_ATM_ALPHA_PATH \
-u OPENPI_ATM_ALPHA_ONES \
-u OPENPI_ATM_ALPHA_HALF \
-u OPENPI_OHB_ENABLE \
-u OPENPI_OHB_BETA_PATH \
-u OPENPI_OHB_SCOPE \
-u OPENPI_OHB_BETA_ONES \
-u OPENPI_OHB_BETA_CONSTANT \
-u OPENPI_OHB_CAPTURE_TAG \
-u OPENPI_OHB_CAPTURE_PATH \
OPENPI_ATM_SCOPE=gemma_expert \
OPENPI_ATM_CAPTURE_TAG=fp16_pair \
OPENPI_ATM_CAPTURE_PATH="$ATM_DIR/atm_pair_fp16.jsonl" \
CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py --port 8000 \
> "$OUT_DIR/atm_pair_fp16.log" 2>&1 &

PID_FP16=$!

echo "[INFO] Start W4A8 student + ATM capture on port 8001, no ATM apply"

env \
-u OPENPI_ATM_ENABLE \
-u OPENPI_ATM_ALPHA_PATH \
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
OPENPI_ATM_SCOPE=gemma_expert \
OPENPI_ATM_CAPTURE_TAG=w4a8_pair \
OPENPI_ATM_CAPTURE_PATH="$ATM_DIR/atm_pair_w4a8.jsonl" \
CUDA_VISIBLE_DEVICES=1 python scripts/serve_policy.py --port 8001 --quantize \
> "$OUT_DIR/atm_pair_w4a8.log" 2>&1 &

PID_W4A8=$!

echo "[INFO] Paired ATM calibration servers launched."
echo "[INFO] PIDs:"
echo "  FP16 teacher : $PID_FP16"
echo "  W4A8 student : $PID_W4A8"
echo
echo "[INFO] Logs:"
echo "  tail -f $OUT_DIR/atm_pair_fp16.log"
echo "  tail -f $OUT_DIR/atm_pair_w4a8.log"
echo
echo "[INFO] Capture outputs:"
echo "  $ATM_DIR/atm_pair_fp16.jsonl"
echo "  $ATM_DIR/atm_pair_w4a8.jsonl"