#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi

OUT_DIR="/home/chengyuxuan/openpi/lab_track/ohb_2"
ATM_DIR="/home/chengyuxuan/openpi/lab_track/atm_1"

ATM_ALPHA_JSON="$ATM_DIR/pi05_atm_alpha_w4a8_pair.json"

mkdir -p "$OUT_DIR"
mkdir -p "$ATM_DIR"

echo "[INFO] Kill old servers on 8000/8001"
fuser -k 8000/tcp || true
fuser -k 8001/tcp || true

echo "[INFO] Clean old paired OHB capture files"
rm -f "$ATM_DIR/ohb_pair_fp16.jsonl"
rm -f "$ATM_DIR/ohb_pair_w4a8_atm.jsonl"

echo "[INFO] Start FP16 teacher + OHB capture on port 8000"

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
-u OPENPI_OHB_BETA_ONES \
-u OPENPI_OHB_BETA_CONSTANT \
OPENPI_OHB_SCOPE=gemma_expert \
OPENPI_OHB_CAPTURE_TAG=fp16_pair \
OPENPI_OHB_CAPTURE_PATH="$ATM_DIR/ohb_pair_fp16.jsonl" \
CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py --port 8000 \
> "$OUT_DIR/ohb_pair_fp16.log" 2>&1 &

PID_FP16=$!


echo "[INFO] Start W4A8 + paired ATM student + OHB capture on port 8001"

env \
-u OPENPI_ATM_ALPHA_ONES \
-u OPENPI_ATM_ALPHA_HALF \
-u OPENPI_ATM_CAPTURE_TAG \
-u OPENPI_ATM_CAPTURE_PATH \
-u OPENPI_OHB_ENABLE \
-u OPENPI_OHB_BETA_PATH \
-u OPENPI_OHB_BETA_ONES \
-u OPENPI_OHB_BETA_CONSTANT \
OPENPI_QUANT_MODE=2 \
OPENPI_ATM_ENABLE=1 \
OPENPI_ATM_SCOPE=gemma_expert \
OPENPI_ATM_ALPHA_PATH="$ATM_ALPHA_JSON" \
OPENPI_OHB_SCOPE=gemma_expert \
OPENPI_OHB_CAPTURE_TAG=w4a8_atm_pair \
OPENPI_OHB_CAPTURE_PATH="$ATM_DIR/ohb_pair_w4a8_atm.jsonl" \
CUDA_VISIBLE_DEVICES=1 python scripts/serve_policy.py --port 8001 --quantize \
> "$OUT_DIR/ohb_pair_w4a8_atm.log" 2>&1 &

PID_W4A8_ATM=$!


echo "[INFO] Paired OHB calibration servers launched."
echo "[INFO] PIDs:"
echo "  FP16 teacher       : $PID_FP16"
echo "  W4A8 + paired ATM : $PID_W4A8_ATM"
echo
echo "[INFO] Logs:"
echo "  tail -f $OUT_DIR/ohb_pair_fp16.log"
echo "  tail -f $OUT_DIR/ohb_pair_w4a8_atm.log"
echo
echo "[INFO] Capture outputs:"
echo "  $ATM_DIR/ohb_pair_fp16.jsonl"
echo "  $ATM_DIR/ohb_pair_w4a8_atm.jsonl"
echo
echo "[INFO] Next paired client command:"
echo "  uv run python examples/test/main_ohb_calib_pair.py \\"
echo "    --args.host 127.0.0.1 \\"
echo "    --args.teacher-port 8000 \\"
echo "    --args.student-port 8001 \\"
echo "    --args.append-action-chunk $ATM_DIR/ohb_pair_action_chunks.jsonl"
echo
echo "[INFO] Shutdown:"
echo "  fuser -k 8000/tcp 8001/tcp"
echo "  or: pkill -f scripts/serve_policy.py"