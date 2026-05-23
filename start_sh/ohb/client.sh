#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi

OUT_DIR="/home/chengyuxuan/openpi/lab_track/final"
mkdir -p "$OUT_DIR"

CLIENT_SCRIPT="examples/libero/main.py"

echo "[INFO] Clean old client outputs"

rm -f "$OUT_DIR/action_chunks_fp16.jsonl"
rm -f "$OUT_DIR/action_chunks_w4a8.jsonl"
rm -f "$OUT_DIR/action_chunks_w4a8_atm.jsonl"
rm -f "$OUT_DIR/action_chunks_w4a8_atm_ohb.jsonl"
rm -f "$OUT_DIR/action_chunks_fp16_ohb.jsonl"

rm -f "$OUT_DIR/client_fp16.log"
rm -f "$OUT_DIR/client_w4a8.log"
rm -f "$OUT_DIR/client_w4a8_atm.log"
rm -f "$OUT_DIR/client_w4a8_atm_ohb.log"
rm -f "$OUT_DIR/client_fp16_ohb.log"

echo "[INFO] Run client against FP16 server, port=8000"
uv run python "$CLIENT_SCRIPT" \
  --args.port 8000 \
  --args.append-action-chunk "$OUT_DIR/action_chunks/action_chunks_fp16.jsonl" \
  > "$OUT_DIR/logs/client_fp16.log" 2>&1 &
PID_FP16=$!

echo "[INFO] Run client against W4A8 server, port=8001"
uv run python "$CLIENT_SCRIPT" \
  --args.port 8001 \
  --args.append-action-chunk "$OUT_DIR/action_chunks/action_chunks_w4a8.jsonl" \
  > "$OUT_DIR/logs/client_w4a8.log" 2>&1 &
PID_W4A8=$!

echo "[INFO] Run client against W4A8 + ATM server, port=8002"
uv run python "$CLIENT_SCRIPT" \
  --args.port 8002 \
  --args.append-action-chunk "$OUT_DIR/action_chunks/action_chunks_w4a8_atm.jsonl" \
  > "$OUT_DIR/logs/client_w4a8_atm.log" 2>&1 &
PID_W4A8_ATM=$!

echo "[INFO] Run client against W4A8 + ATM + OHB server, port=8003"
uv run python "$CLIENT_SCRIPT" \
  --args.port 8003 \
  --args.append-action-chunk "$OUT_DIR/action_chunks/action_chunks_w4a8_atm_ohb.jsonl" \
  > "$OUT_DIR/logs/client_w4a8_atm_ohb.log" 2>&1 &
PID_W4A8_ATM_OHB=$!

echo "[INFO] Run client against W4A8 + OHB server, port=8004"
uv run python "$CLIENT_SCRIPT" \
  --args.port 8004 \
  --args.append-action-chunk "$OUT_DIR/action_chunks/action_chunks_w4a8_ohb.jsonl" \
  > "$OUT_DIR/logs/client_w4a8_ohb.log" 2>&1 &
PID_W4A8_OHB=$!

echo "[INFO] Clients launched in background."
echo "[INFO] PIDs:"
echo "  FP16             : $PID_FP16"
echo "  W4A8             : $PID_W4A8"
echo "  W4A8 + ATM       : $PID_W4A8_ATM"
echo "  W4A8 + ATM + OHB : $PID_W4A8_ATM_OHB"
echo "  W4A8 + OHB       : $PID_W4A8_OHB"

echo ""
echo "[INFO] Stop clients:"
echo "  pkill -f examples/test/main.py"