#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi

# ------------------------------------------------------------
# 这里直接改 atm 或 ohb
# CALIB_KIND="atm"
# CALIB_KIND="ohb"
# ------------------------------------------------------------
CALIB_KIND="ohb"

if [[ "$CALIB_KIND" != "atm" && "$CALIB_KIND" != "ohb" ]]; then
  echo "[ERROR] CALIB_KIND must be atm or ohb, got: $CALIB_KIND"
  exit 1
fi

BASE_DIR="/home/chengyuxuan/openpi/lab_track/quant"

OUT_DIR="$BASE_DIR/logs/client"
CLIENT_DIR="$BASE_DIR/client"

mkdir -p "$OUT_DIR"
mkdir -p "$CLIENT_DIR"

CLIENT_SCRIPT="examples/test/main.py"

echo "[INFO] CALIB_KIND=$CALIB_KIND"
echo "[INFO] CLIENT_SCRIPT=$CLIENT_SCRIPT"

echo "[INFO] Clean old ${CALIB_KIND} client outputs"

rm -f "$CLIENT_DIR/${CALIB_KIND}_staged_pair_w4a4_action_chunks.jsonl"
rm -f "$CLIENT_DIR/${CALIB_KIND}_staged_pair_w4a8_action_chunks.jsonl"
rm -f "$CLIENT_DIR/${CALIB_KIND}_staged_pair_w4a16_action_chunks.jsonl"

rm -f "$OUT_DIR/client_${CALIB_KIND}_staged_w4a4.log"
rm -f "$OUT_DIR/client_${CALIB_KIND}_staged_w4a8.log"
rm -f "$OUT_DIR/client_${CALIB_KIND}_staged_w4a16.log"

echo "[INFO] Launch paired client for W4A4: teacher=8000 student=8001"

uv run python "$CLIENT_SCRIPT" \
  --args.host 127.0.0.1 \
  --args.teacher-port 8000 \
  --args.student-port 8001 \
  --args.append-action-chunk "$CLIENT_DIR/${CALIB_KIND}_staged_pair_w4a4_action_chunks.jsonl" \
  > "$OUT_DIR/client_${CALIB_KIND}_staged_w4a4.log" 2>&1 &

PID_W4A4=$!

echo "[INFO] Launch paired client for W4A8: teacher=8002 student=8003"

uv run python "$CLIENT_SCRIPT" \
  --args.host 127.0.0.1 \
  --args.teacher-port 8002 \
  --args.student-port 8003 \
  --args.append-action-chunk "$CLIENT_DIR/${CALIB_KIND}_staged_pair_w4a8_action_chunks.jsonl" \
  > "$OUT_DIR/client_${CALIB_KIND}_staged_w4a8.log" 2>&1 &

PID_W4A8=$!

echo "[INFO] Launch paired client for W4A16: teacher=8004 student=8005"

uv run python "$CLIENT_SCRIPT" \
  --args.host 127.0.0.1 \
  --args.teacher-port 8004 \
  --args.student-port 8005 \
  --args.append-action-chunk "$CLIENT_DIR/${CALIB_KIND}_staged_pair_w4a16_action_chunks.jsonl" \
  > "$OUT_DIR/client_${CALIB_KIND}_staged_w4a16.log" 2>&1 &

PID_W4A16=$!

echo
echo "[INFO] ${CALIB_KIND^^} paired clients launched."
echo "[INFO] PIDs:"
echo "  W4A4 client : $PID_W4A4"
echo "  W4A8 client : $PID_W4A8"
echo "  W4A16 client: $PID_W4A16"
echo
echo "[INFO] Logs:"
echo "  tail -f $OUT_DIR/client_${CALIB_KIND}_staged_w4a4.log"
echo "  tail -f $OUT_DIR/client_${CALIB_KIND}_staged_w4a8.log"
echo "  tail -f $OUT_DIR/client_${CALIB_KIND}_staged_w4a16.log"
echo
echo "[INFO] Client outputs:"
echo "  $CLIENT_DIR/${CALIB_KIND}_staged_pair_w4a4_action_chunks.jsonl"
echo "  $CLIENT_DIR/${CALIB_KIND}_staged_pair_w4a8_action_chunks.jsonl"
echo "  $CLIENT_DIR/${CALIB_KIND}_staged_pair_w4a16_action_chunks.jsonl"