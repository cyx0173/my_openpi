#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi

BASE_DIR="/home/chengyuxuan/openpi/lab_track/quant"
OUT_DIR="$BASE_DIR/logs/serve"
ATM_DIR="$BASE_DIR/atm_calib"


mkdir -p "$OUT_DIR"
mkdir -p "$ATM_DIR"

# ------------------------------------------------------------
# Quant mode mapping.
# 如果你 quant.py 里面不是这个映射，就只改这里。
# ------------------------------------------------------------
MODE_W4A4=1
MODE_W4A8=2
MODE_W4A16=3

# ------------------------------------------------------------
# Ports.
# 每个配置一个 FP16 teacher + 一个 quant student。
# ------------------------------------------------------------
PORT_FP16_W4A4=8000
PORT_W4A4=8001

PORT_FP16_W4A8=8002
PORT_W4A8=8003

PORT_FP16_W4A16=8004
PORT_W4A16=8005

# ------------------------------------------------------------
# GPU mapping.
# 如果你没有 6 张卡，就在这里改。
# ------------------------------------------------------------
GPU_FP16_W4A4=0
GPU_W4A4=1

GPU_FP16_W4A8=2
GPU_W4A8=3

GPU_FP16_W4A16=4
GPU_W4A16=5

echo "[INFO] Kill old servers on ports 8000-8005"
fuser -k ${PORT_FP16_W4A4}/tcp || true
fuser -k ${PORT_W4A4}/tcp || true
fuser -k ${PORT_FP16_W4A8}/tcp || true
fuser -k ${PORT_W4A8}/tcp || true
fuser -k ${PORT_FP16_W4A16}/tcp || true
fuser -k ${PORT_W4A16}/tcp || true

echo "[INFO] Clean old staged ATM capture files"

rm -f "$ATM_DIR/atm_staged_fp16_for_w4a4.jsonl"
rm -f "$ATM_DIR/atm_staged_w4a4.jsonl"
rm -f "$ATM_DIR/atm_staged_pair_w4a4_action_chunks.jsonl"

rm -f "$ATM_DIR/atm_staged_fp16_for_w4a8.jsonl"
rm -f "$ATM_DIR/atm_staged_w4a8.jsonl"
rm -f "$ATM_DIR/atm_staged_pair_w4a8_action_chunks.jsonl"

rm -f "$ATM_DIR/atm_staged_fp16_for_w4a16.jsonl"
rm -f "$ATM_DIR/atm_staged_w4a16.jsonl"
rm -f "$ATM_DIR/atm_staged_pair_w4a16_action_chunks.jsonl"

rm -f "$OUT_DIR/atm_staged_fp16_for_w4a4.log"
rm -f "$OUT_DIR/atm_staged_w4a4.log"

rm -f "$OUT_DIR/atm_staged_fp16_for_w4a8.log"
rm -f "$OUT_DIR/atm_staged_w4a8.log"

rm -f "$OUT_DIR/atm_staged_fp16_for_w4a16.log"
rm -f "$OUT_DIR/atm_staged_w4a16.log"


start_fp16_teacher() {
  local tag="$1"
  local port="$2"
  local gpu="$3"
  local capture_path="$4"
  local log_path="$5"

  echo "[INFO] Start FP16 teacher: tag=${tag}, port=${port}, gpu=${gpu}"

  env \
  -u OPENPI_QUANT_MODE \
  -u OPENPI_DUQUANT_STAGED \
  -u OPENPI_DUQUANT_INCLUDE \
  -u OPENPI_DUQUANT_EXCLUDE \
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
  OPENPI_ATM_CAPTURE_TAG="${tag}" \
  OPENPI_ATM_CAPTURE_PATH="${capture_path}" \
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/serve_policy.py --port "${port}" \
  > "${log_path}" 2>&1 &

  echo $!
}


start_quant_student() {
  local tag="$1"
  local port="$2"
  local gpu="$3"
  local quant_mode="$4"
  local capture_path="$5"
  local log_path="$6"

  echo "[INFO] Start STAGED quant student: tag=${tag}, port=${port}, gpu=${gpu}, mode=${quant_mode}"

  env \
  -u OPENPI_DUQUANT_INCLUDE \
  -u OPENPI_DUQUANT_EXCLUDE \
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
  OPENPI_QUANT_MODE="${quant_mode}" \
  OPENPI_DUQUANT_STAGED=1 \
  OPENPI_ATM_SCOPE=gemma_expert \
  OPENPI_ATM_CAPTURE_TAG="${tag}" \
  OPENPI_ATM_CAPTURE_PATH="${capture_path}" \
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/serve_policy.py --port "${port}" --quantize \
  > "${log_path}" 2>&1 &

  echo $!
}


echo "[INFO] Launch W4A4 pair"
PID_FP16_W4A4=$(start_fp16_teacher \
  "fp16_for_w4a4_staged_pair" \
  "$PORT_FP16_W4A4" \
  "$GPU_FP16_W4A4" \
  "$ATM_DIR/atm_staged_fp16_for_w4a4.jsonl" \
  "$OUT_DIR/atm_staged_fp16_for_w4a4.log")

PID_W4A4=$(start_quant_student \
  "w4a4_staged_pair" \
  "$PORT_W4A4" \
  "$GPU_W4A4" \
  "$MODE_W4A4" \
  "$ATM_DIR/atm_staged_w4a4.jsonl" \
  "$OUT_DIR/atm_staged_w4a4.log")


echo "[INFO] Launch W4A8 pair"
PID_FP16_W4A8=$(start_fp16_teacher \
  "fp16_for_w4a8_staged_pair" \
  "$PORT_FP16_W4A8" \
  "$GPU_FP16_W4A8" \
  "$ATM_DIR/atm_staged_fp16_for_w4a8.jsonl" \
  "$OUT_DIR/atm_staged_fp16_for_w4a8.log")

PID_W4A8=$(start_quant_student \
  "w4a8_staged_pair" \
  "$PORT_W4A8" \
  "$GPU_W4A8" \
  "$MODE_W4A8" \
  "$ATM_DIR/atm_staged_w4a8.jsonl" \
  "$OUT_DIR/atm_staged_w4a8.log")


echo "[INFO] Launch W4A16 pair"
PID_FP16_W4A16=$(start_fp16_teacher \
  "fp16_for_w4a16_staged_pair" \
  "$PORT_FP16_W4A16" \
  "$GPU_FP16_W4A16" \
  "$ATM_DIR/atm_staged_fp16_for_w4a16.jsonl" \
  "$OUT_DIR/atm_staged_fp16_for_w4a16.log")

PID_W4A16=$(start_quant_student \
  "w4a16_staged_pair" \
  "$PORT_W4A16" \
  "$GPU_W4A16" \
  "$MODE_W4A16" \
  "$ATM_DIR/atm_staged_w4a16.jsonl" \
  "$OUT_DIR/atm_staged_w4a16.log")


echo
echo "[INFO] All 6 ATM calibration servers launched."
echo
echo "[INFO] PIDs:"
echo "  FP16 for W4A4 : $PID_FP16_W4A4"
echo "  W4A4 student  : $PID_W4A4"
echo "  FP16 for W4A8 : $PID_FP16_W4A8"
echo "  W4A8 student  : $PID_W4A8"
echo "  FP16 for W4A16: $PID_FP16_W4A16"
echo "  W4A16 student : $PID_W4A16"
echo
echo "[INFO] Ports:"
echo "  W4A4 : teacher ${PORT_FP16_W4A4}, student ${PORT_W4A4}"
echo "  W4A8 : teacher ${PORT_FP16_W4A8}, student ${PORT_W4A8}"
echo "  W4A16: teacher ${PORT_FP16_W4A16}, student ${PORT_W4A16}"
echo
echo "[INFO] Logs:"
echo "  tail -f $OUT_DIR/atm_staged_fp16_for_w4a4.log"
echo "  tail -f $OUT_DIR/atm_staged_w4a4.log"
echo "  tail -f $OUT_DIR/atm_staged_fp16_for_w4a8.log"
echo "  tail -f $OUT_DIR/atm_staged_w4a8.log"
echo "  tail -f $OUT_DIR/atm_staged_fp16_for_w4a16.log"
echo "  tail -f $OUT_DIR/atm_staged_w4a16.log"
echo
echo "[INFO] Capture outputs:"
echo "  $ATM_DIR/atm_staged_fp16_for_w4a4.jsonl"
echo "  $ATM_DIR/atm_staged_w4a4.jsonl"
echo "  $ATM_DIR/atm_staged_fp16_for_w4a8.jsonl"
echo "  $ATM_DIR/atm_staged_w4a8.jsonl"
echo "  $ATM_DIR/atm_staged_fp16_for_w4a16.jsonl"
echo "  $ATM_DIR/atm_staged_w4a16.jsonl"
echo
echo "[INFO] Next: run paired clients for each pair."
echo
echo "W4A4:"
echo "  uv run python examples/test/main_ohb_calib_pair.py \\"
echo "    --args.host 127.0.0.1 \\"
echo "    --args.teacher-port ${PORT_FP16_W4A4} \\"
echo "    --args.student-port ${PORT_W4A4} \\"
echo "    --args.append-action-chunk $ATM_DIR/atm_staged_pair_w4a4_action_chunks.jsonl"
echo
echo "W4A8:"
echo "  uv run python examples/test/main_ohb_calib_pair.py \\"
echo "    --args.host 127.0.0.1 \\"
echo "    --args.teacher-port ${PORT_FP16_W4A8} \\"
echo "    --args.student-port ${PORT_W4A8} \\"
echo "    --args.append-action-chunk $ATM_DIR/atm_staged_pair_w4a8_action_chunks.jsonl"
echo
echo "W4A16:"
echo "  uv run python examples/test/main_ohb_calib_pair.py \\"
echo "    --args.host 127.0.0.1 \\"
echo "    --args.teacher-port ${PORT_FP16_W4A16} \\"
echo "    --args.student-port ${PORT_W4A16} \\"
echo "    --args.append-action-chunk $ATM_DIR/atm_staged_pair_w4a16_action_chunks.jsonl"