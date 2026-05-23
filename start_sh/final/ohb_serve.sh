#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi

BASE_DIR="/home/chengyuxuan/openpi/lab_track/quant"

OUT_DIR="$BASE_DIR/logs/serve_ohb"
ATM_DIR="$BASE_DIR/atm_calib"
OHB_DIR="$BASE_DIR/ohb_calib"

mkdir -p "$OUT_DIR"
mkdir -p "$ATM_DIR"
mkdir -p "$OHB_DIR"

# ------------------------------------------------------------
# Quant mode mapping.
# 如果 quant.py 里面不是这个映射，就只改这里。
# ------------------------------------------------------------
MODE_W4A4=1
MODE_W4A8=2
MODE_W4A16=3

# ------------------------------------------------------------
# ATM alpha paths.
# 这些文件必须已经由 ATM calibration 生成好。
# ------------------------------------------------------------
ATM_ALPHA_W4A4="$ATM_DIR/pi05_atm_alpha_staged_w4a4.json"
ATM_ALPHA_W4A8="$ATM_DIR/pi05_atm_alpha_staged_w4a8.json"
ATM_ALPHA_W4A16="$ATM_DIR/pi05_atm_alpha_staged_w4a16.json"

for f in "$ATM_ALPHA_W4A4" "$ATM_ALPHA_W4A8" "$ATM_ALPHA_W4A16"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing ATM alpha file: $f"
    echo "[ERROR] Please run ATM calibration and make_atm_alpha.py first."
    exit 1
  fi
done

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
# 如果没有 6 张卡，就在这里改。
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

echo "[INFO] Clean old staged OHB capture files"

rm -f "$OHB_DIR/ohb_staged_fp16_for_w4a4.jsonl"
rm -f "$OHB_DIR/ohb_staged_w4a4_atm.jsonl"
rm -f "$OHB_DIR/ohb_staged_fp16_for_w4a8.jsonl"
rm -f "$OHB_DIR/ohb_staged_w4a8_atm.jsonl"
rm -f "$OHB_DIR/ohb_staged_fp16_for_w4a16.jsonl"
rm -f "$OHB_DIR/ohb_staged_w4a16_atm.jsonl"

rm -f "$OUT_DIR/ohb_staged_fp16_for_w4a4.log"
rm -f "$OUT_DIR/ohb_staged_w4a4_atm.log"
rm -f "$OUT_DIR/ohb_staged_fp16_for_w4a8.log"
rm -f "$OUT_DIR/ohb_staged_w4a8_atm.log"
rm -f "$OUT_DIR/ohb_staged_fp16_for_w4a16.log"
rm -f "$OUT_DIR/ohb_staged_w4a16_atm.log"


start_fp16_teacher() {
  local tag="$1"
  local port="$2"
  local gpu="$3"
  local capture_path="$4"
  local log_path="$5"

  echo "[INFO] Start FP16 teacher + OHB capture: tag=${tag}, port=${port}, gpu=${gpu}"

  env \
  -u OPENPI_QUANT_MODE \
  -u OPENPI_DUQUANT_STAGED \
  -u OPENPI_DUQUANT_INCLUDE \
  -u OPENPI_DUQUANT_EXCLUDE \
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
  OPENPI_OHB_CAPTURE_TAG="${tag}" \
  OPENPI_OHB_CAPTURE_PATH="${capture_path}" \
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/serve_policy.py --port "${port}" \
  > "${log_path}" 2>&1 &

  echo $!
}


start_quant_student_with_atm_ohb_capture() {
  local tag="$1"
  local port="$2"
  local gpu="$3"
  local quant_mode="$4"
  local atm_alpha_path="$5"
  local capture_path="$6"
  local log_path="$7"

  echo "[INFO] Start STAGED quant student + ATM apply + OHB capture: tag=${tag}, port=${port}, gpu=${gpu}, mode=${quant_mode}"
  echo "[INFO] ATM alpha: ${atm_alpha_path}"

  env \
  -u OPENPI_DUQUANT_INCLUDE \
  -u OPENPI_DUQUANT_EXCLUDE \
  -u OPENPI_ATM_ALPHA_ONES \
  -u OPENPI_ATM_ALPHA_HALF \
  -u OPENPI_ATM_CAPTURE_TAG \
  -u OPENPI_ATM_CAPTURE_PATH \
  -u OPENPI_OHB_ENABLE \
  -u OPENPI_OHB_BETA_PATH \
  -u OPENPI_OHB_BETA_ONES \
  -u OPENPI_OHB_BETA_CONSTANT \
  OPENPI_QUANT_MODE="${quant_mode}" \
  OPENPI_DUQUANT_STAGED=1 \
  OPENPI_ATM_ENABLE=1 \
  OPENPI_ATM_SCOPE=gemma_expert \
  OPENPI_ATM_ALPHA_PATH="${atm_alpha_path}" \
  OPENPI_OHB_SCOPE=gemma_expert \
  OPENPI_OHB_CAPTURE_TAG="${tag}" \
  OPENPI_OHB_CAPTURE_PATH="${capture_path}" \
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/serve_policy.py --port "${port}" --quantize \
  > "${log_path}" 2>&1 &

  echo $!
}


echo "[INFO] Launch W4A4 OHB pair"
PID_FP16_W4A4=$(start_fp16_teacher \
  "fp16_for_w4a4_ohb_staged_pair" \
  "$PORT_FP16_W4A4" \
  "$GPU_FP16_W4A4" \
  "$OHB_DIR/ohb_staged_fp16_for_w4a4.jsonl" \
  "$OUT_DIR/ohb_staged_fp16_for_w4a4.log")

PID_W4A4=$(start_quant_student_with_atm_ohb_capture \
  "w4a4_atm_ohb_staged_pair" \
  "$PORT_W4A4" \
  "$GPU_W4A4" \
  "$MODE_W4A4" \
  "$ATM_ALPHA_W4A4" \
  "$OHB_DIR/ohb_staged_w4a4_atm.jsonl" \
  "$OUT_DIR/ohb_staged_w4a4_atm.log")


echo "[INFO] Launch W4A8 OHB pair"
PID_FP16_W4A8=$(start_fp16_teacher \
  "fp16_for_w4a8_ohb_staged_pair" \
  "$PORT_FP16_W4A8" \
  "$GPU_FP16_W4A8" \
  "$OHB_DIR/ohb_staged_fp16_for_w4a8.jsonl" \
  "$OUT_DIR/ohb_staged_fp16_for_w4a8.log")

PID_W4A8=$(start_quant_student_with_atm_ohb_capture \
  "w4a8_atm_ohb_staged_pair" \
  "$PORT_W4A8" \
  "$GPU_W4A8" \
  "$MODE_W4A8" \
  "$ATM_ALPHA_W4A8" \
  "$OHB_DIR/ohb_staged_w4a8_atm.jsonl" \
  "$OUT_DIR/ohb_staged_w4a8_atm.log")


echo "[INFO] Launch W4A16 OHB pair"
PID_FP16_W4A16=$(start_fp16_teacher \
  "fp16_for_w4a16_ohb_staged_pair" \
  "$PORT_FP16_W4A16" \
  "$GPU_FP16_W4A16" \
  "$OHB_DIR/ohb_staged_fp16_for_w4a16.jsonl" \
  "$OUT_DIR/ohb_staged_fp16_for_w4a16.log")

PID_W4A16=$(start_quant_student_with_atm_ohb_capture \
  "w4a16_atm_ohb_staged_pair" \
  "$PORT_W4A16" \
  "$GPU_W4A16" \
  "$MODE_W4A16" \
  "$ATM_ALPHA_W4A16" \
  "$OHB_DIR/ohb_staged_w4a16_atm.jsonl" \
  "$OUT_DIR/ohb_staged_w4a16_atm.log")


echo
echo "[INFO] All 6 OHB calibration servers launched."
echo
echo "[INFO] PIDs:"
echo "  FP16 for W4A4 : $PID_FP16_W4A4"
echo "  W4A4 + ATM    : $PID_W4A4"
echo "  FP16 for W4A8 : $PID_FP16_W4A8"
echo "  W4A8 + ATM    : $PID_W4A8"
echo "  FP16 for W4A16: $PID_FP16_W4A16"
echo "  W4A16 + ATM   : $PID_W4A16"
echo
echo "[INFO] Ports:"
echo "  W4A4 : teacher ${PORT_FP16_W4A4}, student ${PORT_W4A4}"
echo "  W4A8 : teacher ${PORT_FP16_W4A8}, student ${PORT_W4A8}"
echo "  W4A16: teacher ${PORT_FP16_W4A16}, student ${PORT_W4A16}"
echo
echo "[INFO] Logs:"
echo "  tail -f $OUT_DIR/ohb_staged_fp16_for_w4a4.log"
echo "  tail -f $OUT_DIR/ohb_staged_w4a4_atm.log"
echo "  tail -f $OUT_DIR/ohb_staged_fp16_for_w4a8.log"
echo "  tail -f $OUT_DIR/ohb_staged_w4a8_atm.log"
echo "  tail -f $OUT_DIR/ohb_staged_fp16_for_w4a16.log"
echo "  tail -f $OUT_DIR/ohb_staged_w4a16_atm.log"
echo
echo "[INFO] OHB capture outputs:"
echo "  $OHB_DIR/ohb_staged_fp16_for_w4a4.jsonl"
echo "  $OHB_DIR/ohb_staged_w4a4_atm.jsonl"
echo "  $OHB_DIR/ohb_staged_fp16_for_w4a8.jsonl"
echo "  $OHB_DIR/ohb_staged_w4a8_atm.jsonl"
echo "  $OHB_DIR/ohb_staged_fp16_for_w4a16.jsonl"
echo "  $OHB_DIR/ohb_staged_w4a16_atm.jsonl"
echo
echo "[INFO] Next: run paired clients for each pair."