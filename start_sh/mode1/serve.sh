#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi

BASE_DIR="/home/chengyuxuan/openpi/active_quant/mode1_data_select"
LOG_DIR="$BASE_DIR/logs/serve"
mkdir -p "$LOG_DIR"

CALIB_JSON_DIR="/home/chengyuxuan/openpi/src/openpi/models_pytorch/atm/calib_json"

ATM_ALPHA_W4A4="$CALIB_JSON_DIR/pi05_atm_alpha_staged_w4a4.json"
ATM_ALPHA_W4A8="$CALIB_JSON_DIR/pi05_atm_alpha_staged_w4a8.json"
ATM_ALPHA_W4A16="$CALIB_JSON_DIR/pi05_atm_alpha_staged_w4a16.json"

OHB_BETA_W4A4="$CALIB_JSON_DIR/pi05_ohb_beta_staged_w4a4.json"
OHB_BETA_W4A8="$CALIB_JSON_DIR/pi05_ohb_beta_staged_w4a8.json"
OHB_BETA_W4A16="$CALIB_JSON_DIR/pi05_ohb_beta_staged_w4a16.json"

# ------------------------------------------------------------
# Quant mode mapping.
# 如果你 quant.py 里面不是这个映射，就改这里。
# ------------------------------------------------------------
MODE_W4A4=1
MODE_W4A8=2
MODE_W4A16=3

# ------------------------------------------------------------
# Group A: GPU 0-3, ports 8000-8003
# ------------------------------------------------------------
PORT_A_FP16=8000
PORT_A_W4A4=8001
PORT_A_W4A8=8002
PORT_A_W4A16=8003

GPU_A_FP16=0
GPU_A_W4A4=1
GPU_A_W4A8=2
GPU_A_W4A16=3

# ------------------------------------------------------------
# Group B: GPU 4-7, ports 8010-8013
# ------------------------------------------------------------
PORT_B_FP16=8010
PORT_B_W4A4=8011
PORT_B_W4A8=8012
PORT_B_W4A16=8013

GPU_B_FP16=4
GPU_B_W4A4=5
GPU_B_W4A8=6
GPU_B_W4A16=7

check_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing file: $f"
    exit 1
  fi
}

check_file "$ATM_ALPHA_W4A4"
check_file "$ATM_ALPHA_W4A8"
check_file "$ATM_ALPHA_W4A16"
check_file "$OHB_BETA_W4A4"
check_file "$OHB_BETA_W4A8"
check_file "$OHB_BETA_W4A16"

kill_port() {
  local p="$1"
  fuser -k "${p}/tcp" || true
}

echo "[INFO] Kill old servers on ports 8000-8003 and 8010-8013"
for p in \
  "$PORT_A_FP16" "$PORT_A_W4A4" "$PORT_A_W4A8" "$PORT_A_W4A16" \
  "$PORT_B_FP16" "$PORT_B_W4A4" "$PORT_B_W4A8" "$PORT_B_W4A16"; do
  kill_port "$p"
done

start_fp16() {
  local group="$1"
  local port="$2"
  local gpu="$3"
  local log_path="$4"

  echo "[INFO] Start FP16 ${group}: port=${port}, gpu=${gpu}"

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
  -u OPENPI_OHB_SCOPE \
  -u OPENPI_OHB_BETA_ONES \
  -u OPENPI_OHB_BETA_CONSTANT \
  -u OPENPI_OHB_CAPTURE_TAG \
  -u OPENPI_OHB_CAPTURE_PATH \
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/serve_policy.py --port "${port}" \
  > "${log_path}" 2>&1 &

  echo $!
}

start_quant() {
  local group="$1"
  local precision="$2"
  local port="$3"
  local gpu="$4"
  local quant_mode="$5"
  local atm_alpha="$6"
  local ohb_beta="$7"
  local log_path="$8"

  echo "[INFO] Start ${precision} ${group}: port=${port}, gpu=${gpu}, mode=${quant_mode}"
  echo "[INFO]   ATM alpha: ${atm_alpha}"
  echo "[INFO]   OHB beta : ${ohb_beta}"

  env \
  -u OPENPI_DUQUANT_INCLUDE \
  -u OPENPI_DUQUANT_EXCLUDE \
  -u OPENPI_ATM_ALPHA_ONES \
  -u OPENPI_ATM_ALPHA_HALF \
  -u OPENPI_ATM_CAPTURE_TAG \
  -u OPENPI_ATM_CAPTURE_PATH \
  -u OPENPI_OHB_BETA_ONES \
  -u OPENPI_OHB_BETA_CONSTANT \
  -u OPENPI_OHB_CAPTURE_TAG \
  -u OPENPI_OHB_CAPTURE_PATH \
  OPENPI_QUANT_MODE="${quant_mode}" \
  OPENPI_DUQUANT_STAGED=1 \
  OPENPI_ATM_ENABLE=1 \
  OPENPI_ATM_SCOPE=gemma_expert \
  OPENPI_ATM_ALPHA_PATH="${atm_alpha}" \
  OPENPI_OHB_ENABLE=1 \
  OPENPI_OHB_SCOPE=gemma_expert \
  OPENPI_OHB_BETA_PATH="${ohb_beta}" \
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/serve_policy.py --port "${port}" --quantize \
  > "${log_path}" 2>&1 &

  echo $!
}

echo "[INFO] Launch Group A: GPU 0-3"

PID_A_FP16=$(start_fp16 "A" "$PORT_A_FP16" "$GPU_A_FP16" "$LOG_DIR/groupA_fp16_8000.log")
PID_A_W4A4=$(start_quant "A" "W4A4" "$PORT_A_W4A4" "$GPU_A_W4A4" "$MODE_W4A4" "$ATM_ALPHA_W4A4" "$OHB_BETA_W4A4" "$LOG_DIR/groupA_w4a4_8001.log")
PID_A_W4A8=$(start_quant "A" "W4A8" "$PORT_A_W4A8" "$GPU_A_W4A8" "$MODE_W4A8" "$ATM_ALPHA_W4A8" "$OHB_BETA_W4A8" "$LOG_DIR/groupA_w4a8_8002.log")
PID_A_W4A16=$(start_quant "A" "W4A16" "$PORT_A_W4A16" "$GPU_A_W4A16" "$MODE_W4A16" "$ATM_ALPHA_W4A16" "$OHB_BETA_W4A16" "$LOG_DIR/groupA_w4a16_8003.log")

echo "[INFO] Launch Group B: GPU 4-7"

PID_B_FP16=$(start_fp16 "B" "$PORT_B_FP16" "$GPU_B_FP16" "$LOG_DIR/groupB_fp16_8010.log")
PID_B_W4A4=$(start_quant "B" "W4A4" "$PORT_B_W4A4" "$GPU_B_W4A4" "$MODE_W4A4" "$ATM_ALPHA_W4A4" "$OHB_BETA_W4A4" "$LOG_DIR/groupB_w4a4_8011.log")
PID_B_W4A8=$(start_quant "B" "W4A8" "$PORT_B_W4A8" "$GPU_B_W4A8" "$MODE_W4A8" "$ATM_ALPHA_W4A8" "$OHB_BETA_W4A8" "$LOG_DIR/groupB_w4a8_8012.log")
PID_B_W4A16=$(start_quant "B" "W4A16" "$PORT_B_W4A16" "$GPU_B_W4A16" "$MODE_W4A16" "$ATM_ALPHA_W4A16" "$OHB_BETA_W4A16" "$LOG_DIR/groupB_w4a16_8013.log")

echo
echo "[INFO] All final inference servers launched."
echo
echo "[INFO] PIDs:"
echo "  Group A FP16 : $PID_A_FP16"
echo "  Group A W4A4 : $PID_A_W4A4"
echo "  Group A W4A8 : $PID_A_W4A8"
echo "  Group A W4A16: $PID_A_W4A16"
echo "  Group B FP16 : $PID_B_FP16"
echo "  Group B W4A4 : $PID_B_W4A4"
echo "  Group B W4A8 : $PID_B_W4A8"
echo "  Group B W4A16: $PID_B_W4A16"
echo
echo "[INFO] Ports:"
echo "  Group A: FP16=$PORT_A_FP16 W4A4=$PORT_A_W4A4 W4A8=$PORT_A_W4A8 W4A16=$PORT_A_W4A16"
echo "  Group B: FP16=$PORT_B_FP16 W4A4=$PORT_B_W4A4 W4A8=$PORT_B_W4A8 W4A16=$PORT_B_W4A16"
echo
echo "[INFO] Logs:"
echo "  tail -f $LOG_DIR/groupA_fp16_8000.log"
echo "  tail -f $LOG_DIR/groupA_w4a4_8001.log"
echo "  tail -f $LOG_DIR/groupA_w4a8_8002.log"
echo "  tail -f $LOG_DIR/groupA_w4a16_8003.log"
echo "  tail -f $LOG_DIR/groupB_fp16_8010.log"
echo "  tail -f $LOG_DIR/groupB_w4a4_8011.log"
echo "  tail -f $LOG_DIR/groupB_w4a8_8012.log"
echo "  tail -f $LOG_DIR/groupB_w4a16_8013.log"