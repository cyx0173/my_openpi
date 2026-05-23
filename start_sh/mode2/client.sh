#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi

SCRIPT="active_quant/mode2_recovery.py"

BANK_BASE_DIR="/home/chengyuxuan/openpi/active_quant/mode1_data_select"
OUT_BASE_DIR="/home/chengyuxuan/openpi/active_quant/mode2_recovery_select"
COMBINED_DIR="$BANK_BASE_DIR/checkpoint_bank/data"

LOG_DIR="$OUT_BASE_DIR/logs/client"
mkdir -p "$LOG_DIR"

NUM_WORKERS=8

mapfile -t TASK_IDS < <(
python - "$COMBINED_DIR" <<'PY'
import pathlib
import re
import sys

combined_dir = pathlib.Path(sys.argv[1])
task_ids = set()

for p in combined_dir.glob("task*_combined.json"):
    m = re.match(r"task(\d+)_", p.name)
    if m:
        task_ids.add(int(m.group(1)))

for tid in sorted(task_ids):
    print(tid)
PY
)

NUM_TASKS="${#TASK_IDS[@]}"

if [[ "$NUM_TASKS" -eq 0 ]]; then
  echo "[ERROR] No task combined files found in $COMBINED_DIR"
  exit 1
fi

echo "[INFO] Found $NUM_TASKS tasks in $COMBINED_DIR"
echo "[INFO] Task ids: ${TASK_IDS[*]}"

make_task_files() {
  local tasks_csv="$1"

  python - "$COMBINED_DIR" "$tasks_csv" <<'PY'
import pathlib
import sys

combined_dir = pathlib.Path(sys.argv[1])
tasks = [int(x) for x in sys.argv[2].split(",") if x != ""]

paths = []
for tid in tasks:
    prefix = f"task{tid:02d}_"
    paths.extend(sorted(combined_dir.glob(f"{prefix}*_combined.json")))

print(",".join(str(p) for p in paths))
PY
}

base=$((NUM_TASKS / NUM_WORKERS))
rem=$((NUM_TASKS % NUM_WORKERS))

echo "[INFO] Start mode6 recovery workers by task split"
echo "[INFO] base=$base rem=$rem"
echo "[INFO] SCRIPT=$SCRIPT"
echo "[INFO] BANK_BASE_DIR=$BANK_BASE_DIR"
echo "[INFO] OUT_BASE_DIR=$OUT_BASE_DIR"
echo "[INFO] save_video=True"

for worker in 0 1 2 3 4 5 6 7; do
  extra=0
  if [[ "$worker" -lt "$rem" ]]; then
    extra=1
  fi

  count=$((base + extra))

  if [[ "$worker" -lt "$rem" ]]; then
    start=$((worker * (base + 1)))
  else
    start=$((rem * (base + 1) + (worker - rem) * base))
  fi

  if [[ "$count" -le 0 ]]; then
    echo "[INFO] Skip worker=$worker: no assigned tasks"
    continue
  fi

  assigned=()
  for ((j=0; j<count; j++)); do
    assigned+=("${TASK_IDS[$((start + j))]}")
  done

  tasks_csv=$(IFS=,; echo "${assigned[*]}")
  files="$(make_task_files "$tasks_csv")"

  if [[ -z "$files" ]]; then
    echo "[INFO] Skip worker=$worker tasks=$tasks_csv: no files"
    continue
  fi

  port=$((8000 + worker))

  output_subdir=$(printf "recovery_chunk/worker%02d" "$worker")
  obs_subdir=$(printf "recovery_chunk/worker%02d/obs" "$worker")
  video_subdir=$(printf "recovery_chunk/worker%02d/videos" "$worker")
  vlm_hidden_subdir=$(printf "recovery_chunk/worker%02d/vlm_hidden" "$worker")

  log_path=$(printf "%s/mode6_worker%02d_tasks_%s.log" "$LOG_DIR" "$worker" "${tasks_csv//,/_}")

  echo "[INFO] Launch worker=$worker port=$port tasks=$tasks_csv"
  echo "[INFO]   output_subdir=$output_subdir"
  echo "[INFO]   video_subdir=$video_subdir"
  echo "[INFO]   log=$log_path"

  uv run python "$SCRIPT" \
    --args.bank-base-dir "$BANK_BASE_DIR" \
    --args.out-base-dir "$OUT_BASE_DIR" \
    --args.trajectory-list "$files" \
    --args.port-fp16 "$port" \
    --args.output-subdir "$output_subdir" \
    --args.obs-subdir "$obs_subdir" \
    --args.video-subdir "$video_subdir" \
    --args.vlm-hidden-subdir "$vlm_hidden_subdir" \
    --args.save-video \
    > "$log_path" 2>&1 &
done

echo
echo "[INFO] All assigned mode6 recovery workers launched."
echo "[INFO] Logs:"
for f in "$LOG_DIR"/mode6_worker*.log; do
  [[ -e "$f" ]] && echo "  tail -f $f"
done

echo
echo "[INFO] Video dirs:"
for worker in 0 1 2 3 4 5 6 7; do
  echo "  $OUT_BASE_DIR/recovery_chunk/worker$(printf "%02d" "$worker")/videos"
done
