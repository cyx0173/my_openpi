#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/experiments/baseline/fp16_spatial"
LOG_DIR="$BASE_DIR/logs/client"
mkdir -p "$LOG_DIR"

run_one_task() {
  local port="$1"
  local tag="$2"

  {

    uv run python experiments/mode1_baseline.py \
      --args.port-w4a4 "$port" \
      --args.episode-start 0 \
      --args.episode-end 1 \
      --args.base-dir "$BASE_DIR/$tag"

  } > "$LOG_DIR/${tag}.log" 2>&1 &
}

run_one_task 8000 fp16
ps -ef | grep "experiments/mode1_baseline.py" | grep -v grep || true