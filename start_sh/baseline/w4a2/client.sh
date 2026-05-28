#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi
source start_sh/reset_path.sh

BASE_DIR="/home/chengyuxuan/openpi/experiments/baseline/w4a2"
LOG_DIR="$BASE_DIR/logs/client"
mkdir -p "$LOG_DIR"

run_one_task() {
  local port="$1"
  local tag="$2"

  {

    uv run python examples/libero/main.py \
      --args.port "$port" \

  } > "$LOG_DIR/${tag}.log" 2>&1 &
}

run_one_task 8001 w4a2
ps -ef | grep "examples/libero/main.py" | grep -v grep || true