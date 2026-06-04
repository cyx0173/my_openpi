#!/usr/bin/env bash
set -euo pipefail

# 先显示会杀哪些
for p in $(seq 8020 8029); do
  pgrep -af "experiments/mode3_collector.py.*--args.port-w4a4 $p" || true
done

# 再真正杀掉
for p in $(seq 8020 8029); do
  pkill -f "experiments/mode3_collector.py.*--args.port-w4a4 $p" || true
done