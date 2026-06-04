#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-}"
MODE="${2:-dry-run}"

if [[ -z "$ROOT" ]]; then
  echo "Usage:"
  echo "  bash rm_w4a4_w4a8_dirs.sh <root_dir>            # dry-run"
  echo "  bash rm_w4a4_w4a8_dirs.sh <root_dir> --delete   # actually delete"
  exit 1
fi

if [[ ! -d "$ROOT" ]]; then
  echo "[ERROR] Not a directory: $ROOT"
  exit 1
fi

if [[ "$MODE" == "--delete" ]]; then
  find "$ROOT" -type d \( -name "w4a4" -o -name "w4a8" \) -prune -print -exec rm -rf {} +
else
  echo "[DRY-RUN] These directories would be removed:"
  find "$ROOT" -type d \( -name "w4a4" -o -name "w4a8" \) -prune -print
fi