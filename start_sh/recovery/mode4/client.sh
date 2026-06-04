#!/usr/bin/env bash
set -euo pipefail

cd /home/chengyuxuan/openpi

FAILED_JSON="/home/chengyuxuan/openpi/experiments/mode3/data/w4a4_failed_episodes.json"
OUT_ROOT="/home/chengyuxuan/openpi/experiments/mode4/data"
LOG_ROOT="/home/chengyuxuan/openpi/experiments/mode4/logs"

PORT_W4A16=8000
PORT_W4A8=8001
W4A4_PORTS=(8002 8003 8004 8005 8006 8007 8008 8009 8010 8011 8012 8013 8014 8015)

# Usage:
#   bash client.sh          # run w4a16 and w4a8 rescue
#   bash client.sh w4a16    # run only w4a16 rescue
#   bash client.sh w4a8     # run only w4a8 rescue
TAGS=("${@:-w4a16 w4a8}")
if [[ $# -eq 0 ]]; then
  TAGS=(w4a16 w4a8)
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

num_jobs() {
  python - "$FAILED_JSON" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(len(json.load(f)))
PY
}

read_job() {
  local idx="$1"
  python - "$FAILED_JSON" "$idx" <<'PY'
import json, sys
path, idx = sys.argv[1], int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    item = json.load(f)[idx]
print(f'{int(item["task_id"])}\t{int(item["eps_id"])}\t{item["action_chunk_json"]}')
PY
}

inject_port_for_tag() {
  case "$1" in
    w4a16) echo "$PORT_W4A16" ;;
    w4a8)  echo "$PORT_W4A8" ;;
    *) echo "unknown tag: $1" >&2; exit 1 ;;
  esac
}

run_one() {
  local tag="$1"
  local w4a4_port="$2"
  local task_id="$3"
  local eps_id="$4"
  local action_chunk_json="$5"

  local inject_port
  inject_port="$(inject_port_for_tag "$tag")"

  local ep3
  ep3="$(printf "%03d" "$eps_id")"

  local base_dir="${OUT_ROOT}/${tag}"
  local log_dir="${LOG_ROOT}/${tag}"
  local log_file="${log_dir}/task${task_id}_ep${ep3}_main${w4a4_port}.log"
  mkdir -p "$base_dir" "$log_dir"

  uv run python experiments/mode2_recovery.py \
    --args.action-chunk-json "$action_chunk_json" \
    --args.inject-chunks all \
    --args.main-policy w4a4 \
    --args.inject-policy "$tag" \
    --args.port-a16 "$w4a4_port" \
    --args.port-inject "$inject_port" \
    --args.base-dir "$base_dir" \
    > "$log_file" 2>&1
}

run_tag() {
  local tag="$1"
  local n
  n="$(num_jobs)"

  local tmpdir
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT

  echo 0 > "${tmpdir}/next"

  claim_idx() {
    local idx
    {
      flock -x 200
      idx="$(cat "${tmpdir}/next")"
      if (( idx >= n )); then
        echo ""
      else
        echo $((idx + 1)) > "${tmpdir}/next"
        echo "$idx"
      fi
    } 200>"${tmpdir}/lock"
  }

  worker() {
    local w4a4_port="$1"
    while true; do
      local idx
      idx="$(claim_idx)"
      [[ -z "$idx" ]] && break

      local task_id eps_id action_chunk_json
      IFS=$'\t' read -r task_id eps_id action_chunk_json < <(read_job "$idx")
      run_one "$tag" "$w4a4_port" "$task_id" "$eps_id" "$action_chunk_json"
    done
  }

  local pids=()
  for port in "${W4A4_PORTS[@]}"; do
    worker "$port" &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

for tag in "${TAGS[@]}"; do
  run_tag "$tag"
done
