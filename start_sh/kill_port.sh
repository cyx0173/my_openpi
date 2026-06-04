kill_port() {
  local port="$1"
  echo "[KILL] port=${port}"
  lsof -ti:"${port}" | xargs -r kill -9
}

kill_port_range() {
  local start_port="$1"
  local end_port="$2"

  for port in $(seq "${start_port}" "${end_port}"); do
    kill_port "${port}"
  done
}

kill_task_ports() {
  local task_id="$1"
  local base_port="$2"

  echo "[KILL TASK] task=${task_id} base_port=${base_port}"

  kill_port "$((base_port + 0))"
  kill_port "$((base_port + 1))"
  kill_port "$((base_port + 2))"
  kill_port "$((base_port + 3))"
}
kill_port_range 8000 8032