#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/path_retrieve_server.sh start
#   ./scripts/path_retrieve_server.sh stop
#   ./scripts/path_retrieve_server.sh restart
#   ./scripts/path_retrieve_server.sh status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-start}"

INPUT_DIR="${INPUT_DIR:-${PROJECT_DIR}/data/input/WebQSP}"
CACHE="${CACHE:-${PROJECT_DIR}/data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt}"
PATH_RETRIEVE_SERVER_HOST="${PATH_RETRIEVE_SERVER_HOST:-0.0.0.0}"
PORT="${PORT:-8789}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/data/output/WebQSP/path_retrieve_server/logs}"
LOG_PATH="${LOG_PATH:-}"
PID_FILE="${PID_FILE:-/tmp/path_retrieve_server_${PORT}.pid}"
LOG_PATH_FILE="${LOG_PATH_FILE:-/tmp/path_retrieve_server_${PORT}.logpath}"
WAIT_FOR_HEALTH="${WAIT_FOR_HEALTH:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
PORT_BUSY_ACTION="${PORT_BUSY_ACTION:-ask}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/path_retrieve_server.sh start
  ./scripts/path_retrieve_server.sh stop
  ./scripts/path_retrieve_server.sh restart
  ./scripts/path_retrieve_server.sh status

Env vars:
  CACHE             score cache 路径
  INPUT_DIR         WebQSP 数据目录
  PORT              监听端口（默认 8789）
  PORT_BUSY_ACTION  ask | kill | cancel（默认 ask）
EOF
}

find_server_pids() {
  pgrep -f "oh_my_agent\\.path_retrieve_server\\.server.*--port ${PORT}" || true
}

find_port_pids() {
  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "${PORT}" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sort -u || true
  fi
}

stop_pids() {
  local pids="$1"
  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    kill "${pid}" 2>/dev/null || true
  done <<< "${pids}"
}

sync_pid_file() {
  local active_pid
  active_pid="$(find_server_pids | head -n 1)"
  if [[ -n "${active_pid}" ]]; then
    printf '%s\n' "${active_pid}" > "${PID_FILE}"
  else
    rm -f "${PID_FILE}"
  fi
  printf '%s' "${active_pid}"
}

resolve_log_path() {
  if [[ -n "${LOG_PATH}" ]]; then
    mkdir -p "$(dirname "${LOG_PATH}")"
    return
  fi
  mkdir -p "${LOG_ROOT}"
  LOG_PATH="${LOG_ROOT}/$(date '+%Y%m%d_%H%M%S')_port${PORT}_server.log"
}

choose_port_busy_action() {
  case "${PORT_BUSY_ACTION}" in
    kill|cancel) echo "${PORT_BUSY_ACTION}"; return ;;
  esac
  if [[ -t 0 ]]; then
    while true; do
      printf "端口 %s 已被占用。输入 [k] 杀掉占用进程并启动，或 [c] 取消启动: " "${PORT}" >&2
      read -r answer
      case "${answer}" in
        k|K|kill|KILL) echo "kill"; return ;;
        c|C|cancel|CANCEL) echo "cancel"; return ;;
      esac
    done
  fi
  echo "cancel"
}

start_server() {
  if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "[ERROR] INPUT_DIR not found: ${INPUT_DIR}" >&2
    exit 1
  fi
  if [[ ! -f "${CACHE}" ]]; then
    echo "[ERROR] CACHE not found: ${CACHE}" >&2
    exit 1
  fi
  resolve_log_path
  printf '%s\n' "${LOG_PATH}" > "${LOG_PATH_FILE}"

  local port_pids
  port_pids="$(find_port_pids)"
  if [[ -n "${port_pids}" ]]; then
    echo "[WARN] Port ${PORT} is occupied by PID(s): $(echo "${port_pids}" | tr '\n' ' ')" >&2
    if [[ "$(choose_port_busy_action)" != "kill" ]]; then
      echo "[INFO] Start cancelled because port ${PORT} is busy." >&2
      exit 1
    fi
    stop_pids "${port_pids}"
  fi

  cd "${PROJECT_DIR}"
  local cmd=(
    python -m oh_my_agent.path_retrieve_server.server
    --cache "${CACHE}"
    --input_dir "${INPUT_DIR}"
    --host "${PATH_RETRIEVE_SERVER_HOST}"
    --port "${PORT}"
  )
  if command -v setsid >/dev/null 2>&1; then
    setsid nohup "${cmd[@]}" > "${LOG_PATH}" 2>&1 &
  else
    nohup "${cmd[@]}" > "${LOG_PATH}" 2>&1 &
  fi

  local new_pid
  new_pid=$!
  echo "[INFO] Started path_retrieve_server PID=${new_pid} port=${PORT}"
  echo "[INFO] Cache    : ${CACHE}"
  echo "[INFO] Input dir: ${INPUT_DIR}"
  echo "[INFO] Log path : ${LOG_PATH}"

  if [[ "${WAIT_FOR_HEALTH}" != "1" ]]; then
    sync_pid_file >/dev/null
    return
  fi

  local health_url="http://127.0.0.1:${PORT}/health"
  for _ in $(seq 1 "${WAIT_SECONDS}"); do
    if curl -sf "${health_url}" >/dev/null 2>&1; then
      sync_pid_file >/dev/null
      echo "[INFO] Health check passed: ${health_url}"
      return
    fi
    if ! kill -0 "${new_pid}" 2>/dev/null; then
      echo "[ERROR] Server exited before health check passed." >&2
      tail -n 100 "${LOG_PATH}" >&2 || true
      exit 1
    fi
    sleep 1
  done
  echo "[ERROR] Timed out waiting for health check: ${health_url}" >&2
  tail -n 100 "${LOG_PATH}" >&2 || true
  exit 1
}

stop_server() {
  local server_pids
  server_pids="$(find_server_pids)"
  if [[ -z "${server_pids}" ]]; then
    echo "[INFO] No path_retrieve_server process found on port ${PORT}."
    rm -f "${PID_FILE}"
    return
  fi
  echo "[INFO] Stopping path_retrieve_server PID(s): $(echo "${server_pids}" | tr '\n' ' ')"
  stop_pids "${server_pids}"
  sync_pid_file >/dev/null
}

status_server() {
  local active_pid
  active_pid="$(sync_pid_file)"
  local saved_log_path=""
  [[ -f "${LOG_PATH_FILE}" ]] && saved_log_path="$(<"${LOG_PATH_FILE}")"
  if [[ -z "${active_pid}" ]]; then
    echo "[INFO] path_retrieve_server is not running on port ${PORT}."
    [[ -n "${saved_log_path}" ]] && echo "LAST_LOG_PATH=${saved_log_path}"
    echo "HEALTH_URL=http://127.0.0.1:${PORT}/health"
    return
  fi
  echo "[INFO] path_retrieve_server is running."
  echo "PID=${active_pid}"
  [[ -n "${saved_log_path}" ]] && echo "LOG_PATH=${saved_log_path}"
  echo "HEALTH_URL=http://127.0.0.1:${PORT}/health"
  ps -p "${active_pid}" -o pid,etime,stat,cmd
  curl -s "http://127.0.0.1:${PORT}/health" || true
  echo
  curl -s "http://127.0.0.1:${PORT}/info" || true
  echo
}

case "${ACTION}" in
  start) start_server ;;
  stop) stop_server ;;
  restart) stop_server; start_server ;;
  status) status_server ;;
  help|-h|--help) usage ;;
  *) echo "[ERROR] Unsupported action: ${ACTION}" >&2; usage >&2; exit 1 ;;
esac
