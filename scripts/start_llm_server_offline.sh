#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/start_llm_server_offline.sh start
#   ./scripts/start_llm_server_offline.sh stop
#   ./scripts/start_llm_server_offline.sh restart
#   ./scripts/start_llm_server_offline.sh status
#   PORT=8790 ./scripts/start_llm_server_offline.sh start
#   PORT_BUSY_ACTION=kill ./scripts/start_llm_server_offline.sh start
#
# PORT_BUSY_ACTION:
#   ask    - 端口占用时交互选择 kill / cancel（默认）
#   kill   - 端口占用时直接杀掉占用进程后启动
#   cancel - 端口占用时直接取消启动

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-start}"

MODEL_ID="${MODEL_ID:-unsloth/meta-llama-3.1-8b-instruct-bnb-4bit}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/.cache/huggingface/hub}"
MODEL_CACHE_KEY="${MODEL_ID//\//--}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-}"
ADAPTER_PATH="${ADAPTER_PATH:-${PROJECT_DIR}/models/webqsp/ablation/groupJ_schema_name}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8788}"
LOG_PATH="${LOG_PATH:-/tmp/llm_server_${PORT}.log}"
PID_FILE="${PID_FILE:-/tmp/llm_server_${PORT}.pid}"
WAIT_FOR_HEALTH="${WAIT_FOR_HEALTH:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-180}"
PORT_BUSY_ACTION="${PORT_BUSY_ACTION:-ask}"
LLM_SERVER_MAX_BATCH_SIZE="${LLM_SERVER_MAX_BATCH_SIZE:-4}"
LLM_SERVER_BATCH_WAIT_MS="${LLM_SERVER_BATCH_WAIT_MS:-20}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export LLM_SERVER_MAX_BATCH_SIZE
export LLM_SERVER_BATCH_WAIT_MS
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE

usage() {
  cat <<'EOF'
Usage:
  ./scripts/start_llm_server_offline.sh start
  ./scripts/start_llm_server_offline.sh stop
  ./scripts/start_llm_server_offline.sh restart
  ./scripts/start_llm_server_offline.sh status
  PORT=8790 ./scripts/start_llm_server_offline.sh start
  PORT_BUSY_ACTION=kill ./scripts/start_llm_server_offline.sh start

PORT_BUSY_ACTION:
  ask    - 端口占用时交互选择 kill / cancel（默认）
  kill   - 端口占用时直接杀掉占用进程后启动
  cancel - 端口占用时直接取消启动
EOF
}

find_server_pids() {
  pgrep -f "oh_my_agent\\.llm_server\\.server.*--port ${PORT}" || true
}

find_port_pids() {
  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "${PORT}" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sort -u || true
    return
  fi

  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null | sort -u || true
    return
  fi

  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "( sport = :${PORT} )" 2>/dev/null \
      | grep -o 'pid=[0-9]\+' \
      | cut -d= -f2 \
      | sort -u || true
    return
  fi
}

wait_for_pids_exit() {
  local pids="$1"
  local timeout="${2:-30}"

  for _ in $(seq 1 "${timeout}"); do
    local alive=0
    while read -r pid; do
      [[ -z "${pid}" ]] && continue
      if kill -0 "${pid}" 2>/dev/null; then
        alive=1
        break
      fi
    done <<< "${pids}"

    if [[ "${alive}" == "0" ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

stop_pids() {
  local pids="$1"

  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    kill "${pid}" 2>/dev/null || true
  done <<< "${pids}"

  wait_for_pids_exit "${pids}" 30 || true
}

refresh_pid_file() {
  local active_pid
  active_pid="$(find_server_pids | head -n 1)"
  if [[ -n "${active_pid}" ]]; then
    echo "${active_pid}" > "${PID_FILE}"
  else
    rm -f "${PID_FILE}"
  fi
}

resolve_model_snapshot() {
  if [[ -n "${MODEL_SNAPSHOT}" ]]; then
    return
  fi

  local snapshot_dir
  snapshot_dir="${MODEL_CACHE_ROOT}/models--${MODEL_CACHE_KEY}/snapshots"
  if [[ ! -d "${snapshot_dir}" ]]; then
    echo "[ERROR] Snapshot directory not found: ${snapshot_dir}" >&2
    exit 1
  fi

  MODEL_SNAPSHOT="$(find "${snapshot_dir}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  if [[ -z "${MODEL_SNAPSHOT}" ]]; then
    echo "[ERROR] No snapshot found under: ${snapshot_dir}" >&2
    exit 1
  fi
}

ensure_start_inputs() {
  resolve_model_snapshot

  if [[ ! -d "${MODEL_SNAPSHOT}" ]]; then
    echo "[ERROR] Model snapshot not found: ${MODEL_SNAPSHOT}" >&2
    exit 1
  fi

  if [[ ! -d "${ADAPTER_PATH}" ]]; then
    echo "[ERROR] Adapter path not found: ${ADAPTER_PATH}" >&2
    exit 1
  fi
}

choose_port_busy_action() {
  if [[ "${PORT_BUSY_ACTION}" == "kill" || "${PORT_BUSY_ACTION}" == "cancel" ]]; then
    echo "${PORT_BUSY_ACTION}"
    return
  fi

  if [[ -t 0 ]]; then
    while true; do
      printf "端口 %s 已被占用。输入 [k] 杀掉占用进程并启动，或 [c] 取消启动: " "${PORT}" >&2
      read -r answer
      case "${answer}" in
        k|K|kill|KILL)
          echo "kill"
          return
          ;;
        c|C|cancel|CANCEL)
          echo "cancel"
          return
          ;;
      esac
    done
  fi

  echo "cancel"
}

start_server() {
  ensure_start_inputs

  local port_pids
  port_pids="$(find_port_pids)"
  if [[ -n "${port_pids}" ]]; then
    echo "[WARN] Port ${PORT} is occupied by PID(s): $(echo "${port_pids}" | tr '\n' ' ')" >&2
    local action
    action="$(choose_port_busy_action)"
    if [[ "${action}" == "cancel" ]]; then
      echo "[INFO] Start cancelled because port ${PORT} is busy." >&2
      refresh_pid_file
      exit 1
    fi
    stop_pids "${port_pids}"
  fi

  cd "${PROJECT_DIR}"
  nohup python -m oh_my_agent.llm_server.server \
    --model "${MODEL_SNAPSHOT}" \
    --adapter "${ADAPTER_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    > "${LOG_PATH}" 2>&1 &

  local new_pid
  new_pid=$!
  echo "[INFO] Started llm_server PID=${new_pid} port=${PORT}"
  echo "[INFO] Model snapshot: ${MODEL_SNAPSHOT}"
  echo "[INFO] Adapter path : ${ADAPTER_PATH}"
  echo "[INFO] Log path     : ${LOG_PATH}"

  if [[ "${WAIT_FOR_HEALTH}" != "1" ]]; then
    refresh_pid_file
    return
  fi

  local health_url
  health_url="http://127.0.0.1:${PORT}/health"
  for _ in $(seq 1 "${WAIT_SECONDS}"); do
    if curl -sf "${health_url}" >/dev/null 2>&1; then
      refresh_pid_file
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
    echo "[INFO] No llm_server process found on port ${PORT}."
    rm -f "${PID_FILE}"
    return
  fi

  echo "[INFO] Stopping llm_server PID(s): $(echo "${server_pids}" | tr '\n' ' ')"
  stop_pids "${server_pids}"
  refresh_pid_file
}

status_server() {
  refresh_pid_file

  local server_pids
  server_pids="$(find_server_pids)"
  if [[ -z "${server_pids}" ]]; then
    echo "[INFO] llm_server is not running on port ${PORT}."
    echo "PID_FILE=${PID_FILE} (absent)"
    echo "HEALTH_URL=http://127.0.0.1:${PORT}/health"
    return
  fi

  local active_pid
  active_pid="$(echo "${server_pids}" | head -n 1)"
  echo "[INFO] llm_server is running."
  echo "PID=${active_pid}"
  echo "PID_FILE=${PID_FILE}"
  echo "LOG_PATH=${LOG_PATH}"
  echo "HEALTH_URL=http://127.0.0.1:${PORT}/health"
  ps -p "${active_pid}" -o pid,etime,stat,cmd
  curl -s "http://127.0.0.1:${PORT}/health" || true
  echo
}

case "${ACTION}" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  restart)
    stop_server
    start_server
    ;;
  status)
    status_server
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "[ERROR] Unsupported action: ${ACTION}" >&2
    usage >&2
    exit 1
    ;;
esac
