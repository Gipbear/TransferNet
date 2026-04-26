#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/llm_server.sh start
#   ./scripts/llm_server.sh stop
#   ./scripts/llm_server.sh restart
#   ./scripts/llm_server.sh status
#   PORT=8790 ./scripts/llm_server.sh start
#   PORT_BUSY_ACTION=kill ./scripts/llm_server.sh start
#
# Logs:
#   默认写入 data/output/WebQSP/llm_server/<YYYYmmdd_HHMMSS>_port<PORT>_server.log
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
LLM_SERVER_HOST="${LLM_SERVER_HOST:-0.0.0.0}"
PORT="${PORT:-8788}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/data/output/WebQSP/llm_server}"
LOG_PATH="${LOG_PATH:-}"
PID_FILE="${PID_FILE:-/tmp/llm_server_${PORT}.pid}"
LOG_PATH_FILE="${LOG_PATH_FILE:-/tmp/llm_server_${PORT}.logpath}"
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
  ./scripts/llm_server.sh start
  ./scripts/llm_server.sh stop
  ./scripts/llm_server.sh restart
  ./scripts/llm_server.sh status
  PORT=8790 ./scripts/llm_server.sh start
  PORT_BUSY_ACTION=kill ./scripts/llm_server.sh start

Logs:
  默认写入 data/output/WebQSP/llm_server/<YYYYmmdd_HHMMSS>_port<PORT>_server.log

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

stop_pids() {
  local pids="$1"
  local timeout="${2:-30}"

  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    kill "${pid}" 2>/dev/null || true
  done <<< "${pids}"

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

  local timestamp
  timestamp="$(date '+%Y%m%d_%H%M%S')"

  local log_path
  log_path="${LOG_ROOT}/${timestamp}_port${PORT}_server.log"

  local suffix=1
  while [[ -e "${log_path}" ]]; do
    log_path="${LOG_ROOT}/${timestamp}_port${PORT}_server_${suffix}.log"
    suffix=$((suffix + 1))
  done

  LOG_PATH="${log_path}"
}

resolve_start_inputs() {
  if [[ -z "${MODEL_SNAPSHOT}" ]]; then
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
  fi

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
  case "${PORT_BUSY_ACTION}" in
    kill|cancel)
      echo "${PORT_BUSY_ACTION}"
      return
      ;;
  esac

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
  resolve_start_inputs
  resolve_log_path
  printf '%s\n' "${LOG_PATH}" > "${LOG_PATH_FILE}"

  local port_pids
  port_pids="$(find_port_pids)"
  if [[ -n "${port_pids}" ]]; then
    echo "[WARN] Port ${PORT} is occupied by PID(s): $(echo "${port_pids}" | tr '\n' ' ')" >&2
    local action
    action="$(choose_port_busy_action)"
    if [[ "${action}" == "cancel" ]]; then
      echo "[INFO] Start cancelled because port ${PORT} is busy." >&2
      sync_pid_file >/dev/null
      exit 1
    fi
    stop_pids "${port_pids}"
  fi

  cd "${PROJECT_DIR}"
  nohup python -m oh_my_agent.llm_server.server \
    --model "${MODEL_SNAPSHOT}" \
    --adapter "${ADAPTER_PATH}" \
    --host "${LLM_SERVER_HOST}" \
    --port "${PORT}" \
    > "${LOG_PATH}" 2>&1 &

  local new_pid
  new_pid=$!
  echo "[INFO] Started llm_server PID=${new_pid} port=${PORT}"
  echo "[INFO] Model snapshot: ${MODEL_SNAPSHOT}"
  echo "[INFO] Adapter path : ${ADAPTER_PATH}"
  echo "[INFO] Log path     : ${LOG_PATH}"

  if [[ "${WAIT_FOR_HEALTH}" != "1" ]]; then
    sync_pid_file >/dev/null
    return
  fi

  local health_url
  health_url="http://127.0.0.1:${PORT}/health"
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
    echo "[INFO] No llm_server process found on port ${PORT}."
    rm -f "${PID_FILE}"
    return
  fi

  echo "[INFO] Stopping llm_server PID(s): $(echo "${server_pids}" | tr '\n' ' ')"
  stop_pids "${server_pids}"
  sync_pid_file >/dev/null
}

status_server() {
  local active_pid
  active_pid="$(sync_pid_file)"
  local saved_log_path
  saved_log_path=""
  if [[ -f "${LOG_PATH_FILE}" ]]; then
    saved_log_path="$(<"${LOG_PATH_FILE}")"
  fi

  if [[ -z "${active_pid}" ]]; then
    echo "[INFO] llm_server is not running on port ${PORT}."
    echo "PID_FILE=${PID_FILE} (absent)"
    if [[ -n "${saved_log_path}" ]]; then
      echo "LAST_LOG_PATH=${saved_log_path}"
    fi
    echo "HEALTH_URL=http://127.0.0.1:${PORT}/health"
    return
  fi

  echo "[INFO] llm_server is running."
  echo "PID=${active_pid}"
  echo "PID_FILE=${PID_FILE}"
  if [[ -n "${saved_log_path}" ]]; then
    echo "LOG_PATH=${saved_log_path}"
  fi
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
