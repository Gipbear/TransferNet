#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/vllm_server.sh setup-env
#   ./scripts/vllm_server.sh start
#   ./scripts/vllm_server.sh stop
#   ./scripts/vllm_server.sh status
#
# Defaults are tuned for the merged WebQSP Unsloth adapter path. Override with:
#   VLLM_MODEL_PATH=/path/to/model VLLM_SERVED_MODEL_NAME=webqsp-agent ./scripts/vllm_server.sh start

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-start}"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-vllm_server}"
VLLM_PYTHON_VERSION="${VLLM_PYTHON_VERSION:-3.11}"
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-${PROJECT_DIR}/models/webqsp/ablation/groupJ_schema_name_merged_16bit}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8788}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-webqsp-agent}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---generation-config vllm}"
VLLM_INSTALL_CMD="${VLLM_INSTALL_CMD:-python -m pip install -U vllm 'bitsandbytes>=0.48.1'}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/data/output/WebQSP/vllm_server}"
LOG_PATH="${LOG_PATH:-}"
PID_FILE="${PID_FILE:-/tmp/vllm_server_${VLLM_PORT}.pid}"
LOG_PATH_FILE="${LOG_PATH_FILE:-/tmp/vllm_server_${VLLM_PORT}.logpath}"
WAIT_FOR_HEALTH="${WAIT_FOR_HEALTH:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-180}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/vllm_server.sh setup-env
  ./scripts/vllm_server.sh start
  ./scripts/vllm_server.sh stop
  ./scripts/vllm_server.sh restart
  ./scripts/vllm_server.sh status

Environment:
  VLLM_CONDA_ENV             Conda env name (default: vllm_server)
  VLLM_MODEL_PATH            Model directory or HF id
  VLLM_SERVED_MODEL_NAME     OpenAI-compatible model name
  VLLM_PORT                  Server port (default: 8788)
  VLLM_EXTRA_ARGS            Extra arguments appended to vllm serve
  VLLM_INSTALL_CMD           Install command run inside the conda env
EOF
}

load_conda() {
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[ERROR] Conda activation script not found: ${CONDA_SH}" >&2
    exit 1
  fi
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  set -u
}

activate_vllm_env() {
  set +u
  conda activate "${VLLM_CONDA_ENV}"
  set -u
}

setup_env() {
  load_conda
  if ! conda env list | awk '{print $1}' | grep -Fxq "${VLLM_CONDA_ENV}"; then
    conda create -n "${VLLM_CONDA_ENV}" "python=${VLLM_PYTHON_VERSION}" -y
  fi
  activate_vllm_env
  python -m pip install --upgrade pip
  eval "${VLLM_INSTALL_CMD}"
}

find_server_pids() {
  pgrep -f "vllm .*serve .*--port ${VLLM_PORT}" || true
}

is_hf_model_id() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
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
    [[ "${alive}" == "0" ]] && return 0
    sleep 1
  done
}

resolve_log_path() {
  if [[ -n "${LOG_PATH}" ]]; then
    mkdir -p "$(dirname "${LOG_PATH}")"
    return
  fi
  mkdir -p "${LOG_ROOT}"
  local timestamp
  timestamp="$(date '+%Y%m%d_%H%M')"
  LOG_PATH="${LOG_ROOT}/${timestamp}_port${VLLM_PORT}_server.log"
  local suffix=1
  while [[ -e "${LOG_PATH}" ]]; do
    LOG_PATH="${LOG_ROOT}/${timestamp}_port${VLLM_PORT}_server_${suffix}.log"
    suffix=$((suffix + 1))
  done
}

start_server() {
  if [[ ! -d "${VLLM_MODEL_PATH}" ]] && ! is_hf_model_id "${VLLM_MODEL_PATH}"; then
    echo "[ERROR] Model path not found and value does not look like a HF model id: ${VLLM_MODEL_PATH}" >&2
    exit 1
  fi

  load_conda
  activate_vllm_env
  resolve_log_path
  printf '%s\n' "${LOG_PATH}" > "${LOG_PATH_FILE}"

  cd "${PROJECT_DIR}"
  setsid nohup vllm serve "${VLLM_MODEL_PATH}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --served-model-name "${VLLM_SERVED_MODEL_NAME}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    ${VLLM_EXTRA_ARGS} \
    > "${LOG_PATH}" 2>&1 &

  local new_pid
  new_pid=$!
  printf '%s\n' "${new_pid}" > "${PID_FILE}"
  echo "[INFO] Started vLLM PID=${new_pid} port=${VLLM_PORT}"
  echo "[INFO] Model path : ${VLLM_MODEL_PATH}"
  echo "[INFO] Model name : ${VLLM_SERVED_MODEL_NAME}"
  echo "[INFO] Log path   : ${LOG_PATH}"

  if [[ "${WAIT_FOR_HEALTH}" != "1" ]]; then
    return
  fi

  local health_url
  health_url="http://127.0.0.1:${VLLM_PORT}/v1/models"
  for _ in $(seq 1 "${WAIT_SECONDS}"); do
    if curl -sf "${health_url}" >/dev/null 2>&1; then
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
  local pids
  pids="$(find_server_pids)"
  if [[ -z "${pids}" && -f "${PID_FILE}" ]]; then
    pids="$(<"${PID_FILE}")"
  fi
  if [[ -z "${pids}" ]]; then
    echo "[INFO] No vLLM process found on port ${VLLM_PORT}."
    rm -f "${PID_FILE}"
    return
  fi
  echo "[INFO] Stopping vLLM PID(s): $(echo "${pids}" | tr '\n' ' ')"
  stop_pids "${pids}"
  rm -f "${PID_FILE}"
}

status_server() {
  local pid=""
  [[ -f "${PID_FILE}" ]] && pid="$(<"${PID_FILE}")"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    echo "[INFO] vLLM is not running on port ${VLLM_PORT}."
    [[ -f "${LOG_PATH_FILE}" ]] && echo "LAST_LOG_PATH=$(<"${LOG_PATH_FILE}")"
    echo "HEALTH_URL=http://127.0.0.1:${VLLM_PORT}/v1/models"
    return
  fi
  echo "[INFO] vLLM is running."
  echo "PID=${pid}"
  [[ -f "${LOG_PATH_FILE}" ]] && echo "LOG_PATH=$(<"${LOG_PATH_FILE}")"
  ps -p "${pid}" -o pid,etime,stat,cmd
  curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" || true
  echo
}

case "${ACTION}" in
  setup-env)
    setup_env
    ;;
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
