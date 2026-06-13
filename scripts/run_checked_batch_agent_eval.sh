#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

INPUT="${INPUT:-data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-data/output/WebQSP/checked_batch_agent/full_$(date +%Y%m%d_%H%M)}"
LIMIT="${LIMIT:-0}"
DEDUPE_TAIL_PATHS="${DEDUPE_TAIL_PATHS:-0}"
BEAM_SIZE="${BEAM_SIZE:-50}"
BATCH_SIZE="${BATCH_SIZE:-20}"
LAMBDA_VAL="${LAMBDA_VAL:-0.2}"
ALPHA_FINAL="${ALPHA_FINAL:-1.0}"
# 后处理(默认即 docs §1 配置;布尔项用 =0 关闭)
CHECK_MODE="${CHECK_MODE:-hybrid-reject-list}"
SCORE_MARGIN="${SCORE_MARGIN:-4.0}"
EXPANSION_TOP_GROUPS="${EXPANSION_TOP_GROUPS:-2}"
CHECK_CONSTRAINED_DECODING="${CHECK_CONSTRAINED_DECODING:-1}"
HOP_FILTER="${HOP_FILTER:-1}"
LARGE_ANSWER_EXPANSION="${LARGE_ANSWER_EXPANSION:-1}"
PATH_RETRIEVE_PORT="${PATH_RETRIEVE_PORT:-8789}"
LLM_SERVER_PORT="${LLM_SERVER_PORT:-8788}"
PATH_RETRIEVE_URL="${PATH_RETRIEVE_URL:-http://localhost:${PATH_RETRIEVE_PORT}}"
LLM_SERVER_URL="${LLM_SERVER_URL:-http://localhost:${LLM_SERVER_PORT}}"

ensure_service() {
  local name="$1"
  local script="$2"
  local port="$3"
  local base_url="$4"
  local health_url="${base_url%/}/health"

  if curl -sf "${health_url}" >/dev/null 2>&1; then
    echo "[INFO] ${name} is ready: ${health_url}"
    PORT="${port}" "${script}" status
    return
  fi

  echo "[INFO] ${name} is not ready; starting it on port ${port}."
  PORT="${port}" "${script}" status || true
  PORT="${port}" "${script}" start

  if ! curl -sf "${health_url}" >/dev/null 2>&1; then
    echo "[ERROR] ${name} failed health check after start: ${health_url}" >&2
    exit 1
  fi
}

ensure_service "path_retrieve_server" "./scripts/path_retrieve_server.sh" "${PATH_RETRIEVE_PORT}" "${PATH_RETRIEVE_URL}"
ensure_service "llm_server" "./scripts/llm_server.sh" "${LLM_SERVER_PORT}" "${LLM_SERVER_URL}"

CMD=(
  python -m oh_my_agent.cli.eval_checked_batch_agent
  --input "${INPUT}"
  --output "${OUTPUT_DIR}"
  --beam_size "${BEAM_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --lambda_val "${LAMBDA_VAL}"
  --alpha_final "${ALPHA_FINAL}"
  --check_mode "${CHECK_MODE}"
  --score_margin "${SCORE_MARGIN}"
  --expansion_top_groups "${EXPANSION_TOP_GROUPS}"
  --path_retrieve_url "${PATH_RETRIEVE_URL}"
  --llm_server_url "${LLM_SERVER_URL}"
)

if [[ "${CHECK_CONSTRAINED_DECODING}" != "0" ]]; then
  CMD+=(--check_constrained_decoding)
fi

if [[ "${HOP_FILTER}" != "0" ]]; then
  CMD+=(--hop_filter)
fi

if [[ "${LARGE_ANSWER_EXPANSION}" != "0" ]]; then
  CMD+=(--large_answer_expansion)
fi

if [[ "${DEDUPE_TAIL_PATHS}" != "0" ]]; then
  CMD+=(--dedupe_tail_paths)
fi

if [[ "${LIMIT}" != "0" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

echo "[INFO] Running checked-batch agent eval"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "[INFO] Output dir        : ${OUTPUT_DIR}"
echo "[INFO] JSONL             : ${OUTPUT_DIR}/checked_batch_eval.jsonl"
echo "[INFO] Summary           : ${OUTPUT_DIR}/checked_batch_eval_summary.json"
echo "[INFO] Initial retrieval : ${OUTPUT_DIR}/initial_retrieval.jsonl"
echo "[INFO] Initial answer    : ${OUTPUT_DIR}/initial_answer.jsonl"
