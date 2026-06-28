#!/usr/bin/env bash
# Stop-policy exploration for oh_my_agent checked-batch evaluation.
#
# Preferred usage when an adequate trace already exists (offline only):
#   SOURCE_DIR=data/output/WebQSP/checked_batch_agent/full_20260613_1722 \
#     bash scripts/run_stop_policy_experiments.sh
#
# To collect a full trace first (expensive: calls path/LLM/check services):
#   RUN_FULL_TRACE=1 LIMIT=50 bash scripts/run_stop_policy_experiments.sh
#
# The sweep itself never calls LLM/path services. It replays recorded traces and
# marks policies unsupported if the source stopped before the policy would stop.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

INPUT="${INPUT:-data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/output/WebQSP/checked_batch_agent/stop_policy_$(date +%Y%m%d_%H%M)}"
SOURCE_DIR="${SOURCE_DIR:-}"
RUN_FULL_TRACE="${RUN_FULL_TRACE:-0}"
LIMIT="${LIMIT:-0}"
BEAM_SIZE="${BEAM_SIZE:-50}"
BATCH_SIZE="${BATCH_SIZE:-20}"
LAMBDA_VAL="${LAMBDA_VAL:-0.2}"
ALPHA_FINAL="${ALPHA_FINAL:-1.0}"
SCORE_MARGIN="${SCORE_MARGIN:-4}"
EXPANSION_TOP_GROUPS="${EXPANSION_TOP_GROUPS:-2}"
PATH_RETRIEVE_URL="${PATH_RETRIEVE_URL:-http://localhost:8789}"
LLM_SERVER_URL="${LLM_SERVER_URL:-http://localhost:8788}"
ENTITY_MAP="${ENTITY_MAP:-data/resources/WebQSP/fbwq_full/mapped_entities.txt}"

MIXED_STOP_RATIOS="${MIXED_STOP_RATIOS:-0,0.1,1/3,0.5,off}"
MAX_BATCHES="${MAX_BATCHES:-1,2,3,all}"
ALL_WRONG_MODES="${ALL_WRONG_MODES:-on,off}"
NO_NEW_BATCHES="${NO_NEW_BATCHES:-none}"

mkdir -p "${OUTPUT_ROOT}"
echo "[INFO] output_root=${OUTPUT_ROOT}"

if [[ -z "${SOURCE_DIR}" ]]; then
  if [[ "${RUN_FULL_TRACE}" != "1" ]]; then
    cat >&2 <<EOF
[ERROR] SOURCE_DIR is empty and RUN_FULL_TRACE is not 1.
Set SOURCE_DIR to an existing checked_batch_eval directory for offline replay,
or set RUN_FULL_TRACE=1 to create an expensive full trace first.
EOF
    exit 1
  fi
  SOURCE_DIR="${OUTPUT_ROOT}/full_trace"
fi

if [[ "${RUN_FULL_TRACE}" == "1" && ! -f "${SOURCE_DIR}/checked_batch_eval_summary.json" ]]; then
  echo "[RUN ] collecting full trace: ${SOURCE_DIR}"
  cmd=(
    python -m oh_my_agent.cli.eval_checked_batch_agent
    --input "${INPUT}" --output "${SOURCE_DIR}"
    --beam_size "${BEAM_SIZE}" --batch_size "${BATCH_SIZE}"
    --lambda_val "${LAMBDA_VAL}" --alpha_final "${ALPHA_FINAL}"
    --check_mode hybrid-reject-list
    --check_constrained_decoding
    --score_margin "${SCORE_MARGIN}"
    --hop_filter --large_answer_expansion
    --expansion_top_groups "${EXPANSION_TOP_GROUPS}"
    --no_early_stop --no_all_wrong_after_answer_stop
    --path_retrieve_url "${PATH_RETRIEVE_URL}"
    --llm_server_url "${LLM_SERVER_URL}"
    --entity_map "${ENTITY_MAP}"
  )
  [[ "${LIMIT}" != "0" ]] && cmd+=(--limit "${LIMIT}")
  printf ' %q' "${cmd[@]}"; echo
  "${cmd[@]}"
elif [[ "${RUN_FULL_TRACE}" == "1" ]]; then
  echo "[SKIP] full trace already exists: ${SOURCE_DIR}"
fi

SWEEP_DIR="${OUTPUT_ROOT}/stop_policy_sweep"
echo "[RUN ] offline stop-policy sweep from ${SOURCE_DIR}"
python scripts/sweep_stop_policies.py \
  --source_dir "${SOURCE_DIR}" \
  --output_dir "${SWEEP_DIR}" \
  --entity_map "${ENTITY_MAP}" \
  --mixed_stop_ratios "${MIXED_STOP_RATIOS}" \
  --max_batches "${MAX_BATCHES}" \
  --all_wrong_modes "${ALL_WRONG_MODES}" \
  --no_new_batches "${NO_NEW_BATCHES}"

echo "[DONE] summary: ${SWEEP_DIR}/stop_policy_sweep_summary.csv"
