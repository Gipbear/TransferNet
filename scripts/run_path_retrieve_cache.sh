#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CKPT="${CKPT:-${PROJECT_DIR}/data/ckpt/WebQSP_run_20260518_2241/model-49-0.7154.pt}"
INPUT_DIR="${INPUT_DIR:-${PROJECT_DIR}/data/input/WebQSP}"
QA_FILE="${QA_FILE:-${INPUT_DIR}/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt}"
CACHE="${CACHE:-${PROJECT_DIR}/data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${PROJECT_DIR}/data/output/WebQSP/path_retrieve_server/paths/tail_blend_beam50_alpha1_lam0.5.jsonl}"
BERT_NAME="${BERT_NAME:-bert-base-uncased}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TOPK="${TOPK:-500}"
PHASE="${1:-all}"

case "${PHASE}" in
  all|dump|search) ;;
  *) echo "[ERROR] phase must be all | dump | search" >&2; exit 1 ;;
esac

run_dump() {
  mkdir -p "$(dirname "${CACHE}")"
  python -m WebQSP.dump_scores \
    --input_dir "${INPUT_DIR}" \
    --ckpt "${CKPT}" \
    --mode test \
    --bert_name "${BERT_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --topk "${TOPK}" \
    --qa_file "${QA_FILE}" \
    --output "${CACHE}"
}

run_search() {
  mkdir -p "$(dirname "${OUTPUT_JSONL}")"
  python scripts/offline_path_search.py \
    --cache "${CACHE}" \
    --input_dir "${INPUT_DIR}" \
    --method tail_blend \
    --alpha_final 1.0 \
    --threshold 0.01 \
    --lambda_val 0.5 \
    --beam_size 50 \
    --output "${OUTPUT_JSONL}"
}

if [[ "${PHASE}" == "all" || "${PHASE}" == "dump" ]]; then
  run_dump
fi
if [[ "${PHASE}" == "all" || "${PHASE}" == "search" ]]; then
  run_search
fi
