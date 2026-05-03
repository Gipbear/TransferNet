#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

INPUT="${INPUT:-data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt}"
OUTPUT="${OUTPUT:-data/output/WebQSP/checked_batch_agent/full.jsonl}"
LIMIT="${LIMIT:-0}"

./scripts/path_retrieve_server.sh status
./scripts/llm_server.sh status

CMD=(
  conda run -n py312_t271_cuda python -m oh_my_agent.cli.eval_checked_batch_agent
  --input "${INPUT}"
  --output "${OUTPUT}"
)

if [[ "${LIMIT}" != "0" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

echo "[INFO] Running checked-batch agent eval"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

SUMMARY="${OUTPUT%.*}_summary.json"
echo "[INFO] JSONL   : ${OUTPUT}"
echo "[INFO] Summary : ${SUMMARY}"
