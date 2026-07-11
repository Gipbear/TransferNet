#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_DIR}/scripts/run_ablation_lib.sh"

TMP_ROOT="$(mktemp -d)"
TMP_ROOT_LEGACY="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT
trap 'rm -rf "${TMP_ROOT_LEGACY}"' EXIT

mkdir -p "${TMP_ROOT}/data/output/WebQSP/grid_search/paths"
mkdir -p "${TMP_ROOT}/data/output/MetaQA_KB/grid_search/paths"
mkdir -p "${TMP_ROOT}/data/output/CWQ/grid_search/paths"
mkdir -p "${TMP_ROOT}/models/webqsp/ablation/slot_fallback"
mkdir -p "${TMP_ROOT}/models/webqsp/ablation/groupB_noshuffle"

touch "${TMP_ROOT}/data/output/WebQSP/predict_train.jsonl"
touch "${TMP_ROOT}/data/output/MetaQA_KB/predict_train.jsonl"
touch "${TMP_ROOT}/data/output/CWQ/predict_train.jsonl"
touch "${TMP_ROOT}/data/output/WebQSP/grid_search/paths/beam20_lam0.2.jsonl"
touch "${TMP_ROOT}/data/output/MetaQA_KB/grid_search/paths/beam20_lam0.2.jsonl"
touch "${TMP_ROOT}/data/output/CWQ/grid_search/paths/beam20_lam0.2.jsonl"

mkdir -p "${TMP_ROOT_LEGACY}/models/ablation/groupA_v3"

assert_eq() {
    local actual="$1"
    local expected="$2"
    local label="$3"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "[FAIL] ${label}" >&2
        echo "  expected: ${expected}" >&2
        echo "  actual  : ${actual}" >&2
        exit 1
    fi
}

if init_dataset_context "${TMP_ROOT}" "unknown" 2>/dev/null; then
    echo "[FAIL] unknown dataset should fail" >&2
    exit 1
fi

init_dataset_context "${TMP_ROOT}" "metaqa"
assert_eq "${DATASET_OUTPUT_ROOT}" "${TMP_ROOT}/data/output/MetaQA_KB" "metaqa output root"
assert_eq "${DATASET_KEY}" "metaqa" "metaqa dataset key"

slot_adapter="$(resolve_slot_adapter "${TMP_ROOT}" "metaqa" "slot_fallback")"
assert_eq "${slot_adapter}" "${TMP_ROOT}/models/webqsp/ablation/slot_fallback" "fallback to webqsp slot_fallback"

slot_adapter_b="$(resolve_slot_adapter "${TMP_ROOT}" "cwq" "groupB_noshuffle")"
assert_eq "${slot_adapter_b}" "${TMP_ROOT}/models/webqsp/ablation/groupB_noshuffle" "fallback to webqsp groupB_noshuffle"

legacy_slot="$(resolve_slot_adapter "${TMP_ROOT_LEGACY}" "metaqa" "groupA_v3")"
assert_eq "${legacy_slot}" "${TMP_ROOT_LEGACY}/models/ablation/groupA_v3" "legacy ablation fallback"

assert_eq "$(resolve_eval_limit "webqsp")" "0" "webqsp eval limit"
assert_eq "$(resolve_eval_limit "cwq")" "0" "cwq eval limit"
assert_eq "$(resolve_eval_limit "metaqa")" "100" "metaqa eval limit"

echo "[PASS] run_ablation_lib"
