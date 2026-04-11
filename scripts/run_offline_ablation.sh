#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_offline_ablation.sh
#
# 对 offline_search/paths/ 下的路径 JSONL 运行 eval_faithfulness，
# 跑两个 variant：chain+name (groupAname_v2) 和 chain+MID (groupAmid_v2)
# 结果写入 data/output/WebQSP/offline_ablation/{name,mid}/，断点续跑。
#
# 用法：
#   # 评估单个文件
#   bash scripts/run_offline_ablation.sh \
#       --input data/output/WebQSP/offline_search/paths/beam20_lam0.2.jsonl
#
#   # 按 beam/lambda 组合自动定位文件
#   bash scripts/run_offline_ablation.sh --beam 20 --lam 0.2
#
#   # 批量扫描 offline_search/paths/ 下所有 JSONL
#   bash scripts/run_offline_ablation.sh --all
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJ_DIR}/scripts/run_ablation_lib.sh"

PATHS_DIR="${PROJ_DIR}/data/output/WebQSP/offline_search/paths"
OUTPUT_ROOT="${PROJ_DIR}/data/output/WebQSP/offline_ablation"
EVAL_SCRIPT="${PROJ_DIR}/llm_infer/eval_faithfulness.py"
ENTITY_MAP="${PROJ_DIR}/data/resources/WebQSP/fbwq_full/mapped_entities.txt"
MODEL_DATASET="webqsp"
NUM_RUNS="${NUM_RUNS:-2}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"

SCAN_ALL=0
EXPLICIT_INPUTS=()
BEAM_VALS=()
LAM_VALS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)         EXPLICIT_INPUTS+=("$2"); shift 2 ;;
        --paths_dir)     PATHS_DIR="$2";          shift 2 ;;
        --output_root)   OUTPUT_ROOT="$2";        shift 2 ;;
        --model_dataset) MODEL_DATASET="$2";      shift 2 ;;
        --num_runs)      NUM_RUNS="$2";           shift 2 ;;
        --limit)         EVAL_LIMIT="$2";         shift 2 ;;
        --beam)          BEAM_VALS+=("$2");       shift 2 ;;
        --lam)           LAM_VALS+=("$2");        shift 2 ;;
        --all)           SCAN_ALL=1;              shift 1 ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
done

# ── 解析两个 adapter ──────────────────────────────────────────────────────────
ADAPTER_NAME="$(resolve_slot_adapter "${PROJ_DIR}" "${MODEL_DATASET}" "groupAname_v2")"
ADAPTER_MID="$(resolve_slot_adapter  "${PROJ_DIR}" "${MODEL_DATASET}" "groupAmid_v2")"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# eval_one INPUT VARIANT ADAPTER EXTRA_ARGS...
eval_one() {
    local input="$1"
    local variant="$2"   # name | mid
    local adapter="$3"
    shift 3
    local extra_args=("$@")

    local stem out_dir eval_log
    stem="$(basename "${input}" .jsonl)"
    out_dir="${OUTPUT_ROOT}/${variant}"
    eval_log="${out_dir}/${stem}_v2_ft_eval.log"

    if [[ ! -f "${input}" ]]; then
        echo "[WARN] 文件不存在，跳过: ${input}"; return 0
    fi
    if [[ -f "${eval_log}" ]]; then
        echo "[SKIP] ${variant}: ${eval_log}"; return 0
    fi

    echo ""
    echo "  [$(ts)] ${variant}: $(basename "${input}")"
    mkdir -p "${out_dir}"

    local limit_args=()
    [[ "${EVAL_LIMIT}" -gt 0 ]] && limit_args+=(--limit "${EVAL_LIMIT}")

    local T0; T0=$(date +%s)
    python "${EVAL_SCRIPT}" \
        --input         "${input}" \
        --output        "${out_dir}" \
        --adapter       "${adapter}" \
        --output_format "v2" \
        --path_format   "chain" \
        --num_runs      "${NUM_RUNS}" \
        "${extra_args[@]}" \
        "${limit_args[@]+"${limit_args[@]}"}"
    echo "  [INFO] 完成，耗时 $(($(date +%s) - T0))s"
}

# ── 构建输入列表 ──────────────────────────────────────────────────────────────
INPUTS=()
for f in "${EXPLICIT_INPUTS[@]+"${EXPLICIT_INPUTS[@]}"}"; do INPUTS+=("$f"); done
if [[ ${#BEAM_VALS[@]} -gt 0 && ${#LAM_VALS[@]} -gt 0 ]]; then
    for beam in "${BEAM_VALS[@]}"; do
        for lam in "${LAM_VALS[@]}"; do
            INPUTS+=("${PATHS_DIR}/beam${beam}_lam${lam}.jsonl")
        done
    done
fi
if [[ "${SCAN_ALL}" -eq 1 ]]; then
    while IFS= read -r -d '' f; do INPUTS+=("$f"); done \
        < <(find "${PATHS_DIR}" -maxdepth 1 -name "*.jsonl" -print0 | sort -z)
fi
if [[ ${#INPUTS[@]} -eq 0 ]]; then
    echo "[ERROR] 未指定输入。用 --input / --beam+--lam / --all"
    exit 1
fi

# ── 主循环 ────────────────────────────────────────────────────────────────────
echo "======================================================"
echo "  offline ablation  (name + mid, v2)"
echo "  output_root : ${OUTPUT_ROOT}"
echo "  adapter_name: ${ADAPTER_NAME}"
echo "  adapter_mid : ${ADAPTER_MID}"
echo "  文件数      : ${#INPUTS[@]}"
echo "  $(ts)"
echo "======================================================"

for a in "${ADAPTER_NAME}" "${ADAPTER_MID}"; do
    if [[ ! -d "$a" ]]; then
        echo "[ERROR] adapter 不存在: $a"; exit 1
    fi
done

WALL_START=$(date +%s)
for input in "${INPUTS[@]}"; do
    eval_one "${input}" "name" "${ADAPTER_NAME}" --entity_map "${ENTITY_MAP}"
    eval_one "${input}" "mid"  "${ADAPTER_MID}"
done

echo ""
echo "======================================================"
echo "  全部完成，总耗时 $(($(date +%s) - WALL_START))s"
echo "  name 结果: ${OUTPUT_ROOT}/name/"
echo "  mid  结果: ${OUTPUT_ROOT}/mid/"
echo "======================================================"
