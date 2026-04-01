#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_ablation.sh
#
# 消融实验自动化编排脚本（第四章）
#
# 覆盖四组消融实验：
#   Group A: 输出格式消融 (v1 / v2 / v3 / v4)
#   Group B: 训练数据消融 (no_shuffle / no_score / distractor_ratio)
#   Group C: 检索参数消融 (不同 beam/lambda，仅 eval，复用最佳模型)
#   Group D: 路径输入格式消融 (arrow/tuple/chain/nl × MID/name，固定 v2 输出)
#
# 特性：
#   - 三步流程：build_kgcot_dataset → train_sft → eval_faithfulness
#   - 每步断点续跑（跳过已完成的步骤）
#   - 每步打印耗时
#
# 用法：
#   bash scripts/run_ablation.sh                            # 全量运行
#   bash scripts/run_ablation.sh --dataset metaqa          # 使用 MetaQA_KB 数据
#   bash scripts/run_ablation.sh --dataset cwq             # 使用 CWQ 数据
#   bash scripts/run_ablation.sh --model_dataset webqsp    # 优先使用 WebQSP 模型
#   bash scripts/run_ablation.sh --group A                  # 只跑 Group A
#   bash scripts/run_ablation.sh --group B                  # 只跑 Group B
#   bash scripts/run_ablation.sh --group C                  # 只跑 Group C（仅 eval）
#   bash scripts/run_ablation.sh --group D                  # 只跑 Group D（路径格式消融）
#   bash scripts/run_ablation.sh --group A --phase train    # 只做数据构建 + 训练
#   bash scripts/run_ablation.sh --group A --phase eval     # 只做推理评估
#   BEST_ADAPTER=models/my_adapter bash scripts/run_ablation.sh --group C
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/run_ablation_lib.sh"

# 多轮 shuffle 推理次数（每轮用不同种子，report mean±std）；设为 1 关闭多轮
NUM_RUNS="${NUM_RUNS:-3}"

BUILD_SCRIPT="${PROJECT_DIR}/llm_infer/build_kgcot_dataset.py"
TRAIN_SCRIPT="${PROJECT_DIR}/llm_infer/train_sft.py"
EVAL_SCRIPT="${PROJECT_DIR}/llm_infer/eval_faithfulness.py"

# 解析 --group / --phase / --dataset / --model_dataset 参数
RUN_GROUP="ALL"
RUN_PHASE="all"   # all | train | eval
RUN_DATASET="webqsp"
MODEL_DATASET="webqsp"
PREV=""
for arg in "$@"; do
    if [[ "$PREV" == "--group" ]]; then
        RUN_GROUP="$arg"
    elif [[ "$PREV" == "--phase" ]]; then
        RUN_PHASE="$arg"
    elif [[ "$PREV" == "--dataset" ]]; then
        RUN_DATASET="$arg"
    elif [[ "$PREV" == "--model_dataset" ]]; then
        MODEL_DATASET="$arg"
    fi
    PREV="$arg"
done
if [[ "$RUN_PHASE" != "all" && "$RUN_PHASE" != "train" && "$RUN_PHASE" != "eval" ]]; then
    echo "[ERROR] --phase 仅支持: all | train | eval"
    exit 1
fi

init_dataset_context "${PROJECT_DIR}" "${RUN_DATASET}"

# ── 路径配置（由 --dataset / --model_dataset 决定）───────────────────────────
TRAIN_INPUT="${DATASET_TRAIN_INPUT}"
PATHS_DIR="${DATASET_PATHS_DIR}"
ABLATION_DATA="${DATASET_ABLATION_DATA}"
ABLATION_MODELS="${DATASET_ABLATION_MODELS}"
TEST_BEAM20_LAM02="${DATASET_TEST_BEAM20_LAM02}"
BASELINE_ADAPTER="$(resolve_baseline_adapter "${PROJECT_DIR}" "${MODEL_DATASET}")"
EVAL_LIMIT="${EVAL_LIMIT:-$(resolve_eval_limit "${RUN_DATASET}")}"
# Group C 使用的 adapter，可通过环境变量覆盖
BEST_ADAPTER="${BEST_ADAPTER:-${BASELINE_ADAPTER}}"

# ── 辅助函数 ──────────────────────────────────────────────────────────────────

log_section() {
    echo ""
    echo "======================================================"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================"
}

log_step() {
    echo ""
    echo "  --------------------------------------------------"
    echo "  $1"
    echo "  --------------------------------------------------"
}

# run_experiment CONFIG_NAME FORMAT BUILD_EXTRA_FLAGS EVAL_EXTRA_FLAGS
#   CONFIG_NAME      : 实验标识，决定数据和模型目录
#   FORMAT           : v1/v2/v3/v4（同时用于 --format 和 --output_format）
#   BUILD_EXTRA_FLAGS: 额外的 build_kgcot_dataset.py 参数（如 --no_shuffle）
#   EVAL_EXTRA_FLAGS : 额外的 eval_faithfulness.py 参数（如 --no_score）
run_experiment() {
    local config_name="$1"
    local fmt="$2"
    local build_extra="${3:-}"
    local eval_extra="${4:-}"

    local data_dir="${ABLATION_DATA}/${config_name}"
    local model_dir="${ABLATION_MODELS}/${config_name}"
    local dataset="${data_dir}/kgcot_train.jsonl"
    local adapter_flag="${model_dir}/adapter_config.json"
    local eval_adapter="${model_dir}"
    # eval 输出文件名由 eval_faithfulness.py 自动生成
    local eval_log="${data_dir}/beam20_lam0.2_${fmt}_ft_eval.log"

    mkdir -p "${data_dir}" "${model_dir}"

    log_section "实验: ${config_name} (format=${fmt})"

    # ── Step 1: 构建训练数据 ─────────────────────────────────────────────────
    if [[ "${RUN_PHASE}" == "eval" ]]; then
        echo "[SKIP] phase=eval，跳过数据构建"
    elif [[ -f "${dataset}" ]]; then
        echo "[SKIP] 数据集已存在: ${dataset}"
    else
        log_step "Step 1/3: 构建训练数据"
        T0=$(date +%s)
        # shellcheck disable=SC2086
        python "${BUILD_SCRIPT}" \
            --input  "${TRAIN_INPUT}" \
            --output "${dataset}" \
            --format "${fmt}" \
            ${build_extra}
        echo "[INFO] 构建完成，耗时 $(($(date +%s) - T0))s"
    fi

    # ── Step 2: QLoRA 训练 ───────────────────────────────────────────────────
    if [[ "${RUN_PHASE}" == "eval" ]]; then
        echo "[SKIP] phase=eval，跳过训练"
    elif [[ -f "${adapter_flag}" ]]; then
        echo "[SKIP] 模型已存在: ${model_dir}"
    else
        log_step "Step 2/3: QLoRA 训练"
        T0=$(date +%s)
        python "${TRAIN_SCRIPT}" \
            --train      "${dataset}" \
            --output_dir "${model_dir}" \
            --epochs     5
        echo "[INFO] 训练完成，耗时 $(($(date +%s) - T0))s"
    fi

    # ── Step 3: 评估 ─────────────────────────────────────────────────────────
    if [[ "${RUN_PHASE}" == "train" ]]; then
        echo "[SKIP] phase=train，跳过评估"
    elif [[ -f "${eval_log}" ]]; then
        echo "[SKIP] 评估结果已存在: ${eval_log}"
    else
        if [[ "${RUN_PHASE}" == "eval" ]]; then
            eval_adapter="$(resolve_slot_adapter "${PROJECT_DIR}" "${MODEL_DATASET}" "${config_name}")"
        fi
        log_step "Step 3/3: 忠实度评估"
        echo "[INFO] 评估 adapter: ${eval_adapter}"
        T0=$(date +%s)
        # shellcheck disable=SC2086
        python "${EVAL_SCRIPT}" \
            --input         "${TEST_BEAM20_LAM02}" \
            --output        "${data_dir}" \
            --adapter       "${eval_adapter}" \
            --output_format "${fmt}" \
            --num_runs      "${NUM_RUNS}" \
            --limit         "${EVAL_LIMIT}" \
            ${eval_extra}
        echo "[INFO] 评估完成，耗时 $(($(date +%s) - T0))s"
    fi
}

# run_eval_only CONFIG_NAME ADAPTER TEST_INPUT FORMAT EVAL_EXTRA_FLAGS
#   用于 Group C（仅 eval）和基线复用
run_eval_only() {
    local config_name="$1"
    local adapter="$2"
    local test_input="$3"
    local fmt="$4"
    local eval_extra="${5:-}"

    if [[ "${RUN_PHASE}" == "train" ]]; then
        echo "[SKIP] phase=train，跳过评估: ${config_name}"
        return
    fi

    local out_dir="${ABLATION_DATA}/${config_name}"
    local stem
    stem="$(basename "${test_input}" .jsonl)"
    local eval_log="${out_dir}/${stem}_${fmt}_ft_eval.log"

    mkdir -p "${out_dir}"

    if [[ -f "${eval_log}" ]]; then
        echo "[SKIP] 评估结果已存在: ${eval_log}"
        return
    fi

    log_step "评估: ${config_name}  [$(basename "${test_input}")] format=${fmt}"
    echo "[INFO] 评估 adapter: ${adapter}"
    T0=$(date +%s)
    # shellcheck disable=SC2086
    python "${EVAL_SCRIPT}" \
        --input         "${test_input}" \
        --output        "${out_dir}" \
        --adapter       "${adapter}" \
        --output_format "${fmt}" \
        --num_runs      "${NUM_RUNS}" \
        --limit         "${EVAL_LIMIT}" \
        ${eval_extra}
    echo "[INFO] 评估完成，耗时 $(($(date +%s) - T0))s"
}

# ── 入口检查 ──────────────────────────────────────────────────────────────────
echo "======================================================"
echo "  消融实验编排脚本"
echo "  PROJECT_DIR   : ${PROJECT_DIR}"
echo "  DATASET       : ${RUN_DATASET}"
echo "  MODEL_DATASET : ${MODEL_DATASET}"
echo "  TRAIN_INPUT   : ${TRAIN_INPUT}"
echo "  PATHS_DIR     : ${PATHS_DIR}"
echo "  ABLATION_DATA : ${ABLATION_DATA}"
echo "  ABLATION_MODELS: ${ABLATION_MODELS}"
echo "  BASELINE_ADAPTER: ${BASELINE_ADAPTER}"
echo "  BEST_ADAPTER  : ${BEST_ADAPTER}"
echo "  EVAL_LIMIT    : ${EVAL_LIMIT}"
echo "  RUN_GROUP     : ${RUN_GROUP}"
echo "  RUN_PHASE     : ${RUN_PHASE}"
echo "  NUM_RUNS      : ${NUM_RUNS}"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

if [[ "${RUN_PHASE}" != "eval" && ! -f "${TRAIN_INPUT}" ]]; then
    echo "[ERROR] 训练数据不存在: ${TRAIN_INPUT}"
    exit 1
fi
if [[ ! -f "${TEST_BEAM20_LAM02}" ]]; then
    echo "[ERROR] 测试集不存在: ${TEST_BEAM20_LAM02}"
    exit 1
fi
if [[ ! -d "${BASELINE_ADAPTER}" ]]; then
    echo "[ERROR] 基线 adapter 不存在: ${BASELINE_ADAPTER}"
    exit 1
fi

WALL_START=$(date +%s)

# ── Group A: 输出格式消融 ─────────────────────────────────────────────────────
if [[ "${RUN_GROUP}" == "ALL" || "${RUN_GROUP}" == "A" ]]; then
    log_section "Group A: 输出格式消融 (v1 / v2 / v3 / v4 / v5)"

    # v1: answer-only，无 citation
    run_experiment "groupA_v1" "v1" "" ""

    # v2: 基线，直接复用已有 adapter 做 eval
    log_section "实验: groupA_v2 (baseline, format=v2)"
    run_eval_only "groupA_v2" "${BASELINE_ADAPTER}" "${TEST_BEAM20_LAM02}" "v2" ""

    # v3: JSON 格式
    run_experiment "groupA_v3" "v3" "" ""

    # v4: CoT 推理链
    run_experiment "groupA_v4" "v4" "" ""

    # v5: Natural Language Path 输入（输出格式同 v2，路径用自然语言表示）
    run_experiment "groupA_v5" "v5" "" ""
fi

# ── Group B: 训练数据消融 ─────────────────────────────────────────────────────
if [[ "${RUN_GROUP}" == "ALL" || "${RUN_GROUP}" == "B" ]]; then
    log_section "Group B: 训练数据消融 (固定 v2 格式)"

    # 关闭路径顺序打乱
    run_experiment "groupB_noshuffle" "v2" "--no_shuffle" ""

    # 路径字符串不含 score（现为默认行为，无需额外参数）
    run_experiment "groupB_noscore" "v2" "" ""

    # 干扰路径比例 0.3
    run_experiment "groupB_dist0.3" "v2" "--distractor_ratio 0.3" ""

    # 干扰路径比例 0.5
    run_experiment "groupB_dist0.5" "v2" "--distractor_ratio 0.5" ""
fi

# ── Group C: 检索参数消融 ─────────────────────────────────────────────────────
if [[ "${RUN_GROUP}" == "ALL" || "${RUN_GROUP}" == "C" ]]; then
    log_section "Group C: 检索参数消融 (adapter: ${BEST_ADAPTER})"

    if [[ ! -d "${BEST_ADAPTER}" ]]; then
        echo "[ERROR] BEST_ADAPTER 不存在: ${BEST_ADAPTER}"
        exit 1
    fi

    # 固定 lambda=0.2，扫 beam
    for beam in 5 10 15 30; do
        test_file="${PATHS_DIR}/beam${beam}_lam0.2.jsonl"
        if [[ ! -f "${test_file}" ]]; then
            echo "[WARN] 测试集不存在，跳过: ${test_file}"
            continue
        fi
        run_eval_only "groupC" "${BEST_ADAPTER}" "${test_file}" "v2" ""
    done

    # beam20_lam0.2 基线（通常已存在于 grid_search/paths/，复制进 groupC 以便汇总）
    run_eval_only "groupC" "${BEST_ADAPTER}" "${TEST_BEAM20_LAM02}" "v2" ""

    # 固定 beam=20，扫 lambda（跳过 lam0.2 已处理）
    for lam in 0.0 0.5 0.7 1.0; do
        test_file="${PATHS_DIR}/beam20_lam${lam}.jsonl"
        if [[ ! -f "${test_file}" ]]; then
            echo "[WARN] 测试集不存在，跳过: ${test_file}"
            continue
        fi
        run_eval_only "groupC" "${BEST_ADAPTER}" "${test_file}" "v2" ""
    done
fi

# ── Group D: 路径输入格式消融 ─────────────────────────────────────────────────
if [[ "${RUN_GROUP}" == "ALL" || "${RUN_GROUP}" == "D" ]]; then
    log_section "Group D: 路径输入格式消融 (3 格式 × 2 实体表示，固定 v2 输出)"

    ENTITY_MAP="${PROJECT_DIR}/data/resources/WebQSP/fbwq_full/mapped_entities.txt"

    if [[ ! -f "${ENTITY_MAP}" ]]; then
        echo "[ERROR] 实体映射文件不存在: ${ENTITY_MAP}"
        exit 1
    fi

    # D-arrow-mid: 复用基线 adapter（与 GroupA v2 完全相同）
    log_section "实验: groupD_arrow_mid (arrow+MID, baseline eval)"
    run_eval_only "groupD_arrow_mid" "${BASELINE_ADAPTER}" "${TEST_BEAM20_LAM02}" "v2" \
        "--path_format arrow"

    # D-arrow-name: arrow 格式 + 实体名称
    run_experiment "groupD_arrow_name" "v2" \
        "--path_format arrow --entity_map ${ENTITY_MAP}" \
        "--path_format arrow --entity_map ${ENTITY_MAP}"

    # D-tuple/chain/nl × MID/name
    for pfmt in tuple chain nl; do
        # MID 变体
        run_experiment "groupD_${pfmt}_mid" "v2" \
            "--path_format ${pfmt}" \
            "--path_format ${pfmt}"

        # Name 变体
        run_experiment "groupD_${pfmt}_name" "v2" \
            "--path_format ${pfmt} --entity_map ${ENTITY_MAP}" \
            "--path_format ${pfmt} --entity_map ${ENTITY_MAP}"
    done
fi

# ── 完成 ──────────────────────────────────────────────────────────────────────
WALL_END=$(date +%s)
echo ""
echo "======================================================"
echo "  全部实验完成"
echo "  总耗时: $((WALL_END - WALL_START))s"
echo "  结果目录: ${ABLATION_DATA}"
echo ""
echo "  汇总结果："
echo "    python scripts/collect_ablation_results.py"
echo "======================================================"
