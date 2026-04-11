#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_ablation.sh
#
# 消融实验自动化编排脚本（第四章）
#
# 覆盖五组消融实验：
#   Group A: 输出格式消融 (v1 / v2 / v3 / v4)
#   Group B: 训练数据消融 (no_shuffle / no_score / distractor_ratio)
#   Group C: 检索参数消融 (不同 beam/lambda，仅 eval，复用最佳模型)
#   Group D: 路径输入格式消融 (arrow/tuple/chain/nl × MID/name，固定 v2 输出)
#   Group E: Base Model 零样本评估 (chain × MID/name × v1/v2/v3/v4，无微调)
#   Group F: 拒答能力训练 (chain+v2, 含 Hit@K=0 拒答样本)
#   Group G: 训练轮数消融 (epochs 1-5, chain+name+v2, limit=500)
#   Group H: 无路径基线 (base model 直接回答，无检索路径输入)
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
#   bash scripts/run_ablation.sh --group G                  # 只跑 Group G（训练轮数消融）
#   bash scripts/run_ablation.sh --group H                  # 只跑 Group H（无路径基线）
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/run_ablation_lib.sh"

# 多轮 shuffle 推理次数（每轮用不同种子，report mean±std）；设为 1 关闭多轮
NUM_RUNS="${NUM_RUNS:-2}"
# QLoRA 训练轮数
EPOCHS="${EPOCHS:-2}"

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
# 默认使用 groupAname_v2（chain+name+v2，消融最优配置）
BEST_ADAPTER="${BEST_ADAPTER:-$(resolve_slot_adapter "${PROJECT_DIR}" "${MODEL_DATASET}" "groupAname_v2")}"

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
            --epochs     "${EPOCHS}"
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

# run_base_eval CONFIG_NAME TEST_INPUT FORMAT EVAL_EXTRA_FLAGS
#   零样本评估（不加载 adapter，直接用 base model）
run_base_eval() {
    local config_name="$1"
    local test_input="$2"
    local fmt="$3"
    local eval_extra="${4:-}"

    if [[ "${RUN_PHASE}" == "train" ]]; then
        echo "[SKIP] phase=train，跳过评估: ${config_name}"
        return
    fi

    local out_dir="${ABLATION_DATA}/${config_name}"
    local stem
    stem="$(basename "${test_input}" .jsonl)"
    local eval_log="${out_dir}/${stem}_${fmt}_eval.log"

    mkdir -p "${out_dir}"

    if [[ -f "${eval_log}" ]]; then
        echo "[SKIP] 评估结果已存在: ${eval_log}"
        return
    fi

    log_step "零样本评估: ${config_name}  [$(basename "${test_input}")] format=${fmt}"
    T0=$(date +%s)
    # shellcheck disable=SC2086
    python "${EVAL_SCRIPT}" \
        --input         "${test_input}" \
        --output        "${out_dir}" \
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
echo "  EPOCHS        : ${EPOCHS}"
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

# ── Group Amid: 链式路径 + MID 实体表示，v1-v4 ────────────────────────────────
if [[ "${RUN_GROUP}" == "ALL" || "${RUN_GROUP}" == "Amid" ]]; then
    log_section "Group Amid: 输出格式消融 + chain 路径 + MID 实体 (v1 / v2 / v3 / v4)"

    run_experiment "groupAmid_v1" "v1" "--path_format chain" "--path_format chain"
    run_experiment "groupAmid_v2" "v2" "--path_format chain" "--path_format chain"
    run_experiment "groupAmid_v3" "v3" "--path_format chain" "--path_format chain"
    run_experiment "groupAmid_v4" "v4" "--path_format chain" "--path_format chain"
fi

# ── Group Aname: 链式路径 + name 实体表示，v1-v4 ──────────────────────────────
if [[ "${RUN_GROUP}" == "ALL" || "${RUN_GROUP}" == "Aname" ]]; then
    log_section "Group Aname: 输出格式消融 + chain 路径 + name 实体 (v1 / v2 / v3 / v4)"

    ENTITY_MAP_A="${PROJECT_DIR}/data/resources/WebQSP/fbwq_full/mapped_entities.txt"
    if [[ ! -f "${ENTITY_MAP_A}" ]]; then
        echo "[ERROR] 实体映射文件不存在: ${ENTITY_MAP_A}"
        exit 1
    fi

    run_experiment "groupAname_v1" "v1" \
        "--path_format chain --entity_map ${ENTITY_MAP_A}" \
        "--path_format chain --entity_map ${ENTITY_MAP_A}"
    run_experiment "groupAname_v2" "v2" \
        "--path_format chain --entity_map ${ENTITY_MAP_A}" \
        "--path_format chain --entity_map ${ENTITY_MAP_A}"
    run_experiment "groupAname_v3" "v3" \
        "--path_format chain --entity_map ${ENTITY_MAP_A}" \
        "--path_format chain --entity_map ${ENTITY_MAP_A}"
    run_experiment "groupAname_v4" "v4" \
        "--path_format chain --entity_map ${ENTITY_MAP_A}" \
        "--path_format chain --entity_map ${ENTITY_MAP_A}"
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

    ENTITY_MAP_C="${PROJECT_DIR}/data/resources/WebQSP/fbwq_full/mapped_entities.txt"
    if [[ ! -f "${ENTITY_MAP_C}" ]]; then
        echo "[ERROR] 实体映射文件不存在: ${ENTITY_MAP_C}"
        exit 1
    fi
    # groupAname_v2 以 chain+name 格式训练，评估时需保持一致
    GROUPC_EVAL_EXTRA="--path_format chain --entity_map ${ENTITY_MAP_C}"

    # 固定 lambda=0.2，扫 beam
    for beam in 5 10 15 20 30; do
        test_file="${PATHS_DIR}/beam${beam}_lam0.2.jsonl"
        if [[ ! -f "${test_file}" ]]; then
            echo "[WARN] 测试集不存在，跳过: ${test_file}"
            continue
        fi
        run_eval_only "groupC" "${BEST_ADAPTER}" "${test_file}" "v2" "${GROUPC_EVAL_EXTRA}"
    done

    # beam20_lam0.2 基线（通常已存在于 grid_search/paths/，复制进 groupC 以便汇总）
    run_eval_only "groupC" "${BEST_ADAPTER}" "${TEST_BEAM20_LAM02}" "v2" "${GROUPC_EVAL_EXTRA}"

    # 固定 beam=20，扫 lambda（跳过 lam0.2 已处理）
    for lam in 0.0 0.5 0.7 1.0; do
        test_file="${PATHS_DIR}/beam20_lam${lam}.jsonl"
        if [[ ! -f "${test_file}" ]]; then
            echo "[WARN] 测试集不存在，跳过: ${test_file}"
            continue
        fi
        run_eval_only "groupC" "${BEST_ADAPTER}" "${test_file}" "v2" "${GROUPC_EVAL_EXTRA}"
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

# ── Group E: Base Model 零样本评估 ────────────────────────────────────────────
if [[ "${RUN_GROUP}" == "ALL" || "${RUN_GROUP}" == "E" ]]; then
    log_section "Group E: Base Model 零样本评估 (chain × MID/name × v1/v2/v3/v4)"

    ENTITY_MAP="${PROJECT_DIR}/data/resources/WebQSP/fbwq_full/mapped_entities.txt"

    if [[ ! -f "${ENTITY_MAP}" ]]; then
        echo "[ERROR] 实体映射文件不存在: ${ENTITY_MAP}"
        exit 1
    fi

    for fmt in v1 v2 v3 v4; do
        # chain + MID
        run_base_eval "groupE_chain_mid_${fmt}" "${TEST_BEAM20_LAM02}" "${fmt}" \
            "--path_format chain"
        # chain + name
        run_base_eval "groupE_chain_name_${fmt}" "${TEST_BEAM20_LAM02}" "${fmt}" \
            "--path_format chain --entity_map ${ENTITY_MAP}"
    done
fi

# ── Group F: 拒答能力训练 ─────────────────────────────────────────────────────
if [[ "${RUN_GROUP}" == "ALL" || "${RUN_GROUP}" == "F" ]]; then
    log_section "Group F: 拒答能力训练 (chain+v2, 含 Hit@K=0 拒答样本)"

    ENTITY_MAP="${PROJECT_DIR}/data/resources/WebQSP/fbwq_full/mapped_entities.txt"

    if [[ ! -f "${ENTITY_MAP}" ]]; then
        echo "[ERROR] 实体映射文件不存在: ${ENTITY_MAP}"
        exit 1
    fi

    # F1: chain + MID + rejection
    run_experiment "groupF_chain_mid" "v2" \
        "--path_format chain --include_rejection" \
        "--path_format chain --reject_prompt"

    # F2: chain + name + rejection
    run_experiment "groupF_chain_name" "v2" \
        "--path_format chain --entity_map ${ENTITY_MAP} --include_rejection" \
        "--path_format chain --entity_map ${ENTITY_MAP} --reject_prompt"

    # F3/F4: chain + name + rejection + 拒答样本上采样（仅 name 变体）
    run_experiment "groupF_chain_name_os5" "v2" \
        "--path_format chain --entity_map ${ENTITY_MAP} --include_rejection --rejection_oversample 5" \
        "--path_format chain --entity_map ${ENTITY_MAP} --reject_prompt"

    run_experiment "groupF_chain_name_os10" "v2" \
        "--path_format chain --entity_map ${ENTITY_MAP} --include_rejection --rejection_oversample 10" \
        "--path_format chain --entity_map ${ENTITY_MAP} --reject_prompt"
fi

# ── Group G: 训练轮数消融 ─────────────────────────────────────────────────────
if [[ "${RUN_GROUP}" == "ALL" || "${RUN_GROUP}" == "G" ]]; then
    log_section "Group G: 训练轮数消融 (epochs 1-5, chain+name+v2, limit=500)"

    ENTITY_MAP="${PROJECT_DIR}/data/resources/WebQSP/fbwq_full/mapped_entities.txt"

    if [[ ! -f "${ENTITY_MAP}" ]]; then
        echo "[ERROR] 实体映射文件不存在: ${ENTITY_MAP}"
        exit 1
    fi

    SAVED_EPOCHS="${EPOCHS}"
    SAVED_EVAL_LIMIT="${EVAL_LIMIT}"
    EVAL_LIMIT=500

    for ep in 1 2 3 4 5; do
        EPOCHS="${ep}"
        run_experiment "groupG_epoch${ep}" "v2" \
            "--path_format chain --entity_map ${ENTITY_MAP}" \
            "--path_format chain --entity_map ${ENTITY_MAP}"
    done

    EPOCHS="${SAVED_EPOCHS}"
    EVAL_LIMIT="${SAVED_EVAL_LIMIT}"
fi

# ── Group H: 无路径基线 ───────────────────────────────────────────────────────
if [[ "${RUN_GROUP}" == "ALL" || "${RUN_GROUP}" == "H" ]]; then
    log_section "Group H: 无路径基线 (base model 直接回答，无检索路径输入)"

    if [[ "${RUN_PHASE}" == "train" ]]; then
        echo "[SKIP] phase=train，Group H 仅做评估"
    else
        # H1: base model（无 adapter） + 无路径
        run_base_eval "groupH_base_nopaths" "${TEST_BEAM20_LAM02}" "v1" "--no_paths"

        # H2: 微调模型（groupAname_v2） + 无路径，对比参数知识 vs 检索增强
        if [[ -d "${BEST_ADAPTER}" ]]; then
            run_eval_only "groupH_ft_nopaths" "${BEST_ADAPTER}" "${TEST_BEAM20_LAM02}" "v1" "--no_paths"
        else
            echo "[WARN] BEST_ADAPTER 不存在，跳过 H2: ${BEST_ADAPTER}"
        fi
    fi
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
