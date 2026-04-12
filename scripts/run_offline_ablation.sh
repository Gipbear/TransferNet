#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_offline_ablation.sh
#
# 离线消融实验：基于 offline_search/paths/ 的路径文件运行 build→train→eval 流程
#
# 四组实验：
#   Group A (eval-only): 检索参数扫描 (beam/lambda/alpha × MID, chain+v2)
#   Group B (train+eval): 路径序列化格式 (arrow/chain/tuple/nl × MID/name, v2)
#   Group C (train+eval): 输出格式 (v1/v2/v3/v4 × MID/name, chain)
#   Group D (train+eval): 训练轮数 (epoch 1-5, chain+name, v2)
#
# 特性：
#   - Group A 仅 eval，复用已有 adapter；Group B/C/D 支持完整三步流程
#   - 断点续跑（数据集/adapter/eval_jsonl 已存在则跳过）
#   - --phase all|train|eval 控制跑哪些步骤
#   - EVAL_LIMIT 默认 500
#
# 用法：
#   bash scripts/run_offline_ablation.sh --group A
#   bash scripts/run_offline_ablation.sh --group A --all          # 扫描所有路径文件
#   bash scripts/run_offline_ablation.sh --group A --beam 20 --lam 0 --alpha 1
#   bash scripts/run_offline_ablation.sh --group B
#   bash scripts/run_offline_ablation.sh --group C --phase eval
#   bash scripts/run_offline_ablation.sh --group D --phase train
#   bash scripts/run_offline_ablation.sh --group ALL
#   bash scripts/run_offline_ablation.sh --group ALL --limit 10   # 快速冒烟测试
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJ_DIR}/scripts/run_ablation_lib.sh"

# ── 路径配置 ──────────────────────────────────────────────────────────────────
PATHS_DIR="${PROJ_DIR}/data/output/WebQSP/offline_search/paths"
ABLATION_DATA="${PROJ_DIR}/data/output/WebQSP/offline_ablation"
ABLATION_MODELS="${PROJ_DIR}/models/webqsp/offline_ablation"
TRAIN_INPUT="${PROJ_DIR}/data/output/WebQSP/predict_train.jsonl"
ENTITY_MAP="${PROJ_DIR}/data/resources/WebQSP/fbwq_full/mapped_entities.txt"

BUILD_SCRIPT="${PROJ_DIR}/llm_infer/build_kgcot_dataset.py"
TRAIN_SCRIPT="${PROJ_DIR}/llm_infer/train_sft.py"
EVAL_SCRIPT="${PROJ_DIR}/llm_infer/eval_faithfulness.py"

MODEL_DATASET="webqsp"

# ── 运行参数 ──────────────────────────────────────────────────────────────────
RUN_GROUP="ALL"
RUN_PHASE="all"        # all | train | eval
NUM_RUNS="${NUM_RUNS:-2}"
EPOCHS="${EPOCHS:-2}"
EVAL_LIMIT="${EVAL_LIMIT:-500}"

# GroupA 输入控制
SCAN_ALL=0
EXPLICIT_INPUTS=()
BEAM_VALS=()
LAM_VALS=()
ALPHA_VALS=()

# GroupB/C/D 使用的固定路径文件（空=自动推导）
DEFAULT_INPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --group)         RUN_GROUP="$2";              shift 2 ;;
        --phase)         RUN_PHASE="$2";              shift 2 ;;
        --input)         EXPLICIT_INPUTS+=("$2");     shift 2 ;;
        --paths_dir)     PATHS_DIR="$2";              shift 2 ;;
        --model_dataset) MODEL_DATASET="$2";          shift 2 ;;
        --num_runs)      NUM_RUNS="$2";               shift 2 ;;
        --limit)         EVAL_LIMIT="$2";             shift 2 ;;
        --epochs)        EPOCHS="$2";                 shift 2 ;;
        --beam)          BEAM_VALS+=("$2");           shift 2 ;;
        --lam)           LAM_VALS+=("$2");            shift 2 ;;
        --alpha)         ALPHA_VALS+=("$2");          shift 2 ;;
        --all)           SCAN_ALL=1;                  shift 1 ;;
        --default_input) DEFAULT_INPUT="$2";          shift 2 ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
done

if [[ "${RUN_PHASE}" != "all" && "${RUN_PHASE}" != "train" && "${RUN_PHASE}" != "eval" ]]; then
    echo "[ERROR] --phase 仅支持: all | train | eval"
    exit 1
fi

# ── 工具函数 ──────────────────────────────────────────────────────────────────
ts() { date '+%Y-%m-%d %H:%M:%S'; }

log_section() {
    echo ""
    echo "======================================================"
    echo "  $1"
    echo "  $(ts)"
    echo "======================================================"
}

log_step() {
    echo ""
    echo "  --------------------------------------------------"
    echo "  $1"
    echo "  --------------------------------------------------"
}

# 格式化浮点数字符串（去除多余零），与 run_offline_path_search.sh 保持一致
fmt_num() {
    local v
    v=$(printf '%s' "$1" | sed 's/\.*0*$//' | sed 's/^\./0./')
    [[ -z "$v" ]] && v="0"
    printf '%s' "$v"
}

# try_resolve_adapter CONFIG_NAME
# 查找 adapter；找不到时打印 WARN 并返回空串（不中断脚本）
try_resolve_adapter() {
    local config_name="$1"
    local result
    if result="$(resolve_slot_adapter "${PROJ_DIR}" "${MODEL_DATASET}" "${config_name}" 2>/dev/null)"; then
        printf '%s\n' "${result}"
    else
        echo "[WARN] adapter 未找到: ${config_name}，跳过" >&2
        printf ''
    fi
}

# ── eval_one ──────────────────────────────────────────────────────────────────
# eval_one INPUT VARIANT ADAPTER OUTPUT_FORMAT PATH_FORMAT [EXTRA_ARGS...]
#   INPUT         : 路径 JSONL 文件
#   VARIANT       : 输出子目录名（如 groupA, offB_chain_mid）
#   ADAPTER       : LoRA adapter 目录
#   OUTPUT_FORMAT : v1/v2/v3/v4
#   PATH_FORMAT   : arrow/chain/tuple/nl
#   EXTRA_ARGS    : 追加参数（如 --entity_map ...）
eval_one() {
    local input="$1"
    local variant="$2"
    local adapter="$3"
    local output_format="$4"
    local path_format="$5"
    shift 5
    local extra_args=("$@")

    local stem out_dir eval_json
    stem="$(basename "${input}" .jsonl)"
    out_dir="${ABLATION_DATA}/${variant}"
    eval_json="${out_dir}/${stem}_${output_format}_ft_eval.jsonl"

    if [[ ! -f "${input}" ]]; then
        echo "[WARN] 文件不存在，跳过: ${input}"; return 0
    fi
    if [[ -f "${eval_json}" ]]; then
        echo "[SKIP] ${variant}: ${eval_json}"; return 0
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
        --output_format "${output_format}" \
        --path_format   "${path_format}" \
        --num_runs      "${NUM_RUNS}" \
        "${extra_args[@]}" \
        "${limit_args[@]+"${limit_args[@]}"}"
    echo "  [INFO] 完成，耗时 $(($(date +%s) - T0))s"
}

# ── run_offline_experiment ───────────────────────────────────────────────────
# 完整三步流程：build_data → train → eval
# run_offline_experiment CONFIG_NAME FMT BUILD_EXTRA EVAL_INPUT EVAL_EXTRA...
#   CONFIG_NAME : 实验标识（决定数据和模型子目录）
#   FMT         : v1/v2/v3/v4（输出格式）
#   BUILD_EXTRA : 额外的 build_kgcot_dataset 参数（字符串，空格分隔）
#   EVAL_INPUT  : 评估用路径 JSONL 文件
#   EVAL_EXTRA  : 额外的 eval_faithfulness 参数（字符串，空格分隔）
run_offline_experiment() {
    local config_name="$1"
    local fmt="$2"
    local build_extra="${3:-}"
    local eval_input="$4"
    local eval_extra="${5:-}"

    local data_dir="${ABLATION_DATA}/${config_name}"
    local model_dir="${ABLATION_MODELS}/${config_name}"
    local dataset="${data_dir}/kgcot_train.jsonl"
    local adapter_flag="${model_dir}/adapter_config.json"
    local eval_adapter="${model_dir}"

    local stem
    stem="$(basename "${eval_input}" .jsonl)"
    local eval_json="${data_dir}/${stem}_${fmt}_ft_eval.jsonl"

    mkdir -p "${data_dir}" "${model_dir}"

    log_section "实验: ${config_name} (format=${fmt})"

    # ── Step 1: 构建训练数据 ─────────────────────────────────────────────────
    if [[ "${RUN_PHASE}" == "eval" ]]; then
        echo "[SKIP] phase=eval，跳过数据构建"
    elif [[ -f "${dataset}" ]]; then
        echo "[SKIP] 数据集已存在: ${dataset}"
    else
        if [[ ! -f "${TRAIN_INPUT}" ]]; then
            echo "[ERROR] 训练输入不存在: ${TRAIN_INPUT}"
            exit 1
        fi
        log_step "Step 1/3: 构建训练数据"
        local T0; T0=$(date +%s)
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
        log_step "Step 2/3: QLoRA 训练 (epochs=${EPOCHS})"
        local T0; T0=$(date +%s)
        python "${TRAIN_SCRIPT}" \
            --train      "${dataset}" \
            --output_dir "${model_dir}" \
            --epochs     "${EPOCHS}"
        echo "[INFO] 训练完成，耗时 $(($(date +%s) - T0))s"
    fi

    # ── Step 3: 评估 ─────────────────────────────────────────────────────────
    if [[ "${RUN_PHASE}" == "train" ]]; then
        echo "[SKIP] phase=train，跳过评估"
    elif [[ -f "${eval_json}" ]]; then
        echo "[SKIP] 评估结果已存在: ${eval_json}"
    else
        if [[ "${RUN_PHASE}" == "eval" ]]; then
            eval_adapter="$(resolve_slot_adapter "${PROJ_DIR}" "${MODEL_DATASET}" "${config_name}" \
                || resolve_slot_adapter "${PROJ_DIR}" "${MODEL_DATASET}" "${config_name%-*}" 2>/dev/null \
                || echo "${model_dir}")"
        fi
        if [[ ! -d "${eval_adapter}" ]]; then
            echo "[ERROR] adapter 不存在: ${eval_adapter}"
            exit 1
        fi
        if [[ ! -f "${eval_input}" ]]; then
            echo "[ERROR] 评估输入不存在: ${eval_input}"
            exit 1
        fi
        log_step "Step 3/3: 忠实度评估"
        echo "[INFO] adapter: ${eval_adapter}"
        local limit_args=()
        [[ "${EVAL_LIMIT}" -gt 0 ]] && limit_args+=(--limit "${EVAL_LIMIT}")
        local T0; T0=$(date +%s)
        # shellcheck disable=SC2086
        python "${EVAL_SCRIPT}" \
            --input         "${eval_input}" \
            --output        "${data_dir}" \
            --adapter       "${eval_adapter}" \
            --output_format "${fmt}" \
            --num_runs      "${NUM_RUNS}" \
            "${limit_args[@]+"${limit_args[@]}"}" \
            ${eval_extra}
        echo "[INFO] 评估完成，耗时 $(($(date +%s) - T0))s"
    fi
}

# ── 自动推导 DEFAULT_INPUT ────────────────────────────────────────────────────
if [[ -z "${DEFAULT_INPUT}" ]]; then
    DEFAULT_INPUT="${PATHS_DIR}/tail_blend_beam20_alpha1_lam0.jsonl"
    if [[ ! -f "${DEFAULT_INPUT}" ]]; then
        # 回退：PATHS_DIR 下第一个 .jsonl
        DEFAULT_INPUT="$(find "${PATHS_DIR}" -maxdepth 1 -name '*.jsonl' -print0 \
            | sort -z | head -z -n1 | tr -d '\0' || true)"
    fi
fi

# ── GroupA 输入列表构建 ───────────────────────────────────────────────────────
# 在 GroupA 块内使用；此处先声明空数组
GROUP_A_INPUTS=()

build_group_a_inputs() {
    # 明确指定的文件
    for f in "${EXPLICIT_INPUTS[@]+"${EXPLICIT_INPUTS[@]}"}"; do
        GROUP_A_INPUTS+=("$f")
    done

    # --beam + --lam + --alpha 组合（文件名: tail_blend_beam{B}_alpha{A}_lam{L}.jsonl）
    if [[ ${#BEAM_VALS[@]} -gt 0 && ${#LAM_VALS[@]} -gt 0 ]]; then
        local alphas=("${ALPHA_VALS[@]+"${ALPHA_VALS[@]}"}")
        [[ ${#alphas[@]} -eq 0 ]] && alphas=("1")   # 默认 alpha=1
        for beam in "${BEAM_VALS[@]}"; do
            for lam in "${LAM_VALS[@]}"; do
                for alpha in "${alphas[@]}"; do
                    local alpha_fmt lam_fmt
                    alpha_fmt="$(fmt_num "${alpha}")"
                    lam_fmt="$(fmt_num "${lam}")"
                    GROUP_A_INPUTS+=("${PATHS_DIR}/tail_blend_beam${beam}_alpha${alpha_fmt}_lam${lam_fmt}.jsonl")
                done
            done
        done
    fi

    # --all：扫描 PATHS_DIR 下所有 JSONL
    if [[ "${SCAN_ALL}" -eq 1 ]]; then
        while IFS= read -r -d '' f; do GROUP_A_INPUTS+=("$f"); done \
            < <(find "${PATHS_DIR}" -maxdepth 1 -name "*.jsonl" -print0 | sort -z)
    fi

    # 若全无指定，GroupA 默认 --all
    if [[ ${#GROUP_A_INPUTS[@]} -eq 0 ]]; then
        while IFS= read -r -d '' f; do GROUP_A_INPUTS+=("$f"); done \
            < <(find "${PATHS_DIR}" -maxdepth 1 -name "*.jsonl" -print0 | sort -z)
    fi
}

# ── Banner ─────────────────────────────────────────────────────────────────────
echo "======================================================"
echo "  offline ablation"
echo "  group       : ${RUN_GROUP}"
echo "  phase       : ${RUN_PHASE}"
echo "  paths_dir   : ${PATHS_DIR}"
echo "  default_input: ${DEFAULT_INPUT}"
echo "  eval_limit  : ${EVAL_LIMIT}"
echo "  num_runs    : ${NUM_RUNS}"
echo "  epochs      : ${EPOCHS}"
echo "  $(ts)"
echo "======================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Group A: 检索参数扫描 (eval-only)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${RUN_GROUP}" == "ALL" || "${RUN_GROUP}" == "A" ]]; then
    log_section "Group A: 检索参数扫描 (beam/lambda/alpha × MID, chain+v2, eval-only)"

    ADAPTER_A="$(try_resolve_adapter "groupAmid_v2")"
    if [[ -z "${ADAPTER_A}" ]]; then
        echo "[ERROR] Group A 需要 groupAmid_v2 adapter"; exit 1
    fi
    if [[ ! -d "${ADAPTER_A}" ]]; then
        echo "[ERROR] adapter 不存在: ${ADAPTER_A}"; exit 1
    fi

    build_group_a_inputs
    echo "  adapter: ${ADAPTER_A}"
    echo "  文件数 : ${#GROUP_A_INPUTS[@]}"

    WALL_A=$(date +%s)
    for input in "${GROUP_A_INPUTS[@]}"; do
        eval_one "${input}" "groupA" "${ADAPTER_A}" "v2" "chain"
    done
    echo ""
    echo "  [Group A 完成，耗时 $(($(date +%s) - WALL_A))s]"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Group B: 路径序列化格式 (train+eval)
# arrow/chain/tuple/nl × mid/name, 固定 v2 输出
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${RUN_GROUP}" == "ALL" || "${RUN_GROUP}" == "B" ]]; then
    log_section "Group B: 路径序列化格式 (arrow/chain/tuple/nl × MID/name, v2, train+eval)"

    if [[ "${RUN_PHASE}" != "train" && ! -f "${DEFAULT_INPUT}" ]]; then
        echo "[ERROR] Group B 评估需要 default_input: ${DEFAULT_INPUT}"; exit 1
    fi
    if [[ ! -f "${ENTITY_MAP}" ]]; then
        echo "[ERROR] 实体映射文件不存在: ${ENTITY_MAP}"; exit 1
    fi

    WALL_B=$(date +%s)
    for pfmt in arrow chain tuple nl; do
        # MID 变体
        run_offline_experiment \
            "offB_${pfmt}_mid" "v2" \
            "--path_format ${pfmt}" \
            "${DEFAULT_INPUT}" \
            "--path_format ${pfmt}"

        # Name 变体
        run_offline_experiment \
            "offB_${pfmt}_name" "v2" \
            "--path_format ${pfmt} --entity_map ${ENTITY_MAP}" \
            "${DEFAULT_INPUT}" \
            "--path_format ${pfmt} --entity_map ${ENTITY_MAP}"
    done
    echo ""
    echo "  [Group B 完成，耗时 $(($(date +%s) - WALL_B))s]"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Group C: 输出格式 (train+eval)
# chain × mid/name, v1/v2/v3/v4 输出格式
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${RUN_GROUP}" == "ALL" || "${RUN_GROUP}" == "C" ]]; then
    log_section "Group C: 输出格式 (v1/v2/v3/v4 × MID/name, chain, train+eval)"

    if [[ "${RUN_PHASE}" != "train" && ! -f "${DEFAULT_INPUT}" ]]; then
        echo "[ERROR] Group C 评估需要 default_input: ${DEFAULT_INPUT}"; exit 1
    fi
    if [[ ! -f "${ENTITY_MAP}" ]]; then
        echo "[ERROR] 实体映射文件不存在: ${ENTITY_MAP}"; exit 1
    fi

    WALL_C=$(date +%s)
    for fmt in v1 v2 v3 v4; do
        # MID 变体
        run_offline_experiment \
            "offC_mid_${fmt}" "${fmt}" \
            "--path_format chain" \
            "${DEFAULT_INPUT}" \
            "--path_format chain"

        # Name 变体
        run_offline_experiment \
            "offC_name_${fmt}" "${fmt}" \
            "--path_format chain --entity_map ${ENTITY_MAP}" \
            "${DEFAULT_INPUT}" \
            "--path_format chain --entity_map ${ENTITY_MAP}"
    done
    echo ""
    echo "  [Group C 完成，耗时 $(($(date +%s) - WALL_C))s]"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Group D: 训练轮数 (train+eval)
# chain+name, v2, epochs 1-5
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${RUN_GROUP}" == "ALL" || "${RUN_GROUP}" == "D" ]]; then
    log_section "Group D: 训练轮数 (epoch 1-5, chain+name, v2, train+eval)"

    if [[ "${RUN_PHASE}" != "train" && ! -f "${DEFAULT_INPUT}" ]]; then
        echo "[ERROR] Group D 评估需要 default_input: ${DEFAULT_INPUT}"; exit 1
    fi
    if [[ ! -f "${ENTITY_MAP}" ]]; then
        echo "[ERROR] 实体映射文件不存在: ${ENTITY_MAP}"; exit 1
    fi

    SAVED_EPOCHS="${EPOCHS}"
    WALL_D=$(date +%s)
    for ep in 1 2 3 4 5; do
        EPOCHS="${ep}"
        run_offline_experiment \
            "offD_epoch${ep}" "v2" \
            "--path_format chain --entity_map ${ENTITY_MAP}" \
            "${DEFAULT_INPUT}" \
            "--path_format chain --entity_map ${ENTITY_MAP}"
    done
    EPOCHS="${SAVED_EPOCHS}"
    echo ""
    echo "  [Group D 完成，耗时 $(($(date +%s) - WALL_D))s]"
fi

# ── 完成 ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  全部完成"
echo "  结果目录: ${ABLATION_DATA}/"
echo "  模型目录: ${ABLATION_MODELS}/"
echo "  $(ts)"
echo "======================================================"
