#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_offline_ablation.sh
#
# 离线消融实验：基于 offline_search/paths/ 的路径文件运行 build→train→eval 流程
#
# 十组实验：
#   Group A (eval-only): 检索参数扫描 (beam/lambda/alpha × chain × name, v2)
#   Group B (train+eval): 路径序列化格式 (arrow/chain/tuple/nl/schema/schema_gloss × name, v2)
#   Group BBase (eval-only): base model 路径序列化格式 (arrow/chain/tuple/nl/schema/schema_gloss × name, v2)
#   Group C (train+eval): 输出格式 (v1/v2/v3/v4 × name, chain)
#   Group CBase (eval-only): base model 输出格式 (v1/v2/v3/v4 × name, chain)
#   Group D (train+eval): 训练轮数 (epoch 1-5, chain+name, v2)
#   Group E (train+eval): chain+name+v2 固定下的路径顺序/score/去重/干扰比例消融
#   Group F (eval-only): base model 无 adapter，固定 12 组检索参数，chain+name+v2
#   Group G (train+eval): 拒答训练策略 (no rejection / real / random synthetic 10% / random synthetic 15%)
#   Group H (train+eval): beam 匹配训练/推理 (B=5/10/15/20 × chain × name, v2)
#
# 特性：
#   - Group A/F 仅 eval；Group B/C/D/E/G/H 支持完整三步流程
#   - 断点续跑（数据集/adapter/eval_jsonl 已存在则跳过）
#   - --phase all|train|eval 控制跑哪些步骤
#   - EVAL_LIMIT 默认 500
#
# 用法：
#   bash scripts/run_offline_ablation.sh --group A
#   bash scripts/run_offline_ablation.sh --group A --all          # 扫描所有路径文件
#   bash scripts/run_offline_ablation.sh --group A --beam 20 --lam 0.2 --alpha 1
#   bash scripts/run_offline_ablation.sh --group B
#   bash scripts/run_offline_ablation.sh --group B --configs schema,schema_gloss
#   bash scripts/run_offline_ablation.sh --group BBase
#   bash scripts/run_offline_ablation.sh --group BBase --configs chain,schema
#   bash scripts/run_offline_ablation.sh --group C --phase eval
#   bash scripts/run_offline_ablation.sh --group C --configs v2,v4
#   bash scripts/run_offline_ablation.sh --group CBase
#   bash scripts/run_offline_ablation.sh --group CBase --configs v2,v4
#   bash scripts/run_offline_ablation.sh --group D --phase train
#   bash scripts/run_offline_ablation.sh --group E
#   bash scripts/run_offline_ablation.sh --group E --configs base,score
#   bash scripts/run_offline_ablation.sh --group F
#   bash scripts/run_offline_ablation.sh --group G
#   bash scripts/run_offline_ablation.sh --group G --configs base,real,syn10,syn15
#   bash scripts/run_offline_ablation.sh --group H
#   bash scripts/run_offline_ablation.sh --group H --grid_beams "5 10 15 20"
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
RUN_VARIANT="name"     # 兼容旧参数；当前离线消融仅跑 name
NUM_RUNS="${NUM_RUNS:-2}"
EPOCHS="${EPOCHS:-2}"
EVAL_LIMIT="${EVAL_LIMIT:-500}"

# GroupA 输入控制
SCAN_ALL=0
EXPLICIT_INPUTS=()
BEAM_VALS=()
LAM_VALS=()
ALPHA_VALS=()
# 网格批量指定（空格分隔，与 run_offline_path_search.sh 风格一致）
GRID_BEAMS=""
GRID_LAMS=""
GRID_ALPHAS=""

# GroupB/C/D 使用的固定路径文件（空=自动推导）
DEFAULT_INPUT=""
# 多配置 Group 的配置选择（逗号分隔；空=全部）
E_CONFIGS=""

# GroupH: beam 匹配训练/推理。要求提供真实 per-beam train split 路径文件，
# 不从 beam20/predict_train 前缀裁剪，避免把 MMR beam 实验伪装成前缀实验。
GROUPH_BEAMS="${GROUPH_BEAMS:-5 10 15 20}"
GROUPH_ALPHA="${GROUPH_ALPHA:-1}"
GROUPH_LAM="${GROUPH_LAM:-0.2}"
GROUPH_AUTO_PATH_SEARCH="${GROUPH_AUTO_PATH_SEARCH:-1}"
GROUPH_TRAIN_OFFLINE_DIR="${GROUPH_TRAIN_OFFLINE_DIR:-${PROJ_DIR}/data/output/WebQSP/offline_search_train}"
GROUPH_TRAIN_PATHS_DIR="${GROUPH_TRAIN_PATHS_DIR:-${PROJ_DIR}/data/output/WebQSP/offline_search_train/paths}"
GROUPH_TRAIN_CACHE="${GROUPH_TRAIN_CACHE:-${GROUPH_TRAIN_OFFLINE_DIR}/score_cache/webqsp_train.pt}"
GROUPH_TRAIN_INPUT_DIR="${GROUPH_TRAIN_INPUT_DIR:-${PROJ_DIR}/data/input/WebQSP}"
GROUPH_TRAIN_QA_FILE="${GROUPH_TRAIN_QA_FILE:-${GROUPH_TRAIN_INPUT_DIR}/QA_data/WebQuestionsSP/qa_train_webqsp.txt}"
GROUPH_TRAIN_CKPT="${GROUPH_TRAIN_CKPT:-}"
GROUPH_MAX_SEQ_LEN="${GROUPH_MAX_SEQ_LEN:-2048}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --group)         RUN_GROUP="$2";              shift 2 ;;
        --phase)         RUN_PHASE="$2";              shift 2 ;;
        --variant)       RUN_VARIANT="$2";           shift 2 ;;
        --input)         EXPLICIT_INPUTS+=("$2");     shift 2 ;;
        --paths_dir)     PATHS_DIR="$2";              shift 2 ;;
        --model_dataset) MODEL_DATASET="$2";          shift 2 ;;
        --num_runs)      NUM_RUNS="$2";               shift 2 ;;
        --limit)         EVAL_LIMIT="$2";             shift 2 ;;
        --epochs)        EPOCHS="$2";                 shift 2 ;;
        --beam)          BEAM_VALS+=("$2");           shift 2 ;;
        --lam)           LAM_VALS+=("$2");            shift 2 ;;
        --alpha)         ALPHA_VALS+=("$2");          shift 2 ;;
        --grid_beams)    GRID_BEAMS="$2";             shift 2 ;;
        --grid_lams)     GRID_LAMS="$2";              shift 2 ;;
        --grid_alphas)   GRID_ALPHAS="$2";            shift 2 ;;
        --all)           SCAN_ALL=1;                  shift 1 ;;
        --default_input) DEFAULT_INPUT="$2";          shift 2 ;;
        --configs)       E_CONFIGS="$2";              shift 2 ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
done

if [[ "${RUN_PHASE}" != "all" && "${RUN_PHASE}" != "train" && "${RUN_PHASE}" != "eval" ]]; then
    echo "[ERROR] --phase 仅支持: all | train | eval"
    exit 1
fi
if [[ "${RUN_VARIANT}" != "name" ]]; then
    echo "[ERROR] 当前离线消融仅支持 --variant name；MID/name 正交请使用单独实验"
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

# eval_output_complete EVAL_JSON NUM_RUNS
# num_runs>1 时 eval_faithfulness.py 只写 *_runN.jsonl，聚合完成标记在日志末尾。
eval_output_complete() {
    local eval_json="$1"
    local num_runs="$2"

    if [[ "${num_runs}" -le 1 ]]; then
        [[ -f "${eval_json}" ]]
        return
    fi

    local stem log_path i run_path
    stem="${eval_json%.jsonl}"
    log_path="${stem}.log"

    [[ -f "${log_path}" ]] || return 1
    grep -Fq "finish_time:" "${log_path}" || return 1
    grep -Fq "多轮汇总 (num_runs=${num_runs})" "${log_path}" || return 1

    for ((i = 0; i < num_runs; i++)); do
        run_path="${stem}_run${i}.jsonl"
        [[ -s "${run_path}" ]] || return 1
    done
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
#   VARIANT       : 输出子目录名（如 groupA, offB_chain_name）
#   ADAPTER       : LoRA adapter 目录
#   OUTPUT_FORMAT : v1/v2/v3/v4
#   PATH_FORMAT   : arrow/chain/tuple/nl/schema/schema_gloss
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
    if eval_output_complete "${eval_json}" "${NUM_RUNS}"; then
        echo "[SKIP] ${variant}: 评估已完成: ${eval_json}"; return 0
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
# run_offline_experiment CONFIG_NAME FMT BUILD_EXTRA EVAL_INPUT EVAL_EXTRA [TRAIN_INPUT_OVERRIDE] [TRAIN_EXTRA]
#   CONFIG_NAME : 实验标识（决定数据和模型子目录）
#   FMT         : v1/v2/v3/v4（输出格式）
#   BUILD_EXTRA : 额外的 build_kgcot_dataset 参数（字符串，空格分隔）
#   EVAL_INPUT  : 评估用路径 JSONL 文件
#   EVAL_EXTRA  : 额外的 eval_faithfulness 参数（字符串，空格分隔）
#   TRAIN_INPUT_OVERRIDE: 可选，覆盖默认训练路径文件
#   TRAIN_EXTRA : 可选，额外的 train_sft 参数（字符串，空格分隔）
run_offline_experiment() {
    local config_name="$1"
    local fmt="$2"
    local build_extra="${3:-}"
    local eval_input="$4"
    local eval_extra="${5:-}"
    local train_input="${6:-${TRAIN_INPUT}}"
    local train_extra="${7:-}"

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
        if [[ ! -f "${train_input}" ]]; then
            echo "[ERROR] 训练输入不存在: ${train_input}"
            exit 1
        fi
        log_step "Step 1/3: 构建训练数据"
        local T0; T0=$(date +%s)
        # shellcheck disable=SC2086
        python "${BUILD_SCRIPT}" \
            --input  "${train_input}" \
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
            --epochs     "${EPOCHS}" \
            ${train_extra}
        echo "[INFO] 训练完成，耗时 $(($(date +%s) - T0))s"
    fi

    # ── Step 3: 评估 ─────────────────────────────────────────────────────────
    if [[ "${RUN_PHASE}" == "train" ]]; then
        echo "[SKIP] phase=train，跳过评估"
    elif eval_output_complete "${eval_json}" "${NUM_RUNS}"; then
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

# ── run_offline_eval_variant ──────────────────────────────────────────────────
# 只新增独立评估目录，复用同组内已训练 adapter。
# run_offline_eval_variant CONFIG_NAME ADAPTER_CONFIG FMT EVAL_INPUT EVAL_EXTRA
run_offline_eval_variant() {
    local config_name="$1"
    local adapter_config="$2"
    local fmt="$3"
    local eval_input="$4"
    local eval_extra="${5:-}"

    local data_dir="${ABLATION_DATA}/${config_name}"
    local model_dir="${ABLATION_MODELS}/${adapter_config}"
    local eval_adapter="${model_dir}"

    local stem
    stem="$(basename "${eval_input}" .jsonl)"
    local eval_json="${data_dir}/${stem}_${fmt}_ft_eval.jsonl"

    mkdir -p "${data_dir}"

    log_section "实验: ${config_name} (eval adapter=${adapter_config}, format=${fmt})"

    if [[ "${RUN_PHASE}" == "train" ]]; then
        echo "[SKIP] phase=train，跳过评估"
        return 0
    fi
    if eval_output_complete "${eval_json}" "${NUM_RUNS}"; then
        echo "[SKIP] 评估结果已存在: ${eval_json}"
        return 0
    fi
    if [[ "${RUN_PHASE}" == "eval" ]]; then
        eval_adapter="$(resolve_slot_adapter "${PROJ_DIR}" "${MODEL_DATASET}" "${adapter_config}" 2>/dev/null \
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

    log_step "Step: 忠实度评估"
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
}

# ── run_offline_base_eval ─────────────────────────────────────────────────────
# 无 adapter 评估：直接使用 base model，保留指定路径格式 / 输出格式。
# run_offline_base_eval CONFIG_NAME FMT EVAL_INPUT EVAL_EXTRA
run_offline_base_eval() {
    local config_name="$1"
    local fmt="$2"
    local eval_input="$3"
    local eval_extra="${4:-}"

    local data_dir="${ABLATION_DATA}/${config_name}"
    local stem
    stem="$(basename "${eval_input}" .jsonl)"
    local eval_json="${data_dir}/${stem}_${fmt}_eval.jsonl"

    mkdir -p "${data_dir}"

    log_section "实验: ${config_name} (base model, no adapter, format=${fmt})"

    if [[ "${RUN_PHASE}" == "train" ]]; then
        echo "[SKIP] phase=train，跳过无 adapter 评估"
        return 0
    fi
    if eval_output_complete "${eval_json}" "${NUM_RUNS}"; then
        echo "[SKIP] 评估结果已存在: ${eval_json}"
        return 0
    fi
    if [[ ! -f "${eval_input}" ]]; then
        echo "[ERROR] 评估输入不存在: ${eval_input}"
        exit 1
    fi

    log_step "Step: 无 adapter 忠实度评估"
    local limit_args=()
    [[ "${EVAL_LIMIT}" -gt 0 ]] && limit_args+=(--limit "${EVAL_LIMIT}")
    local T0; T0=$(date +%s)
    # shellcheck disable=SC2086
    python "${EVAL_SCRIPT}" \
        --input         "${eval_input}" \
        --output        "${data_dir}" \
        --output_format "${fmt}" \
        --num_runs      "${NUM_RUNS}" \
        "${limit_args[@]+"${limit_args[@]}"}" \
        ${eval_extra}
    echo "[INFO] 评估完成，耗时 $(($(date +%s) - T0))s"
}

config_selected() {
    local name="$1"
    shift || true
    [[ -z "${E_CONFIGS}" ]] && return 0
    local token
    local alias
    local -a _selected
    IFS=',' read -r -a _selected <<< "${E_CONFIGS}"
    for token in "${_selected[@]}"; do
        token="${token//[[:space:]]/}"
        [[ "${token}" == "${name}" ]] && return 0
        for alias in "$@"; do
            [[ "${token}" == "${alias}" ]] && return 0
        done
    done
    return 1
}

group_b_config_selected() {
    local name="$1"
    [[ "${RUN_GROUP}" != "B" ]] || config_selected "${name}"
}

group_bbase_config_selected() {
    local name="$1"
    [[ "${RUN_GROUP}" != "BBase" ]] || config_selected "${name}"
}

group_c_config_selected() {
    local name="$1"
    [[ "${RUN_GROUP}" != "C" ]] || config_selected "${name}"
}

group_cbase_config_selected() {
    local name="$1"
    [[ "${RUN_GROUP}" != "CBase" ]] || config_selected "${name}"
}

group_e_config_selected() {
    local name="$1"
    case "${name}" in
        dist0.3) config_selected "${name}" "dist03" ;;
        dist0.5) config_selected "${name}" "dist05" ;;
        *)       config_selected "${name}" ;;
    esac
}

group_g_config_selected() {
    local name="$1"
    config_selected "${name}"
}

build_group_h_beams() {
    GROUP_H_BEAMS=()

    # --beam/--grid_beams 优先；未指定时使用 GROUPH_BEAMS 默认值。
    for beam in "${BEAM_VALS[@]+"${BEAM_VALS[@]}"}"; do
        GROUP_H_BEAMS+=("${beam}")
    done
    if [[ -n "${GRID_BEAMS}" ]]; then
        local -a _ghb
        read -r -a _ghb <<< "${GRID_BEAMS}"
        GROUP_H_BEAMS+=("${_ghb[@]}")
    fi
    if [[ ${#GROUP_H_BEAMS[@]} -eq 0 ]]; then
        read -r -a GROUP_H_BEAMS <<< "${GROUPH_BEAMS}"
    fi
}

config_token_allowed() {
    local token="$1"
    shift
    local allowed
    for allowed in "$@"; do
        [[ "${token}" == "${allowed}" ]] && return 0
    done
    return 1
}

validate_selected_configs() {
    [[ -z "${E_CONFIGS}" ]] && return 0
    local token
    local -a _selected _allowed

    case "${RUN_GROUP}" in
        B|BBase) _allowed=(arrow chain tuple nl schema schema_gloss) ;;
        C|CBase) _allowed=(v1 v2 v3 v4) ;;
        E)   _allowed=(base eval_shuffle train_noshuffle train_noshuffle_eval_shuffle score dist0.3 dist0.5 dist03 dist05 dedupe_tail) ;;
        G)   _allowed=(base real syn10 syn15) ;;
        ALL) _allowed=(base eval_shuffle train_noshuffle train_noshuffle_eval_shuffle score dist0.3 dist0.5 dist03 dist05 dedupe_tail real syn10 syn15) ;;
        *)   return 0 ;;
    esac

    IFS=',' read -r -a _selected <<< "${E_CONFIGS}"
    for token in "${_selected[@]}"; do
        token="${token//[[:space:]]/}"
        config_token_allowed "${token}" "${_allowed[@]}" && continue
        echo "[ERROR] 未知 config: ${token} (group=${RUN_GROUP})"
        exit 1
    done
}

validate_selected_configs

# ── 自动推导 DEFAULT_INPUT ────────────────────────────────────────────────────
if [[ -z "${DEFAULT_INPUT}" ]]; then
    DEFAULT_INPUT="${PATHS_DIR}/tail_blend_beam20_alpha1_lam0.2.jsonl"
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

    # --beam/--lam/--alpha 与 --grid_beams/--grid_lams/--grid_alphas 合并后做笛卡尔积
    # 将 --grid_* 空格分隔值追加进对应数组
    local beams=("${BEAM_VALS[@]+"${BEAM_VALS[@]}"}")
    local lams=("${LAM_VALS[@]+"${LAM_VALS[@]}"}")
    local alphas=("${ALPHA_VALS[@]+"${ALPHA_VALS[@]}"}")
    [[ -n "${GRID_BEAMS}"  ]] && read -r -a _gb  <<< "${GRID_BEAMS}"  && beams+=("${_gb[@]}")
    [[ -n "${GRID_LAMS}"   ]] && read -r -a _gl  <<< "${GRID_LAMS}"   && lams+=("${_gl[@]}")
    [[ -n "${GRID_ALPHAS}" ]] && read -r -a _ga  <<< "${GRID_ALPHAS}" && alphas+=("${_ga[@]}")
    [[ ${#alphas[@]} -eq 0 ]] && alphas=("1")   # 默认 alpha=1

    if [[ ${#beams[@]} -gt 0 && ${#lams[@]} -gt 0 ]]; then
        for beam in "${beams[@]}"; do
            for lam in "${lams[@]}"; do
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

group_enabled() {
    local group="$1"
    [[ "${RUN_GROUP}" == "ALL" || "${RUN_GROUP}" == "${group}" ]]
}

require_entity_map() {
    if [[ ! -f "${ENTITY_MAP}" ]]; then
        echo "[ERROR] 实体映射文件不存在: ${ENTITY_MAP}"
        exit 1
    fi
}

require_default_input() {
    local group="$1"
    if [[ "${RUN_PHASE}" != "train" && ! -f "${DEFAULT_INPUT}" ]]; then
        echo "[ERROR] Group ${group} 评估需要 default_input: ${DEFAULT_INPUT}"
        exit 1
    fi
}

finish_group() {
    local group="$1"
    local start_ts="$2"
    echo ""
    echo "  [Group ${group} 完成，耗时 $(($(date +%s) - start_ts))s]"
}

print_banner() {
    echo "======================================================"
    echo "  offline ablation"
    echo "  group       : ${RUN_GROUP}"
    echo "  phase       : ${RUN_PHASE}"
    echo "  paths_dir   : ${PATHS_DIR}"
    echo "  default_input: ${DEFAULT_INPUT}"
    echo "  eval_limit  : ${EVAL_LIMIT}"
    echo "  num_runs    : ${NUM_RUNS}"
    echo "  epochs      : ${EPOCHS}"
    echo "  configs     : ${E_CONFIGS:-ALL}"
    echo "  $(ts)"
    echo "======================================================"
}

run_group_a() {
    log_section "Group A: 检索参数扫描 (beam/lambda/alpha × chain × name, v2, eval-only)"

    local adapter
    adapter="${ABLATION_MODELS}/offB_chain_name"
    if [[ ! -d "${adapter}" ]]; then
        adapter="$(try_resolve_adapter "groupAname_v2")"
    fi
    if [[ -z "${adapter}" ]]; then
        echo "[ERROR] Group A chain/name 评估需要 offB_chain_name 或 groupAname_v2 adapter"
        exit 1
    fi

    build_group_a_inputs
    echo "  adapter_chain_name: ${adapter:-（未找到）}"
    echo "  serialization     : output_format=v2, path_format=chain, entity=name"
    echo "  文件数             : ${#GROUP_A_INPUTS[@]}"

    require_entity_map

    local wall
    wall=$(date +%s)
    local input
    for input in "${GROUP_A_INPUTS[@]}"; do
        eval_one "${input}" "groupA/chain_name" \
            "${adapter}" "v2" "chain" --entity_map "${ENTITY_MAP}"
    done
    finish_group "A" "${wall}"
}

run_group_b() {
    log_section "Group B: 路径序列化格式 (arrow/chain/tuple/nl/schema/schema_gloss × name, v2, train+eval)"
    require_default_input "B"
    require_entity_map

    local wall pfmt
    wall=$(date +%s)
    for pfmt in arrow chain tuple nl schema schema_gloss; do
        group_b_config_selected "${pfmt}" || continue
        run_offline_experiment \
            "offB_${pfmt}_name" "v2" \
            "--path_format ${pfmt} --entity_map ${ENTITY_MAP}" \
            "${DEFAULT_INPUT}" \
            "--path_format ${pfmt} --entity_map ${ENTITY_MAP}"
    done
    finish_group "B" "${wall}"
}

run_group_bbase() {
    log_section "Group BBase: base model 路径序列化格式 (arrow/chain/tuple/nl/schema/schema_gloss × name, v2, eval-only)"
    require_default_input "BBase"
    require_entity_map

    local wall pfmt
    wall=$(date +%s)
    for pfmt in arrow chain tuple nl schema schema_gloss; do
        group_bbase_config_selected "${pfmt}" || continue
        run_offline_base_eval \
            "offBBase_${pfmt}_name" "v2" \
            "${DEFAULT_INPUT}" \
            "--path_format ${pfmt} --entity_map ${ENTITY_MAP}"
    done
    finish_group "BBase" "${wall}"
}

run_group_c() {
    log_section "Group C: 输出格式 (v1/v2/v3/v4 × name, chain, train+eval)"
    require_default_input "C"
    require_entity_map

    local wall fmt
    wall=$(date +%s)
    for fmt in v1 v2 v3 v4; do
        group_c_config_selected "${fmt}" || continue
        run_offline_experiment \
            "offC_name_${fmt}" "${fmt}" \
            "--path_format chain --entity_map ${ENTITY_MAP}" \
            "${DEFAULT_INPUT}" \
            "--path_format chain --entity_map ${ENTITY_MAP}"
    done
    finish_group "C" "${wall}"
}

run_group_cbase() {
    log_section "Group CBase: base model 输出格式 (v1/v2/v3/v4 × name, chain, eval-only)"
    require_default_input "CBase"
    require_entity_map

    local wall fmt
    wall=$(date +%s)
    for fmt in v1 v2 v3 v4; do
        group_cbase_config_selected "${fmt}" || continue
        run_offline_base_eval \
            "offCBase_name_${fmt}" "${fmt}" \
            "${DEFAULT_INPUT}" \
            "--path_format chain --entity_map ${ENTITY_MAP}"
    done
    finish_group "CBase" "${wall}"
}

run_group_d() {
    log_section "Group D: 训练轮数 (epoch 1-5, chain+name, v2, train+eval)"
    require_default_input "D"
    require_entity_map

    local saved_epochs wall ep
    saved_epochs="${EPOCHS}"
    wall=$(date +%s)
    for ep in 1 2 3 4 5; do
        EPOCHS="${ep}"
        run_offline_experiment \
            "offD_epoch${ep}" "v2" \
            "--path_format chain --entity_map ${ENTITY_MAP}" \
            "${DEFAULT_INPUT}" \
            "--path_format chain --entity_map ${ENTITY_MAP}"
    done
    EPOCHS="${saved_epochs}"
    finish_group "D" "${wall}"
}

run_group_e() {
    log_section "Group E: chain+name+v2 路径顺序/score/去重/干扰比例消融 (epoch=${EPOCHS})"
    require_default_input "E"
    require_entity_map

    local base_args wall row key kind field3 field4 field5 field6
    base_args="--path_format chain --entity_map ${ENTITY_MAP}"
    wall=$(date +%s)

    local -a variants=(
        "base|experiment|offE_base|v2|${base_args}|${base_args}"
        "eval_shuffle|eval_variant|offE_eval_shuffle|offE_base|v2|${base_args} --shuffle_paths"
        "train_noshuffle|experiment|offE_train_noshuffle|v2|${base_args} --no_shuffle|${base_args}"
        "train_noshuffle_eval_shuffle|eval_variant|offE_train_noshuffle_eval_shuffle|offE_train_noshuffle|v2|${base_args} --shuffle_paths"
        "score|experiment|offE_score|v2|${base_args} --show_score|${base_args} --show_score"
        "dist0.3|experiment|offE_dist0.3|v2|${base_args} --distractor_ratio 0.3|${base_args}"
        "dist0.5|experiment|offE_dist0.5|v2|${base_args} --distractor_ratio 0.5|${base_args}"
        "dedupe_tail|experiment|offE_dedupe_tail|v2|${base_args} --dedupe_tail_paths|${base_args} --dedupe_tail_paths"
    )

    for row in "${variants[@]}"; do
        IFS='|' read -r key kind field3 field4 field5 field6 <<< "${row}"
        group_e_config_selected "${key}" || continue
        if [[ "${kind}" == "experiment" ]]; then
            run_offline_experiment "${field3}" "${field4}" \
                "${field5}" "${DEFAULT_INPUT}" "${field6}"
        else
            run_offline_eval_variant "${field3}" "${field4}" \
                "${field5}" "${DEFAULT_INPUT}" "${field6}"
        fi
    done
    finish_group "E" "${wall}"
}

run_group_f() {
    log_section "Group F: base model 无 adapter (12 retrieval configs, chain × name, v2, eval-only)"
    require_entity_map

    local wall saved_num_runs saved_eval_limit input
    wall=$(date +%s)
    saved_num_runs="${NUM_RUNS}"
    saved_eval_limit="${EVAL_LIMIT}"
    NUM_RUNS=1
    EVAL_LIMIT=0

    local -a inputs=(
        "${PATHS_DIR}/tail_blend_beam3_alpha1_lam0.2.jsonl"
        "${PATHS_DIR}/tail_blend_beam3_alpha0_lam0.2.jsonl"
        "${PATHS_DIR}/tail_blend_beam3_alpha1_lam0.jsonl"
        "${PATHS_DIR}/tail_blend_beam3_alpha0_lam0.jsonl"
        "${PATHS_DIR}/tail_blend_beam10_alpha1_lam0.2.jsonl"
        "${PATHS_DIR}/tail_blend_beam10_alpha0_lam0.2.jsonl"
        "${PATHS_DIR}/tail_blend_beam10_alpha1_lam0.jsonl"
        "${PATHS_DIR}/tail_blend_beam10_alpha0_lam0.jsonl"
        "${PATHS_DIR}/tail_blend_beam20_alpha1_lam0.2.jsonl"
        "${PATHS_DIR}/tail_blend_beam20_alpha0_lam0.2.jsonl"
        "${PATHS_DIR}/tail_blend_beam20_alpha1_lam0.jsonl"
        "${PATHS_DIR}/tail_blend_beam20_alpha0_lam0.jsonl"
    )

    echo "  configs:"
    echo "    alpha=1 lambda=0.2 beam=3"
    echo "    alpha=0 lambda=0.2 beam=3"
    echo "    alpha=1 lambda=0   beam=3"
    echo "    alpha=0 lambda=0   beam=3"
    echo "    alpha=1 lambda=0.2 beam=10"
    echo "    alpha=0 lambda=0.2 beam=10"
    echo "    alpha=1 lambda=0   beam=10"
    echo "    alpha=0 lambda=0   beam=10"
    echo "    alpha=1 lambda=0.2 beam=20"
    echo "    alpha=0 lambda=0.2 beam=20"
    echo "    alpha=1 lambda=0   beam=20"
    echo "    alpha=0 lambda=0   beam=20"
    echo "  eval_limit: full"
    echo "  num_runs  : 1"

    for input in "${inputs[@]}"; do
        if [[ "${RUN_PHASE}" != "train" && ! -f "${input}" ]]; then
            echo "[ERROR] Group F 评估输入不存在: ${input}"
            exit 1
        fi
        run_offline_base_eval \
            "offF_base_chain_name" "v2" \
            "${input}" \
            "--path_format chain --entity_map ${ENTITY_MAP}"
    done
    NUM_RUNS="${saved_num_runs}"
    EVAL_LIMIT="${saved_eval_limit}"
    finish_group "F" "${wall}"
}

run_group_g() {
    log_section "Group G: 拒答训练策略消融 (chain × name, v2, epoch=${EPOCHS})"
    require_default_input "G"
    require_entity_map

    local base_args wall row key config_name build_extra eval_extra
    base_args="--path_format chain --entity_map ${ENTITY_MAP}"
    wall=$(date +%s)

    local -a variants=(
        "base|offG_base|${base_args}|${base_args}"
        "real|offG_real_rejection|${base_args} --include_rejection|${base_args} --reject_prompt"
        "syn10|offG_synthetic_rejection_10|${base_args} --include_rejection --synthetic_rejection_ratio 0.10|${base_args} --reject_prompt"
        "syn15|offG_synthetic_rejection_15|${base_args} --include_rejection --synthetic_rejection_ratio 0.15|${base_args} --reject_prompt"
    )

    for row in "${variants[@]}"; do
        IFS='|' read -r key config_name build_extra eval_extra <<< "${row}"
        group_g_config_selected "${key}" || continue
        run_offline_experiment "${config_name}" "v2" \
            "${build_extra}" "${DEFAULT_INPUT}" "${eval_extra}"
    done
    finish_group "G" "${wall}"
}

ensure_group_h_train_paths() {
    local beam="$1"
    local train_paths="$2"

    [[ -f "${train_paths}" ]] && return 0

    if [[ "${GROUPH_AUTO_PATH_SEARCH}" != "1" ]]; then
        echo "[ERROR] GroupH 训练路径文件不存在: ${train_paths}"
        echo "        可设置 GROUPH_AUTO_PATH_SEARCH=1 自动生成，或设置 GROUPH_TRAIN_PATHS_DIR 指向已生成目录。"
        exit 1
    fi

    log_step "GroupH: 生成 train split 检索路径 (beam=${beam})"
    echo "[INFO] output: ${train_paths}"
    echo "[INFO] cache : ${GROUPH_TRAIN_CACHE}"
    echo "[INFO] qa    : ${GROUPH_TRAIN_QA_FILE}"

    local -a search_args=(
        --dataset "${MODEL_DATASET}"
        --phase all
        --mode train
        --input_dir "${GROUPH_TRAIN_INPUT_DIR}"
        --qa_file "${GROUPH_TRAIN_QA_FILE}"
        --cache "${GROUPH_TRAIN_CACHE}"
        --offline_dir "${GROUPH_TRAIN_OFFLINE_DIR}"
        --paths_dir "${GROUPH_TRAIN_PATHS_DIR}"
        --summary_file "${GROUPH_TRAIN_OFFLINE_DIR}/summary.csv"
        --grid
        --grid_alphas "${GROUPH_ALPHA}"
        --grid_lambdas "${GROUPH_LAM}"
        --grid_thresholds "0.01"
        --grid_beams "${beam}"
    )
    if [[ -n "${GROUPH_TRAIN_CKPT}" ]]; then
        search_args+=(--ckpt "${GROUPH_TRAIN_CKPT}")
    fi

    bash "${PROJ_DIR}/scripts/run_offline_path_search.sh" "${search_args[@]}"

    if [[ ! -f "${train_paths}" ]]; then
        echo "[ERROR] GroupH train path 生成后仍未找到: ${train_paths}"
        exit 1
    fi
}

run_group_h() {
    log_section "Group H: beam 匹配训练/推理 (chain × name, v2, epoch=${EPOCHS})"
    require_entity_map
    build_group_h_beams

    local alpha_fmt lam_fmt wall beam train_paths eval_input config_name base_args eval_args train_args
    alpha_fmt="$(fmt_num "${GROUPH_ALPHA}")"
    lam_fmt="$(fmt_num "${GROUPH_LAM}")"
    base_args="--path_format chain --entity_map ${ENTITY_MAP}"
    eval_args="${base_args} --max_seq_length ${GROUPH_MAX_SEQ_LEN}"
    train_args="--max_seq_len ${GROUPH_MAX_SEQ_LEN}"
    wall=$(date +%s)

    echo "  beams             : ${GROUP_H_BEAMS[*]}"
    echo "  alpha/lambda      : alpha=${alpha_fmt}, lambda=${lam_fmt}"
    echo "  auto_path_search  : ${GROUPH_AUTO_PATH_SEARCH}"
    echo "  train_offline_dir : ${GROUPH_TRAIN_OFFLINE_DIR}"
    echo "  train_paths_dir   : ${GROUPH_TRAIN_PATHS_DIR}"
    echo "  eval_paths_dir    : ${PATHS_DIR}"
    echo "  max_seq_len       : ${GROUPH_MAX_SEQ_LEN}"

    for beam in "${GROUP_H_BEAMS[@]}"; do
        train_paths="${GROUPH_TRAIN_PATHS_DIR}/tail_blend_beam${beam}_alpha${alpha_fmt}_lam${lam_fmt}.jsonl"
        eval_input="${PATHS_DIR}/tail_blend_beam${beam}_alpha${alpha_fmt}_lam${lam_fmt}.jsonl"
        config_name="offH_beam${beam}_chain_name"

        if [[ "${RUN_PHASE}" != "eval" ]]; then
            ensure_group_h_train_paths "${beam}" "${train_paths}"
        fi
        if [[ ! -f "${eval_input}" && "${RUN_PHASE}" != "train" ]]; then
            echo "[ERROR] GroupH 评估输入不存在: ${eval_input}"
            exit 1
        fi

        run_offline_experiment \
            "${config_name}" "v2" \
            "${base_args}" \
            "${eval_input}" \
            "${eval_args}" \
            "${train_paths}" \
            "${train_args}"
    done
    finish_group "H" "${wall}"
}

print_summary() {
    echo ""
    echo "======================================================"
    echo "  全部完成"
    echo "  结果目录: ${ABLATION_DATA}/"
    echo "  模型目录: ${ABLATION_MODELS}/"
    echo "  $(ts)"
    echo "======================================================"
}

print_banner
group_enabled "A" && run_group_a
group_enabled "B" && run_group_b
group_enabled "BBase" && run_group_bbase
group_enabled "C" && run_group_c
group_enabled "CBase" && run_group_cbase
group_enabled "D" && run_group_d
group_enabled "E" && run_group_e
group_enabled "F" && run_group_f
group_enabled "G" && run_group_g
group_enabled "H" && run_group_h
print_summary
