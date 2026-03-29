#!/usr/bin/env bash

init_dataset_context() {
    local project_dir="$1"
    local dataset="$2"

    case "${dataset}" in
        webqsp)
            DATASET_KEY="webqsp"
            DATASET_NAME="WebQSP"
            ;;
        metaqa)
            DATASET_KEY="metaqa"
            DATASET_NAME="MetaQA_KB"
            ;;
        cwq)
            DATASET_KEY="cwq"
            DATASET_NAME="CWQ"
            ;;
        *)
            echo "[ERROR] --dataset 仅支持: webqsp | metaqa | cwq" >&2
            return 1
            ;;
    esac

    DATASET_OUTPUT_ROOT="${project_dir}/data/output/${DATASET_NAME}"
    DATASET_PATHS_DIR="${DATASET_OUTPUT_ROOT}/grid_search/paths"
    DATASET_TRAIN_INPUT="${DATASET_OUTPUT_ROOT}/predict_train.jsonl"
    DATASET_ABLATION_DATA="${DATASET_OUTPUT_ROOT}/ablation"
    DATASET_ABLATION_MODELS="${project_dir}/models/${DATASET_KEY}/ablation"
    DATASET_TEST_BEAM20_LAM02="${DATASET_PATHS_DIR}/beam20_lam0.2.jsonl"
}

resolve_eval_limit() {
    local dataset="$1"

    case "${dataset}" in
        metaqa)
            printf '100\n'
            ;;
        webqsp|cwq)
            printf '0\n'
            ;;
        *)
            echo "[ERROR] 未知数据集，无法设置 eval limit: ${dataset}" >&2
            return 1
            ;;
    esac
}

resolve_baseline_adapter() {
    local project_dir="$1"
    local model_dataset="$2"
    local primary="${project_dir}/models/${model_dataset}/${model_dataset}_v2"
    local fallback="${project_dir}/models/webqsp/webqsp_v2"
    local legacy_fallback="${project_dir}/models/webqsp_v2_best"

    if [[ -d "${primary}" ]]; then
        printf '%s\n' "${primary}"
        return 0
    fi
    if [[ -d "${fallback}" ]]; then
        printf '%s\n' "${fallback}"
        return 0
    fi
    if [[ -d "${legacy_fallback}" ]]; then
        printf '%s\n' "${legacy_fallback}"
        return 0
    fi

    echo "[ERROR] baseline adapter 不存在: ${primary} 或 ${fallback} 或 ${legacy_fallback}" >&2
    return 1
}

resolve_slot_adapter() {
    local project_dir="$1"
    local model_dataset="$2"
    local config_name="$3"
    local primary="${project_dir}/models/${model_dataset}/ablation/${config_name}"
    local fallback="${project_dir}/models/webqsp/ablation/${config_name}"
    local legacy_fallback="${project_dir}/models/ablation/${config_name}"

    if [[ -d "${primary}" ]]; then
        printf '%s\n' "${primary}"
        return 0
    fi
    if [[ -d "${fallback}" ]]; then
        printf '%s\n' "${fallback}"
        return 0
    fi
    if [[ -d "${legacy_fallback}" ]]; then
        printf '%s\n' "${legacy_fallback}"
        return 0
    fi

    echo "[ERROR] slot adapter 不存在: ${primary} 或 ${fallback} 或 ${legacy_fallback}" >&2
    return 1
}
