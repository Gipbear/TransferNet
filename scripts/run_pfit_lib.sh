#!/usr/bin/env bash
# run_pfit.sh 的库函数:上下文初始化 / 实验目录 / 三阶段命令拼装。
# DRY_RUN=1 时只打印命令不执行(供 bash 测试与人工检查)。

pfit_init_context() {
    local project_dir="$1" dataset="$2"
    case "${dataset}" in
        webqsp)
            PFIT_DS_ROOT="${project_dir}/data/output/kgqa/webqsp"
            PFIT_RETRIEVE_TRAIN="${PFIT_DS_ROOT}/retrieve/train.jsonl"
            PFIT_RETRIEVE_TEST="${PFIT_DS_ROOT}/retrieve/test.jsonl"
            ;;
        metaqa)
            PFIT_DS_ROOT="${project_dir}/data/output/kgqa/metaqa"
            PFIT_RETRIEVE_TRAIN="${PFIT_DS_ROOT}/retrieve/train_20k.jsonl"
            PFIT_RETRIEVE_TEST="${PFIT_DS_ROOT}/retrieve/test.jsonl"
            ;;
        *)
            echo "[ERROR] 未知数据集: ${dataset}(可用 webqsp|metaqa)" >&2
            return 1
            ;;
    esac
}

pfit_exp_dir() {
    # 实验目录名随变体后缀自描述:FMT 变体 / ADAPTER 覆盖(_ft)/ smoke(LIMIT)
    local exp_id="$1"
    local dir="${PFIT_DS_ROOT}/pfit/${exp_id}"
    [[ -n "${FMT:-}" ]] && dir="${dir}_${FMT}"
    [[ -n "${ADAPTER:-}" ]] && dir="${dir}_ft"
    [[ "${LIMIT:-0}" -gt 0 ]] && dir="${dir}_smoke${LIMIT}"
    echo "${dir}"
}

pfit_run_cmd() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY: $*"
    else
        echo "[RUN] $*"
        "$@"
    fi
}

pfit_phase_build() {
    local exp_id="$1" exp_ds="$2" exp_dir="$3" build_args="$4"
    local extra=""
    [[ "${LIMIT:-0}" -gt 0 ]] && extra="--sample ${LIMIT}"   # 追加在末尾,覆盖注册表 --sample
    # shellcheck disable=SC2086
    pfit_run_cmd "${PY:-python}" -m kgqa.pfit.build \
        --dataset "${exp_ds}" --input "${PFIT_RETRIEVE_TRAIN}" \
        --exp_dir "${exp_dir}" ${build_args} ${extra}
}

pfit_phase_train() {
    local exp_dir="$1"
    pfit_run_cmd "${PY:-python}" -m kgqa.pfit.train \
        --exp_dir "${exp_dir}" --epochs "${EPOCHS:-2}"
}

pfit_phase_eval() {
    local exp_id="$1" exp_ds="$2" exp_dir="$3" exp_type="$4" eval_args="$5"
    local adapter_flag=""
    if [[ -n "${ADAPTER:-}" ]]; then
        adapter_flag="--adapter ${ADAPTER}"
    elif [[ "${exp_type}" == "train" ]]; then
        adapter_flag="--adapter ${exp_dir}/adapter"
    fi
    local extra=""
    [[ "${LIMIT:-0}" -gt 0 ]] && extra="--limit ${LIMIT}"
    [[ -n "${FMT:-}" ]] && eval_args="${eval_args} --format ${FMT}"   # 末位覆盖
    # shellcheck disable=SC2086
    pfit_run_cmd "${PY:-python}" -m kgqa.pfit.eval \
        --dataset "${exp_ds}" --input "${PFIT_RETRIEVE_TEST}" \
        --exp_dir "${exp_dir}" ${adapter_flag} ${eval_args} ${extra}
}
