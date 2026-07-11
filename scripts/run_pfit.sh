#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pfit.sh — pfit 实验编排(stage2)
#
# 实验注册表(8 个;断点续跑由各模块 manifest 保证,脚本本身无状态):
#   webqsp_main           训练   chain+name+v2(Ch4 最优 groupAname_v2 同配置,parity 锚)
#   webqsp_spot_nl        训练   nl+name+v2(消融通路抽查,对照 Ch4 groupD)
#   webqsp_base_zeroshot  仅评测 base 零样本 chain+name(FMT=v1 换输出格式变体)
#   webqsp_nopaths        仅评测 无路径基线(ADAPTER=... 得微调无路径变体)
#   metaqa_main           训练   chain+name+v2,5K 分层混合跳数(核心新数字)
#   metaqa_spot_nl        训练   nl+name+v2
#   metaqa_base_zeroshot  仅评测 base 零样本
#   metaqa_nopaths        仅评测 无路径基线
#
# 用法:
#   bash scripts/run_pfit.sh --exp webqsp_main --phase all      # build→train→eval
#   bash scripts/run_pfit.sh --exp metaqa_main --phase build
#   LIMIT=100 bash scripts/run_pfit.sh --exp webqsp_main --phase all   # smoke
#   FMT=v1 bash scripts/run_pfit.sh --exp webqsp_base_zeroshot         # 输出格式变体
#   ADAPTER=<dir> bash scripts/run_pfit.sh --exp webqsp_nopaths        # 微调无路径变体
#
# 环境变量:LIMIT(smoke 条数,0=全量)、EPOCHS(默认 2)、FMT、ADAPTER、DRY_RUN、PY
# 前置数据(Task9/10 准备):data/output/kgqa/<ds>/retrieve/{train*,test}.jsonl
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/run_pfit_lib.sh"

pfit_exp_config() {
    # 设置 EXP_DS / EXP_TYPE(train|eval_only)/ EXP_BUILD_ARGS / EXP_EVAL_ARGS
    case "$1" in
        webqsp_main)
            EXP_DS=webqsp; EXP_TYPE=train
            EXP_BUILD_ARGS="--format v2 --path_format chain --entity_repr name"
            EXP_EVAL_ARGS="--format v2 --path_format chain --entity_repr name"
            ;;
        webqsp_spot_nl)
            EXP_DS=webqsp; EXP_TYPE=train
            EXP_BUILD_ARGS="--format v2 --path_format nl --entity_repr name"
            EXP_EVAL_ARGS="--format v2 --path_format nl --entity_repr name"
            ;;
        webqsp_base_zeroshot)
            EXP_DS=webqsp; EXP_TYPE=eval_only
            EXP_BUILD_ARGS=""
            EXP_EVAL_ARGS="--format v2 --path_format chain --entity_repr name"
            ;;
        webqsp_nopaths)
            EXP_DS=webqsp; EXP_TYPE=eval_only
            EXP_BUILD_ARGS=""
            EXP_EVAL_ARGS="--format v1 --no_paths --entity_repr name"
            ;;
        metaqa_main)
            EXP_DS=metaqa; EXP_TYPE=train
            EXP_BUILD_ARGS="--format v2 --path_format chain --sample 5000 --stratify_by_hop"
            EXP_EVAL_ARGS="--format v2 --path_format chain"
            ;;
        metaqa_spot_nl)
            EXP_DS=metaqa; EXP_TYPE=train
            EXP_BUILD_ARGS="--format v2 --path_format nl --sample 5000 --stratify_by_hop"
            EXP_EVAL_ARGS="--format v2 --path_format nl"
            ;;
        metaqa_base_zeroshot)
            EXP_DS=metaqa; EXP_TYPE=eval_only
            EXP_BUILD_ARGS=""
            EXP_EVAL_ARGS="--format v2 --path_format chain"
            ;;
        metaqa_nopaths)
            EXP_DS=metaqa; EXP_TYPE=eval_only
            EXP_BUILD_ARGS=""
            EXP_EVAL_ARGS="--format v1 --no_paths"
            ;;
        *)
            echo "[ERROR] 未注册实验: $1" >&2
            return 1
            ;;
    esac
}

pfit_run_exp() {
    local exp_id="$1" phase="$2"
    pfit_exp_config "${exp_id}"
    pfit_init_context "${PROJECT_DIR}" "${EXP_DS}"
    local exp_dir
    exp_dir="$(pfit_exp_dir "${exp_id}")"
    echo "==== ${exp_id}  type=${EXP_TYPE}  phase=${phase}  dir=${exp_dir} ===="

    if [[ "${EXP_TYPE}" == "train" && ( "${phase}" == "all" || "${phase}" == "build" ) ]]; then
        pfit_phase_build "${exp_id}" "${EXP_DS}" "${exp_dir}" "${EXP_BUILD_ARGS}"
    fi
    if [[ "${EXP_TYPE}" == "train" && ( "${phase}" == "all" || "${phase}" == "train" ) ]]; then
        pfit_phase_train "${exp_dir}"
    fi
    if [[ "${phase}" == "all" || "${phase}" == "eval" ]]; then
        pfit_phase_eval "${exp_id}" "${EXP_DS}" "${exp_dir}" "${EXP_TYPE}" "${EXP_EVAL_ARGS}"
    fi
}

main() {
    local exp="" phase="all" prev=""
    for arg in "$@"; do
        if [[ "${prev}" == "--exp" ]]; then exp="${arg}"; fi
        if [[ "${prev}" == "--phase" ]]; then phase="${arg}"; fi
        prev="${arg}"
    done
    if [[ -z "${exp}" ]]; then
        echo "用法: bash scripts/run_pfit.sh --exp <exp_id> [--phase build|train|eval|all]" >&2
        exit 1
    fi
    if [[ "${phase}" != "all" && "${phase}" != "build" && "${phase}" != "train" && "${phase}" != "eval" ]]; then
        echo "[ERROR] --phase 仅支持: build|train|eval|all" >&2
        exit 1
    fi
    pfit_run_exp "${exp}" "${phase}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
