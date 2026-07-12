#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_offline_path_search.sh
#
# 两步式离线路径搜索实验脚本：
#   Step 1 (dump)   : 运行 kgqa.retrieve.cli.dump_scores，
#                     将模型中间得分矩阵写入缓存文件
#   Step 2 (search) : 运行 scripts/offline_path_search.py，离线重放路径搜索
#
# 特性：
#   - Step 1 支持断点跳过（缓存文件已存在则不重跑）
#   - Step 2 运行统一 MMR 路径检索，并支持收窄后的超参数网格
#   - 结果写入带时间戳的日志文件
#
# 用法：
#   # 单次运行（先 dump，再用默认参数 search）
#   bash scripts/run_offline_path_search.sh \
#       --ckpt data/ckpt/WebQSP/model.pt \
#       --input_dir data/input/WebQSP
#
#   # 只做 dump（不运行 search）
#   bash scripts/run_offline_path_search.sh \
#       --ckpt data/ckpt/WebQSP/model.pt \
#       --input_dir data/input/WebQSP \
#       --phase dump
#
#   # 只做 search（缓存已存在）
#   bash scripts/run_offline_path_search.sh \
#       --input_dir data/input/WebQSP \
#       --phase search \
#       --cache data/output/WebQSP/offline_search/score_cache/webqsp_val.pt
#
#   # 做超参数网格搜索
#   bash scripts/run_offline_path_search.sh \
#       --input_dir data/input/WebQSP \
#       --phase search \
#       --cache data/output/WebQSP/offline_search/score_cache/webqsp_val.pt \
#       --grid
#
#   # CWQ（默认 checkpoint/input/output 会自动切到 data/ckpt/CWQ、data/input/CWQ、data/output/CWQ）
#   bash scripts/run_offline_path_search.sh --dataset cwq --phase all
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 项目根目录（脚本所在目录的上一级）────────────────────────────────────────
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── 默认参数 ──────────────────────────────────────────────────────────────────
# 数据集：webqsp | cwq
DATASET="webqsp"
# checkpoint / 数据目录 / BERT 默认值按 DATASET 自动补齐；显式参数优先
CKPT=""
INPUT_DIR=""
# QA 文件；WebQSP 默认使用过滤掉无有效答案样本后的 1581 条测试集，CWQ 不需要
QA_FILE=""
# 得分缓存路径（空 = 自动推导为 OUTPUT_DIR/${dataset}_${MODE}.pt）
CACHE=""
MODE="val"
BERT_NAME=""
BATCH_SIZE=16
TOPK=500
PHASE="all"          # all | dump | search

# search 参数
ALPHA_FINAL="2.0"
THRESHOLD="0.01"
LAMBDA_VAL="0.2"
BEAM_SIZE="20"

# 只搜索当前正式实验超参，不恢复旧 scoring/selector 搜索空间。
GRID=0
GRID_ALPHAS="0.0 1.0 2.0"
GRID_LAMBDAS="0 0.2 0.5 0.7 1.0"
GRID_THRESHOLDS="0.01"
GRID_BEAMS="3 5 10 15 20 30 40 50"

# 缓存、日志与路径结果放在同一父目录下；空值按 DATASET 自动补齐
OFFLINE_DIR=""
OUTPUT_DIR=""
LOG_DIR=""
PATHS_DIR=""
SUMMARY_FILE=""

# ── 参数解析 ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)     DATASET="$2";     shift 2 ;;
        --ckpt)        CKPT="$2";        shift 2 ;;
        --input_dir)   INPUT_DIR="$2";   shift 2 ;;
        --qa_file)     QA_FILE="$2";     shift 2 ;;
        --cache)       CACHE="$2";       shift 2 ;;
        --mode)        MODE="$2";        shift 2 ;;
        --bert_name)   BERT_NAME="$2";   shift 2 ;;
        --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
        --topk)        TOPK="$2";        shift 2 ;;
        --phase)       PHASE="$2";       shift 2 ;;
        --alpha_final) ALPHA_FINAL="$2"; shift 2 ;;
        --threshold)   THRESHOLD="$2";   shift 2 ;;
        --lambda_val)  LAMBDA_VAL="$2";  shift 2 ;;
        --beam_size)   BEAM_SIZE="$2";   shift 2 ;;
        --grid)        GRID=1;           shift 1 ;;
        --grid_alphas) GRID_ALPHAS="$2"; shift 2 ;;
        --grid_lambdas) GRID_LAMBDAS="$2"; shift 2 ;;
        --grid_thresholds) GRID_THRESHOLDS="$2"; shift 2 ;;
        --grid_beams)  GRID_BEAMS="$2";  shift 2 ;;
        --offline_dir) OFFLINE_DIR="$2"; shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --log_dir)     LOG_DIR="$2";     shift 2 ;;
        --paths_dir)   PATHS_DIR="$2";   shift 2 ;;
        --summary_file) SUMMARY_FILE="$2"; shift 2 ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
done

# ── 数据集默认值 ──────────────────────────────────────────────────────────────
DATASET="${DATASET,,}"
case "$DATASET" in
    webqsp)
        CACHE_PREFIX="webqsp"
        [[ -z "$CKPT" ]] && CKPT="${PROJ_DIR}/data/ckpt/WebQSP_run_20260518_2241/model-49-0.7154.pt"
        [[ -z "$INPUT_DIR" ]] && INPUT_DIR="${PROJ_DIR}/data/input/WebQSP"
        [[ -z "$BERT_NAME" ]] && BERT_NAME="BAAI/bge-base-en-v1.5"
        [[ -z "$OFFLINE_DIR" ]] && OFFLINE_DIR="${PROJ_DIR}/data/output/WebQSP/offline_search"
        ;;
    cwq)
        CACHE_PREFIX="cwq"
        [[ -z "$CKPT" ]] && CKPT="${PROJ_DIR}/data/ckpt/CWQ/model-29-0.4206.pt"
        [[ -z "$INPUT_DIR" ]] && INPUT_DIR="${PROJ_DIR}/data/input/CWQ"
        [[ -z "$BERT_NAME" ]] && BERT_NAME="bert-base-cased"
        [[ -z "$OFFLINE_DIR" ]] && OFFLINE_DIR="${PROJ_DIR}/data/output/CWQ/offline_search"
        ;;
    *)
        echo "[ERROR] --dataset 仅支持: webqsp | cwq，当前值: ${DATASET}"
        exit 1
        ;;
esac

[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="${OFFLINE_DIR}/score_cache"
[[ -z "$LOG_DIR" ]] && LOG_DIR="${OFFLINE_DIR}/logs"
[[ -z "$PATHS_DIR" ]] && PATHS_DIR="${OFFLINE_DIR}/paths"
[[ -z "$SUMMARY_FILE" ]] && SUMMARY_FILE="${OFFLINE_DIR}/summary.csv"

# ── 自动推导缓存路径 ──────────────────────────────────────────────────────────
if [[ -z "$CACHE" ]]; then
    CACHE="${OUTPUT_DIR}/${CACHE_PREFIX}_${MODE}.pt"
fi
if [[ "$DATASET" == "webqsp" && -z "$QA_FILE" ]]; then
    if [[ "$MODE" == "train" ]]; then
        QA_FILE="${INPUT_DIR}/QA_data/WebQuestionsSP/qa_train_webqsp.txt"
    else
        QA_FILE="${INPUT_DIR}/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt"
    fi
fi
if [[ "$DATASET" == "cwq" && -z "$QA_FILE" ]]; then
    case "$MODE" in
        train) QA_FILE="${INPUT_DIR}/train_simple.json" ;;
        test)  QA_FILE="${INPUT_DIR}/test_simple.json" ;;
        *)     QA_FILE="${INPUT_DIR}/dev_simple.json" ;;
    esac
fi

# 将路径归一成绝对路径，避免 dump_scores 对相对 qa_file 再拼 input_dir，
# 产生 data/input/WebQSP/data/input/WebQSP/... 这种双重前缀。
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
if [[ -n "$QA_FILE" && "$QA_FILE" != /* ]]; then
    if [[ -f "$QA_FILE" ]]; then
        QA_FILE="$(cd "$(dirname "$QA_FILE")" && pwd)/$(basename "$QA_FILE")"
    else
        QA_FILE="${INPUT_DIR}/${QA_FILE}"
    fi
fi

# ── 工具函数 ──────────────────────────────────────────────────────────────────
ts() { date '+%Y-%m-%d %H:%M:%S'; }

print_header() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════"
}

metric_from_log() {
    local label="$1"
    local log_file="$2"
    grep -F "$label" "$log_file" | tail -1 | awk -F':' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}'
}

# ── Step 1: dump ──────────────────────────────────────────────────────────────
run_dump() {
    print_header "Step 1: 导出得分缓存"
    echo "  ckpt       : ${CKPT}"
    echo "  dataset    : ${DATASET}"
    echo "  input_dir  : ${INPUT_DIR}"
    echo "  qa_file    : ${QA_FILE}"
    echo "  mode       : ${MODE}"
    echo "  output     : ${CACHE}"
    echo "  topk       : ${TOPK}"
    echo ""

    if [[ -z "$CKPT" ]]; then
        echo "[ERROR] --ckpt 未指定，无法运行 dump。"
        exit 1
    fi
    if [[ -z "$INPUT_DIR" ]]; then
        echo "[ERROR] --input_dir 未指定。"
        exit 1
    fi
    if [[ ! -f "$QA_FILE" ]]; then
        echo "[ERROR] QA 文件不存在: ${QA_FILE}"
        exit 1
    fi

    if [[ -f "$CACHE" ]]; then
        echo "[INFO] 缓存已存在，跳过 dump: ${CACHE}"
        return 0
    fi

    mkdir -p "$(dirname "$CACHE")"

    local t0=$SECONDS
    echo "[$(ts)] 开始 dump ..."
    local dump_args=(
        --dataset    "$DATASET" \
        --ckpt       "$CKPT" \
        --input_dir  "$INPUT_DIR" \
        --qa_file    "$QA_FILE" \
        --split      "$MODE" \
        --bert_name  "$BERT_NAME" \
        --batch_size "$BATCH_SIZE" \
        --topk       "$TOPK" \
        --output     "$CACHE"
    )
    python -m kgqa.retrieve.cli.dump_scores "${dump_args[@]}"
    echo "[$(ts)] dump 完成，耗时 $((SECONDS - t0))s"
}

# ── Step 2: 单次 search ───────────────────────────────────────────────────────
run_search_once() {
    local alpha_final="$1"
    local threshold="$2"
    local lambda_val="$3"
    local beam_size="$4"
    local log_file="$5"
    local output_jsonl="$6"   # 可选，空串表示不写 JSONL

    echo "[$(ts)] alpha_final=${alpha_final} threshold=${threshold} lambda=${lambda_val} beam=${beam_size}"

    local output_args=()
    if [[ -n "$output_jsonl" ]]; then
        mkdir -p "$(dirname "$output_jsonl")"
        output_args=(--output "$output_jsonl")
    fi

    python scripts/offline_path_search.py \
        --cache       "$CACHE" \
        --input_dir   "$INPUT_DIR" \
        --alpha_final "$alpha_final" \
        --threshold   "$threshold" \
        --lambda_val  "$lambda_val" \
        --beam_size   "$beam_size" \
        "${output_args[@]}" \
        | tee -a "$log_file"
    echo "" >> "$log_file"
}

# ── Step 2: search ───────────────────────────────────────────────────────────
run_search() {
    print_header "Step 2: 离线路径搜索"
    echo "  cache      : ${CACHE}"
    echo "  input_dir  : ${INPUT_DIR}"

    if [[ -z "$INPUT_DIR" ]]; then
        echo "[ERROR] --input_dir 未指定。"
        exit 1
    fi
    if [[ ! -f "$CACHE" ]]; then
        echo "[ERROR] 缓存文件不存在: ${CACHE}，请先运行 --phase dump。"
        exit 1
    fi

    mkdir -p "$LOG_DIR"
    local timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')

    if [[ "$GRID" -eq 1 ]]; then
        local log_file="${LOG_DIR}/grid_${timestamp}.log"
        echo "  模式       : 网格搜索"
        echo "  alphas     : ${GRID_ALPHAS}"
        echo "  lambdas    : ${GRID_LAMBDAS}"
        echo "  thresholds : ${GRID_THRESHOLDS}"
        echo "  beams      : ${GRID_BEAMS}"
        echo "  log        : ${log_file}"
        echo "  summary    : ${SUMMARY_FILE}"
        echo ""

        echo "# 网格搜索  $(ts)" > "$log_file"
        echo "# cache=${CACHE}  input_dir=${INPUT_DIR}" >> "$log_file"
        echo "" >> "$log_file"
        mkdir -p "$(dirname "$SUMMARY_FILE")"
        echo "alpha_final,lambda_val,threshold,beam_size,total,empty_path,answer_hit,top1_hit,precision,recall,f1,jaccard_diversity,relation_jaccard_diversity,tail_diversity,relation_coverage,edge_coverage,elapsed_s,jsonl_path" > "$SUMMARY_FILE"

        local t0=$SECONDS
        local count=0
        for alpha in $GRID_ALPHAS; do
            for lam in $GRID_LAMBDAS; do
                for thresh in $GRID_THRESHOLDS; do
                    for beam in $GRID_BEAMS; do
                        local alpha_fmt
                        alpha_fmt=$(printf '%s' "$alpha" | sed 's/\.*0*$//' | sed 's/^\./0./')
                        [[ -z "$alpha_fmt" ]] && alpha_fmt="0"
                        local lam_fmt
                        lam_fmt=$(printf '%s' "$lam" | sed 's/\.*0*$//' | sed 's/^\./0./')
                        [[ -z "$lam_fmt" ]] && lam_fmt="0"
                        local jsonl_path="${PATHS_DIR}/beam${beam}_alpha${alpha_fmt}_lam${lam_fmt}.jsonl"
                        local iter_start=$SECONDS
                        echo "─── [alpha=${alpha} lambda=${lam} threshold=${thresh} beam=${beam}] ───────────────────" >> "$log_file"
                        run_search_once "$alpha" "$thresh" "$lam" "$beam" "$log_file" "$jsonl_path"
                        local iter_elapsed=$((SECONDS - iter_start))
                        local total empty_path answer_hit top1_hit precision recall f1 edge_div rel_div tail_unique rel_cov edge_cov
                        total=$(metric_from_log "总样本数" "$log_file")
                        empty_path=$(metric_from_log "空路径数" "$log_file")
                        answer_hit=$(metric_from_log "Answer Hit" "$log_file")
                        top1_hit=$(metric_from_log "Top-1 Hit" "$log_file")
                        precision=$(metric_from_log "Precision" "$log_file")
                        recall=$(metric_from_log "Recall" "$log_file")
                        f1=$(metric_from_log "F1" "$log_file")
                        edge_div=$(metric_from_log "Edge Diversity" "$log_file")
                        rel_div=$(metric_from_log "Relation Diversity" "$log_file")
                        tail_unique=$(metric_from_log "Tail Unique" "$log_file")
                        rel_cov=$(metric_from_log "Relation Coverage" "$log_file")
                        edge_cov=$(metric_from_log "Edge Coverage" "$log_file")
                        echo "${alpha},${lam},${thresh},${beam},${total},${empty_path},${answer_hit},${top1_hit},${precision},${recall},${f1},${edge_div},${rel_div},${tail_unique},${rel_cov},${edge_cov},${iter_elapsed},${jsonl_path}" >> "$SUMMARY_FILE"
                        count=$((count + 1))
                    done
                done
            done
        done
        echo "[$(ts)] 网格搜索完成，共 ${count} 组，耗时 $((SECONDS - t0))s"
        echo "[INFO] 完整日志: ${log_file}"
        echo "[INFO] 汇总表: ${SUMMARY_FILE}"
        return 0
    fi

    local log_file="${LOG_DIR}/single_${timestamp}.log"
    echo "  模式       : 单次"
    echo "  alpha_final: ${ALPHA_FINAL}"
    echo "  threshold  : ${THRESHOLD}"
    echo "  lambda_val : ${LAMBDA_VAL}"
    echo "  beam_size  : ${BEAM_SIZE}"
    echo "  log        : ${log_file}"
    echo ""

    echo "# 单次搜索  $(ts)" > "$log_file"
    echo "# cache=${CACHE}  input_dir=${INPUT_DIR}" >> "$log_file"
    echo "" >> "$log_file"

    local t0=$SECONDS
    local lam_fmt
    lam_fmt=$(printf '%s' "$LAMBDA_VAL" | sed 's/\.*0*$//' | sed 's/^\./0./')
    [[ -z "$lam_fmt" ]] && lam_fmt="0"
    local jsonl_path="${PATHS_DIR}/beam${BEAM_SIZE}_lam${lam_fmt}.jsonl"
    run_search_once "$ALPHA_FINAL" "$THRESHOLD" "$LAMBDA_VAL" "$BEAM_SIZE" "$log_file" "$jsonl_path"
    echo "[$(ts)] search 完成，耗时 $((SECONDS - t0))s"
    echo "[INFO] 完整日志: ${log_file}"
}

# ── 主流程 ────────────────────────────────────────────────────────────────────
case "$PHASE" in
    all)
        run_dump
        run_search
        ;;
    dump)
        run_dump
        ;;
    search)
        run_search
        ;;
    *)
        echo "[ERROR] --phase 必须为 all | dump | search，当前值: ${PHASE}"
        exit 1
        ;;
esac
