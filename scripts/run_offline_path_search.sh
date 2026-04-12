#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_offline_path_search.sh
#
# 两步式离线路径搜索实验脚本：
#   Step 1 (dump)   : 运行 WebQSP.dump_scores，将模型中间得分矩阵写入缓存文件
#   Step 2 (search) : 运行 scripts/offline_path_search.py，离线重放路径搜索
#
# 特性：
#   - Step 1 支持断点跳过（缓存文件已存在则不重跑）
#   - Step 2 运行 tail_blend 新实验或 baseline，并支持收窄后的超参数网格
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
#   # 运行 baseline 方法
#   bash scripts/run_offline_path_search.sh \
#       --input_dir data/input/WebQSP \
#       --phase search \
#       --cache data/output/WebQSP/offline_search/score_cache/webqsp_val.pt \
#       --method baseline
#
#   # 对 tail_blend 实验做超参数网格搜索
#   bash scripts/run_offline_path_search.sh \
#       --input_dir data/input/WebQSP \
#       --phase search \
#       --cache data/output/WebQSP/offline_search/score_cache/webqsp_val.pt \
#       --grid
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 项目根目录（脚本所在目录的上一级）────────────────────────────────────────
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── 默认参数 ──────────────────────────────────────────────────────────────────
# 最佳 WebQSP checkpoint（acc=0.6411 @ epoch 29）
CKPT="${PROJ_DIR}/data/ckpt/WebQSP/model-29-0.6411.pt"
# WebQSP 数据目录
INPUT_DIR="${PROJ_DIR}/data/input/WebQSP"
# 得分缓存路径（空 = 自动推导为 OUTPUT_DIR/webqsp_${MODE}.pt）
CACHE=""
MODE="val"
BERT_NAME="bert-base-uncased"
BATCH_SIZE=16
TOPK=500
PHASE="all"          # all | dump | search

# search 参数
METHOD="tail_blend"
ALPHA_FINAL="2.0"
THRESHOLD="0.01"
LAMBDA_VAL="0.2"
BEAM_SIZE="20"

# tail_blend 实验网格搜索。只搜索正式实验超参，不恢复旧 scoring/selector 搜索空间。
GRID=0
GRID_ALPHAS="0.0 1.0 2.0"
GRID_LAMBDAS="0 0.2 0.5 0.7 1.0"
GRID_THRESHOLDS="0.01"
GRID_BEAMS="3 5 10 15 20 30 40 50"

# 缓存、日志与路径结果放在同一父目录下
OFFLINE_DIR="${PROJ_DIR}/data/output/WebQSP/offline_search"
OUTPUT_DIR="${OFFLINE_DIR}/score_cache"
LOG_DIR="${OFFLINE_DIR}/logs"
PATHS_DIR="${OFFLINE_DIR}/paths"

# ── 参数解析 ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)        CKPT="$2";        shift 2 ;;
        --input_dir)   INPUT_DIR="$2";   shift 2 ;;
        --cache)       CACHE="$2";       shift 2 ;;
        --mode)        MODE="$2";        shift 2 ;;
        --bert_name)   BERT_NAME="$2";   shift 2 ;;
        --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
        --topk)        TOPK="$2";        shift 2 ;;
        --phase)       PHASE="$2";       shift 2 ;;
        --method)      METHOD="$2";      shift 2 ;;
        --alpha_final) ALPHA_FINAL="$2"; shift 2 ;;
        --threshold)   THRESHOLD="$2";   shift 2 ;;
        --lambda_val)  LAMBDA_VAL="$2";  shift 2 ;;
        --beam_size)   BEAM_SIZE="$2";   shift 2 ;;
        --grid)        GRID=1;           shift 1 ;;
        --grid_alphas) GRID_ALPHAS="$2"; shift 2 ;;
        --grid_lambdas) GRID_LAMBDAS="$2"; shift 2 ;;
        --grid_thresholds) GRID_THRESHOLDS="$2"; shift 2 ;;
        --grid_beams)  GRID_BEAMS="$2";  shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --paths_dir)   PATHS_DIR="$2";   shift 2 ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
done

# ── 自动推导缓存路径 ──────────────────────────────────────────────────────────
if [[ -z "$CACHE" ]]; then
    CACHE="${OUTPUT_DIR}/webqsp_${MODE}.pt"
fi

# ── 工具函数 ──────────────────────────────────────────────────────────────────
ts() { date '+%Y-%m-%d %H:%M:%S'; }

print_header() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════"
}

# ── Step 1: dump ──────────────────────────────────────────────────────────────
run_dump() {
    print_header "Step 1: 导出得分缓存"
    echo "  ckpt       : ${CKPT}"
    echo "  input_dir  : ${INPUT_DIR}"
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

    if [[ -f "$CACHE" ]]; then
        echo "[INFO] 缓存已存在，跳过 dump: ${CACHE}"
        return 0
    fi

    mkdir -p "$(dirname "$CACHE")"

    local t0=$SECONDS
    echo "[$(ts)] 开始 dump ..."
    python -m WebQSP.dump_scores \
        --ckpt       "$CKPT" \
        --input_dir  "$INPUT_DIR" \
        --mode       "$MODE" \
        --bert_name  "$BERT_NAME" \
        --batch_size "$BATCH_SIZE" \
        --topk       "$TOPK" \
        --output     "$CACHE"
    echo "[$(ts)] dump 完成，耗时 $((SECONDS - t0))s"
}

# ── Step 2: 单次 search ───────────────────────────────────────────────────────
run_search_once() {
    local method="$1"
    local alpha_final="$2"
    local threshold="$3"
    local lambda_val="$4"
    local beam_size="$5"
    local log_file="$6"
    local output_jsonl="$7"   # 可选，空串表示不写 JSONL

    echo "[$(ts)] method=${method} alpha_final=${alpha_final} threshold=${threshold} lambda=${lambda_val} beam=${beam_size}"

    local output_args=()
    if [[ -n "$output_jsonl" ]]; then
        mkdir -p "$(dirname "$output_jsonl")"
        output_args=(--output "$output_jsonl")
    fi

    python scripts/offline_path_search.py \
        --cache       "$CACHE" \
        --input_dir   "$INPUT_DIR" \
        --method      "$method" \
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
        echo "  method     : ${METHOD}"
        echo "  alphas     : ${GRID_ALPHAS}"
        echo "  lambdas    : ${GRID_LAMBDAS}"
        echo "  thresholds : ${GRID_THRESHOLDS}"
        echo "  beams      : ${GRID_BEAMS}"
        echo "  log        : ${log_file}"
        echo ""

        echo "# 网格搜索  $(ts)" > "$log_file"
        echo "# cache=${CACHE}  input_dir=${INPUT_DIR}  method=${METHOD}" >> "$log_file"
        echo "" >> "$log_file"

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
                        local jsonl_path="${PATHS_DIR}/${METHOD}_beam${beam}_alpha${alpha_fmt}_lam${lam_fmt}.jsonl"
                        echo "─── [method=${METHOD} alpha=${alpha} lambda=${lam} threshold=${thresh} beam=${beam}] ───────────────────" >> "$log_file"
                        run_search_once "$METHOD" "$alpha" "$thresh" "$lam" "$beam" "$log_file" "$jsonl_path"
                        count=$((count + 1))
                    done
                done
            done
        done
        echo "[$(ts)] 网格搜索完成，共 ${count} 组，耗时 $((SECONDS - t0))s"
        echo "[INFO] 完整日志: ${log_file}"
        return 0
    fi

    local log_file="${LOG_DIR}/single_${timestamp}.log"
    echo "  模式       : 单次"
    echo "  method     : ${METHOD}"
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
    local jsonl_path="${PATHS_DIR}/${METHOD}_beam${BEAM_SIZE}_lam${lam_fmt}.jsonl"
    run_search_once "$METHOD" "$ALPHA_FINAL" "$THRESHOLD" "$LAMBDA_VAL" "$BEAM_SIZE" "$log_file" "$jsonl_path"
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
