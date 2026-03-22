#!/usr/bin/env bash
# Grid search: beam_size x lambda_val for MMR path retrieval
#
# 用法：
#   bash scripts/run_grid.sh <dataset> [ckpt_path]
#
# dataset: webqsp | metaqa | cwq
#
# 示例：
#   bash scripts/run_grid.sh webqsp
#   bash scripts/run_grid.sh metaqa
#   bash scripts/run_grid.sh cwq
#   bash scripts/run_grid.sh webqsp data/ckpt/WebQSP/model-29-0.6411.pt

set -euo pipefail

DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
    echo "Usage: bash scripts/run_grid.sh <dataset> [ckpt_path]"
    echo "  dataset: webqsp | metaqa | cwq"
    exit 1
fi

case "$DATASET" in
    webqsp)
        DEFAULT_CKPT="data/ckpt/WebQSP/model-29-0.6411.pt"
        INPUT_DIR="data/input/WebQSP"
        OUTPUT_BASE="data/output/WebQSP/grid_search"
        MODULE="WebQSP.predict"
        MODE="test"
        ;;
    metaqa)
        DEFAULT_CKPT="data/ckpt/MetaQA_KB/model_epoch-6_acc-0.9937.pt"
        INPUT_DIR="data/input/MetaQA_KB"
        OUTPUT_BASE="data/output/MetaQA_KB/grid_search"
        MODULE="MetaQA_KB.predict"
        MODE="val"
        ;;
    cwq)
        DEFAULT_CKPT="data/ckpt/CWQ/model-29-0.4206.pt"
        INPUT_DIR="data/input/CWQ"
        OUTPUT_BASE="data/output/CWQ/grid_search"
        MODULE="CompWebQ.predict"
        MODE="val"
        ;;
    *)
        echo "Unknown dataset: $DATASET (choose webqsp, metaqa, or cwq)"
        exit 1
        ;;
esac

CKPT="${2:-$DEFAULT_CKPT}"

BEAM_SIZES="3,5,7,10,20,30,40,50"
LAMBDA_VALS="0.0,0.2,0.5,0.7,1.0"

LOG_DIR="${OUTPUT_BASE}/logs"
PATHS_DIR="${OUTPUT_BASE}/paths"
SUMMARY_FILE="${OUTPUT_BASE}/summary.csv"

mkdir -p "$LOG_DIR" "$PATHS_DIR"

IFS=',' read -ra BEAM_ARR  <<< "$BEAM_SIZES"
IFS=',' read -ra LAMBDA_ARR <<< "$LAMBDA_VALS"

echo "dataset=${DATASET}  ckpt=${CKPT}"
echo "beam_size,lambda_val,qa_acc,mmr_answer_path_hit,mmr_precision,mmr_answer_recall,mmr_f1,mmr_top1_hit,jaccard_diversity,tail_diversity,edge_coverage,elapsed_s" > "$SUMMARY_FILE"

SCRIPT_START=$SECONDS
total=$(( ${#BEAM_ARR[@]} * ${#LAMBDA_ARR[@]} ))
done_count=0

for beam in "${BEAM_ARR[@]}"; do
    for lam in "${LAMBDA_ARR[@]}"; do
        done_count=$(( done_count + 1 ))
        ITER_START=$SECONDS
        tag="beam${beam}_lam${lam}"
        log_file="${LOG_DIR}/${tag}.log"

        echo "=========================================="
        echo "[${done_count}/${total}] beam_size=${beam}, lambda=${lam}"
        echo "=========================================="

        out_jsonl="${PATHS_DIR}/${tag}.jsonl"

        python -m "$MODULE" \
            --ckpt "$CKPT" \
            --input_dir "$INPUT_DIR" \
            --mode "$MODE" \
            --beam_size "$beam" \
            --lambda_val "$lam" \
            --output_path "$out_jsonl" \
            --no_compare_standard \
            2>&1 | tee "$log_file"
        ITER_ELAPSED=$(( SECONDS - ITER_START ))

        qa_acc=$(grep -E '^[01]\.[0-9]+$' "$log_file" | tail -1 || echo "N/A")
        answer_hit=$(grep "mmr_answer_path_hit" "$log_file" \
                     | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "N/A")
        precision=$(grep "mmr_precision" "$log_file" \
                    | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "N/A")
        answer_recall=$(grep "mmr_answer_recall" "$log_file" \
                        | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "N/A")
        f1=$(grep "mmr_f1" "$log_file" \
             | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "N/A")
        top1_hit=$(grep "mmr_top1_hit" "$log_file" \
                   | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "N/A")
        jaccard_div=$(grep "jaccard_diversity" "$log_file" \
                      | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "N/A")
        tail_div=$(grep "tail_diversity" "$log_file" \
                   | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "N/A")
        edge_cov=$(grep "edge_coverage" "$log_file" \
                   | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "N/A")

        echo "${beam},${lam},${qa_acc},${answer_hit},${precision},${answer_recall},${f1},${top1_hit},${jaccard_div},${tail_div},${edge_cov},${ITER_ELAPSED}" >> "$SUMMARY_FILE"
        echo "  -> qa_acc=${qa_acc}  hit@${beam}=${answer_hit}  P=${precision}  R=${answer_recall}  F1=${f1}  top1=${top1_hit}  jaccard=${jaccard_div}  tail=${tail_div}  edge_cov=${edge_cov}  耗时=${ITER_ELAPSED}s"
        echo "  -> paths saved: ${out_jsonl}"
    done
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "=========================================="
echo "All done. Results:"
echo "  Summary : $SUMMARY_FILE"
echo "  Paths   : $PATHS_DIR/"
echo "  Logs    : $LOG_DIR/"
echo "  总耗时 : ${TOTAL_ELAPSED}s  ($(( TOTAL_ELAPSED/60 ))min $(( TOTAL_ELAPSED%60 ))s)"
echo "=========================================="
column -t -s',' "$SUMMARY_FILE"
