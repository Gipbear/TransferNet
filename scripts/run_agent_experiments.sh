#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# KG-ReAct Agent Experiment Runner
# Chapter 5 evaluation experiments
#
# Runs 5 experiments for the thesis Chapter 5 evaluation:
#   Exp 1: Full test set — Agent vs V2-SFT vs V0 (zero-shot)
#   Exp 2: Hard subset analysis (questions where Ch4-SFT failed)
#   Exp 3: Ablation study (max_steps sweep, default_k sweep)
#   Exp 4: Efficiency analysis (step distribution, tool call counts)
#
# Usage:
#   bash scripts/run_agent_experiments.sh                        # all experiments
#   PREDICT_JSONL=/path/to/test.jsonl bash scripts/run_agent_experiments.sh
#   OUTPUT_DIR=results/my_run bash scripts/run_agent_experiments.sh
#
# All paths and params are configurable via environment variables.
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Configuration — override any of these via environment variables
# ---------------------------------------------------------------------------
PREDICT_JSONL="${PREDICT_JSONL:-${PROJECT_DIR}/data/webqsp_test_mmr.jsonl}"    # Ch3 MMR output
SFT_RESULTS="${SFT_RESULTS:-${PROJECT_DIR}/results/ch4_sft_v2_results.jsonl}"  # Ch4 SFT results
ADAPTER_DIR="${ADAPTER_DIR:-${PROJECT_DIR}/models/webqsp_sft_v2}"              # LoRA adapter
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/agent}"
MODEL="${MODEL:-unsloth/meta-llama-3.1-8b-instruct-bnb-4bit}"
MAX_STEPS="${MAX_STEPS:-8}"
DEFAULT_K="${DEFAULT_K:-10}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Helper: print section header with timestamp
# ---------------------------------------------------------------------------
section() {
    echo
    echo "==========================================================================="
    echo "  $*"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==========================================================================="
    echo
}

# ---------------------------------------------------------------------------
# Startup summary
# ---------------------------------------------------------------------------
section "KG-ReAct Agent Experiment Runner"
echo "  PROJECT_DIR   : ${PROJECT_DIR}"
echo "  PREDICT_JSONL : ${PREDICT_JSONL}"
echo "  SFT_RESULTS   : ${SFT_RESULTS}"
echo "  ADAPTER_DIR   : ${ADAPTER_DIR}"
echo "  OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "  MODEL         : ${MODEL}"
echo "  MAX_STEPS     : ${MAX_STEPS}"
echo "  DEFAULT_K     : ${DEFAULT_K}"
echo "  DEVICE        : ${DEVICE}"
echo

if [[ ! -f "${PREDICT_JSONL}" ]]; then
    echo "[ERROR] Test JSONL not found: ${PREDICT_JSONL}"
    echo "        Set PREDICT_JSONL=/path/to/file.jsonl to override."
    exit 1
fi

if [[ ! -d "${ADAPTER_DIR}" ]]; then
    echo "[ERROR] LoRA adapter directory not found: ${ADAPTER_DIR}"
    echo "        Set ADAPTER_DIR=/path/to/adapter to override."
    exit 1
fi

WALL_START=$(date +%s)

# ===========================================================================
# Experiment 1: Full test set — Agent vs V2-SFT vs V0 (zero-shot)
# ===========================================================================
section "Experiment 1: Full test set comparison (Agent / V2-SFT / V0 zero-shot)"

# 1a. KG-ReAct Agent (Chapter 5 main result)
EXP1_AGENT="${OUTPUT_DIR}/exp1_agent.jsonl"
if [[ -f "${EXP1_AGENT}" ]]; then
    echo "[SKIP] Agent results already exist: ${EXP1_AGENT}"
else
    echo "[RUN] 1a. KG-ReAct Agent inference ..."
    T0=$(date +%s)
    python -m llm_infer.agent.eval_agent \
        --input         "${PREDICT_JSONL}" \
        --output        "${EXP1_AGENT}" \
        --adapter       "${ADAPTER_DIR}" \
        --model         "${MODEL}" \
        --max_steps     "${MAX_STEPS}" \
        --default_k     "${DEFAULT_K}" \
        --output_format v2 \
        --device        "${DEVICE}"
    echo "[INFO] 1a done, elapsed $(($(date +%s) - T0))s"
fi

# 1b. V0 Zero-shot baseline (Chapter 3 reference point)
EXP1_V0="${OUTPUT_DIR}/exp1_v0_zeroshot.jsonl"
if [[ -f "${EXP1_V0}" ]]; then
    echo "[SKIP] V0 zero-shot results already exist: ${EXP1_V0}"
else
    echo "[RUN] 1b. V0 zero-shot baseline inference ..."
    T0=$(date +%s)
    python -m llm_infer.eval_faithfulness \
        --input         "${PREDICT_JSONL}" \
        --output        "${OUTPUT_DIR}/exp1_v0_zeroshot" \
        --model         "${MODEL}" \
        --output_format v0 \
        --device        "${DEVICE}"
    echo "[INFO] 1b done, elapsed $(($(date +%s) - T0))s"
fi

# 1c. V2-SFT baseline (Chapter 4 main result)
# Reuse Ch4 results if they already exist; otherwise run inference and save locally.
if [[ -f "${SFT_RESULTS}" ]]; then
    echo "[SKIP] Ch4 SFT results found at ${SFT_RESULTS} — reusing, no re-inference needed."
    EXP1_SFT="${SFT_RESULTS}"
else
    EXP1_SFT="${OUTPUT_DIR}/exp1_v2_sft.jsonl"
    if [[ -f "${EXP1_SFT}" ]]; then
        echo "[SKIP] V2-SFT results already exist: ${EXP1_SFT}"
    else
        echo "[RUN] 1c. V2-SFT baseline inference (Ch4 results not found) ..."
        T0=$(date +%s)
        python -m llm_infer.eval_faithfulness \
            --input         "${PREDICT_JSONL}" \
            --output        "${OUTPUT_DIR}/exp1_v2_sft" \
            --adapter       "${ADAPTER_DIR}" \
            --model         "${MODEL}" \
            --output_format v2 \
            --device        "${DEVICE}"
        echo "[INFO] 1c done, elapsed $(($(date +%s) - T0))s"
    fi
fi

# ===========================================================================
# Experiment 2: Hard subset analysis
# Questions where Ch4-SFT failed (F1=0 or hallucination_rate>0)
# ===========================================================================
section "Experiment 2: Hard subset analysis"

HARD_SUBSET="${OUTPUT_DIR}/hard_subset.txt"

if [[ -f "${HARD_SUBSET}" ]]; then
    echo "[SKIP] Hard subset file already exists: ${HARD_SUBSET}"
else
    echo "[RUN] Generating hard subset from SFT results: ${EXP1_SFT} ..."
    python - "${EXP1_SFT}" "${HARD_SUBSET}" <<'PYEOF'
import json
import sys

sft_results_path = sys.argv[1]
hard_file_path   = sys.argv[2]

hard_questions = []
with open(sft_results_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("f1", 1.0) == 0.0 or r.get("hallucination_rate", 0.0) > 0:
            hard_questions.append(r.get("question", ""))

with open(hard_file_path, "w", encoding="utf-8") as f:
    f.write("\n".join(hard_questions))

print(f"Hard subset size: {len(hard_questions)} samples written to {hard_file_path}")
PYEOF
fi

EXP2_AGENT_HARD="${OUTPUT_DIR}/exp2_agent_hard.jsonl"
if [[ -f "${EXP2_AGENT_HARD}" ]]; then
    echo "[SKIP] Hard subset agent results already exist: ${EXP2_AGENT_HARD}"
else
    echo "[RUN] 2. Agent inference on hard subset ..."
    T0=$(date +%s)
    python -m llm_infer.agent.eval_agent \
        --input         "${PREDICT_JSONL}" \
        --output        "${EXP2_AGENT_HARD}" \
        --adapter       "${ADAPTER_DIR}" \
        --model         "${MODEL}" \
        --max_steps     "${MAX_STEPS}" \
        --default_k     "${DEFAULT_K}" \
        --output_format v2 \
        --hard_subset   "${HARD_SUBSET}" \
        --device        "${DEVICE}"
    echo "[INFO] 2 done, elapsed $(($(date +%s) - T0))s"
fi

# ===========================================================================
# Experiment 3: Ablation study
# ===========================================================================
section "Experiment 3: Ablation study"

# ---------------------------------------------------------------------------
# TODO: No-verify ablation
# ---------------------------------------------------------------------------
# A "no-verify" ablation (agent without citation verification) would require
# adding a --no_verify flag to llm_infer/agent/eval_agent.py and wiring it
# through to the react_loop so that verify_citation is skipped.
# This is intentionally omitted for now to keep Step 5 self-contained.
# When --no_verify is implemented, add the following run here:
#
#   python -m llm_infer.agent.eval_agent \
#       --input "$PREDICT_JSONL" --output "$OUTPUT_DIR/exp3_no_verify.jsonl" \
#       --adapter "$ADAPTER_DIR" --model "$MODEL" --max_steps "$MAX_STEPS" \
#       --default_k "$DEFAULT_K" --output_format v2 --no_verify --device "$DEVICE"
# ---------------------------------------------------------------------------

# 3a. max_steps sweep
echo "[RUN] 3a. max_steps ablation ..."
for steps in 1 3 5 8; do
    OUT="${OUTPUT_DIR}/exp3_steps${steps}.jsonl"
    if [[ -f "${OUT}" ]]; then
        echo "  [SKIP] max_steps=${steps} results exist: ${OUT}"
        continue
    fi
    echo "  [RUN] max_steps=${steps} ..."
    T0=$(date +%s)
    python -m llm_infer.agent.eval_agent \
        --input         "${PREDICT_JSONL}" \
        --output        "${OUT}" \
        --adapter       "${ADAPTER_DIR}" \
        --model         "${MODEL}" \
        --max_steps     "${steps}" \
        --default_k     "${DEFAULT_K}" \
        --output_format v2 \
        --device        "${DEVICE}"
    echo "  [INFO] max_steps=${steps} done, elapsed $(($(date +%s) - T0))s"
done

# 3b. default_k sweep (input path count)
echo "[RUN] 3b. default_k ablation ..."
for k in 5 10 15 20; do
    OUT="${OUTPUT_DIR}/exp3_k${k}.jsonl"
    if [[ -f "${OUT}" ]]; then
        echo "  [SKIP] default_k=${k} results exist: ${OUT}"
        continue
    fi
    echo "  [RUN] default_k=${k} ..."
    T0=$(date +%s)
    python -m llm_infer.agent.eval_agent \
        --input         "${PREDICT_JSONL}" \
        --output        "${OUT}" \
        --adapter       "${ADAPTER_DIR}" \
        --model         "${MODEL}" \
        --max_steps     "${MAX_STEPS}" \
        --default_k     "${k}" \
        --output_format v2 \
        --device        "${DEVICE}"
    echo "  [INFO] default_k=${k} done, elapsed $(($(date +%s) - T0))s"
done

# ===========================================================================
# Experiment 4: Efficiency analysis
# Print step distribution and tool call counts from Experiment 1 agent results
# ===========================================================================
section "Experiment 4: Efficiency analysis"

EXP1_AGENT_SUMMARY="${EXP1_AGENT}.summary.json"
if [[ ! -f "${EXP1_AGENT_SUMMARY}" ]]; then
    echo "[WARN] Agent summary not found: ${EXP1_AGENT_SUMMARY}"
    echo "       Ensure Experiment 1 completed successfully."
else
    python - "${EXP1_AGENT_SUMMARY}" <<'PYEOF'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    summary = json.load(f)

agent_metrics = summary.get("agent_metrics", {})
standard_metrics = summary.get("standard_metrics", {})

print("=== Experiment 4: Efficiency Analysis ===")
print()
print(f"  Samples evaluated : {standard_metrics.get('n', 'N/A')}")
print(f"  Avg steps / sample: {agent_metrics.get('avg_steps', 'N/A')}")
print()
print("  Step distribution:")
for step, count in sorted(
    agent_metrics.get("steps_distribution", {}).items(), key=lambda x: int(x[0])
):
    print(f"    {step} steps: {count}")

print()
print("  Tool call counts:")
for tool, count in sorted(agent_metrics.get("tool_call_counts", {}).items()):
    print(f"    {tool}: {count}")

print()
print(f"  Terminated by 'finish'   : {agent_metrics.get('terminated_by_finish', 0)}")
print(f"  Terminated by 'max_steps': {agent_metrics.get('terminated_by_max_steps', 0)}")
print()
print(f"  Hit@1             : {standard_metrics.get('hit1', 'N/A'):.4f}")
print(f"  Macro F1          : {standard_metrics.get('macro_f1', 'N/A'):.4f}")
print(f"  Citation Accuracy : {standard_metrics.get('citation_accuracy', 'N/A'):.4f}")
print(f"  Hallucination Rate: {standard_metrics.get('hallucination_rate', 'N/A'):.4f}")
PYEOF
fi

# ===========================================================================
# Done
# ===========================================================================
WALL_END=$(date +%s)
section "All experiments complete"
echo "  Total elapsed : $((WALL_END - WALL_START))s"
echo "  Output dir    : ${OUTPUT_DIR}"
echo
echo "  Key output files:"
echo "    ${OUTPUT_DIR}/exp1_agent.jsonl                 (Exp 1 — main agent result)"
echo "    ${OUTPUT_DIR}/exp1_agent.jsonl.summary.json    (Exp 1 — agent summary)"
echo "    ${OUTPUT_DIR}/exp1_v0_zeroshot/                (Exp 1 — V0 zero-shot baseline)"
echo "    ${OUTPUT_DIR}/hard_subset.txt                  (Exp 2 — hard question list)"
echo "    ${OUTPUT_DIR}/exp2_agent_hard.jsonl            (Exp 2 — hard subset recovery)"
echo "    ${OUTPUT_DIR}/exp3_steps{1,3,5,8}.jsonl        (Exp 3 — max_steps ablation)"
echo "    ${OUTPUT_DIR}/exp3_k{5,10,15,20}.jsonl         (Exp 3 — default_k ablation)"
echo
echo "Done."
