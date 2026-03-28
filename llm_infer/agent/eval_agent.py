"""Evaluation pipeline for the KG-ReAct Agent on WebQSP test data.

Evaluates the KGReActAgent end-to-end and computes:
  - Standard answer metrics (Hit@1, F1, EM) via eval_faithfulness helpers
  - Faithfulness metrics (Citation Accuracy, Recall, Hallucination Rate)
  - Agent-specific metrics (steps distribution, termination mode, tool call counts)
  - Optional hard-subset recovery analysis

Usage:
  python llm_infer/agent/eval_agent.py \\
      --input  data/output/WebQSP/predict_test.jsonl \\
      --output data/output/WebQSP/eval_agent.jsonl \\
      --adapter models/webqsp_v2 \\
      --max_steps 8

  # Debug on first 50 samples, no adapter
  python llm_infer/agent/eval_agent.py \\
      --input  data/output/WebQSP/predict_test.jsonl \\
      --output /tmp/eval_agent_debug.jsonl \\
      --limit 50

  # Hard-subset recovery analysis
  python llm_infer/agent/eval_agent.py \\
      --input  data/output/WebQSP/predict_test.jsonl \\
      --output data/output/WebQSP/eval_agent.jsonl \\
      --adapter models/webqsp_v2 \\
      --hard_subset data/output/WebQSP/ch4_failed_questions.txt
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

# Must be set before transformers/unsloth imports
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("UNSLOTH_DISABLE_STATS", "1")

import torch

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Ensure llm_infer/ is importable (for eval_faithfulness, kg_format, etc.)
_LLMINFER_DIR = os.path.join(os.path.dirname(__file__), "..")
if _LLMINFER_DIR not in sys.path:
    sys.path.insert(0, _LLMINFER_DIR)

from eval_faithfulness import (
    aggregate,
    compute_answer_metrics,
    compute_faithfulness,
    get_all_path_entities,
    label_golden_indices,
    parse_output,
)

from .agent_config import AgentConfig
from .react_loop import KGReActAgent, AgentResult
from .tools import AgentContext, build_default_registry


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(args) -> logging.Logger:
    """Set up stdout + file logger, mirroring eval_faithfulness.setup_logger."""
    log_path = os.path.splitext(os.path.abspath(args.output))[0] + ".log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("eval_agent")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KG-ReAct Agent evaluation on WebQSP test JSONL")
    p.add_argument("--input", required=True,
                   help="Path to test JSONL (with mmr_reason_paths and golden fields)")
    p.add_argument("--output", required=True,
                   help="Path to output JSONL (per-sample results)")
    p.add_argument("--adapter", default=None,
                   help="Path to LoRA adapter directory (optional; for SFT model)")
    p.add_argument("--model", default="unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
                   help="Base model name/path (default: Llama-3.1-8B-Instruct 4bit)")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Override AgentConfig.max_steps")
    p.add_argument("--default_k", type=int, default=None,
                   help="Override AgentConfig.default_k")
    p.add_argument("--output_format", default="v2",
                   choices=["v0", "v1", "v2", "v3", "v4", "v5", "v11"],
                   help="SFT output format version (default: v2)")
    p.add_argument("--limit", type=int, default=0,
                   help="Only evaluate first N samples (0 = all; for debugging)")
    p.add_argument("--hard_subset", default=None,
                   help="Path to file listing question strings (one per line) that "
                        "failed in Ch4-SFT; enables recovery_rate reporting")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size for LLM inference (agent runs sequentially; default: 1)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--device", default="cuda",
                   help="Device for model inference (default: cuda)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_llm(args, log: logging.Logger):
    """Load the quantized base model and optional LoRA adapter.

    Mirrors the pattern used in eval_faithfulness.main().

    Returns
    -------
    tuple[model, tokenizer]
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        log.error("unsloth is not installed. Run: pip install unsloth")
        sys.exit(1)

    log.info("Loading model: %s", args.model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        local_files_only=True,
    )

    if args.adapter:
        log.info("Loading LoRA adapter: %s", args.adapter)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)

    FastLanguageModel.for_inference(model)
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Model ready on device: %s", next(model.parameters()).device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    """Load a JSONL file and return a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_hard_subset(path: str) -> set:
    """Load a newline-delimited list of question strings."""
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


# ---------------------------------------------------------------------------
# Zero-metric record (for exception handling)
# ---------------------------------------------------------------------------

_ZERO_ANSWER_METRICS = {
    "hit1": 0, "hit_any": 0,
    "precision": 0.0, "recall": 0.0, "f1": 0.0,
    "exact_match": False,
    "tp": 0, "pred_n": 0, "gold_n": 0,
}

_ZERO_FAITH_METRICS = {
    "citation_accuracy": 0.0,
    "citation_recall": 0.0,
    "hallucination_rate": 0.0,
    "hallucinated_entities": [],
}


def _failure_record(record: dict, error_msg: str) -> dict:
    """Build a zeroed-metric result record for a failed sample."""
    return {
        "question": record.get("question", ""),
        "golden": record.get("golden", []),
        "pred_answer": [],
        "cited_paths": [],
        "steps_taken": 0,
        "terminated_by": "error",
        "reasoning_trace": [],
        "format_ok": False,
        "error": error_msg,
        **_ZERO_ANSWER_METRICS,
        **_ZERO_FAITH_METRICS,
    }


# ---------------------------------------------------------------------------
# Agent-specific metrics
# ---------------------------------------------------------------------------

def aggregate_agent_metrics(results: list, hard_subset: set | None = None) -> dict:
    """Compute agent-specific aggregate metrics.

    Parameters
    ----------
    results:
        List of per-sample result dicts (as written to the output JSONL).
    hard_subset:
        Optional set of question strings that failed in Ch4-SFT.

    Returns
    -------
    dict
        Agent-specific metrics dict.
    """
    n = len(results)
    if n == 0:
        return {}

    # Average steps
    steps_list = [r.get("steps_taken", 0) for r in results]
    avg_steps = sum(steps_list) / n

    # Steps distribution
    steps_dist: dict[int, int] = defaultdict(int)
    for s in steps_list:
        steps_dist[s] += 1

    # Termination modes
    terminated_by_finish = sum(
        1 for r in results if r.get("terminated_by") == "finish"
    )
    terminated_by_max_steps = sum(
        1 for r in results if r.get("terminated_by") == "max_steps"
    )

    # Tool call counts from reasoning traces
    tool_call_counts: dict[str, int] = defaultdict(int)
    for r in results:
        for step in r.get("reasoning_trace", []):
            action = step.get("action") or ""
            if action:
                # action format: "tool_name(...)" or None
                tool_name = action.split("(")[0].strip()
                if tool_name:
                    tool_call_counts[tool_name] += 1

    out: dict = {
        "avg_steps": round(avg_steps, 4),
        "steps_distribution": dict(sorted(steps_dist.items())),
        "terminated_by_finish": terminated_by_finish,
        "terminated_by_max_steps": terminated_by_max_steps,
        "tool_call_counts": dict(tool_call_counts),
    }

    # Hard-subset recovery analysis
    if hard_subset is not None:
        hard_results = [r for r in results if r.get("question", "") in hard_subset]
        hard_n = len(hard_results)
        if hard_n > 0:
            recovery_rate = sum(
                1 for r in hard_results if r.get("hit1", 0) > 0
            ) / hard_n
            hard_f1 = sum(r.get("f1", 0.0) for r in hard_results) / hard_n
        else:
            recovery_rate = 0.0
            hard_f1 = 0.0

        out["recovery_rate"] = round(recovery_rate, 4)
        out["hard_subset_f1"] = round(hard_f1, 4)
        out["hard_subset_n"] = hard_n

    return out


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_results(
    results: list,
    agg: dict,
    agent_agg: dict,
    args: argparse.Namespace,
    log: logging.Logger,
) -> None:
    """Write per-sample JSONL and summary JSON to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # 1. Per-sample JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Per-sample results written to: %s", args.output)

    # 2. Summary JSON
    summary_path = args.output + ".summary.json"
    summary = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": args.input,
            "output": args.output,
            "model": args.model,
            "adapter": args.adapter,
            "output_format": args.output_format,
            "max_steps": args.max_steps,
            "default_k": args.default_k,
            "limit": args.limit,
            "n_evaluated": len(results),
        },
        "standard_metrics": agg,
        "agent_metrics": agent_agg,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("Summary written to: %s", summary_path)


def print_summary(agg: dict, agent_agg: dict, log: logging.Logger) -> None:
    """Print aggregated metrics to the log."""
    log.info("=" * 60)
    log.info("  [ KG-ReAct Agent Evaluation Results ]  (n=%d)", agg.get("n", 0))
    log.info("")
    log.info("  --- Answer Accuracy ---")
    log.info("    Hit@1             : %.4f", agg.get("hit1", 0))
    log.info("    Hit@Any           : %.4f", agg.get("hit_any", 0))
    log.info("    Macro  P/R/F1     : %.4f / %.4f / %.4f",
             agg.get("macro_p", 0), agg.get("macro_r", 0), agg.get("macro_f1", 0))
    log.info("    Micro  P/R/F1     : %.4f / %.4f / %.4f",
             agg.get("micro_p", 0), agg.get("micro_r", 0), agg.get("micro_f1", 0))
    log.info("    Exact Match       : %.4f", agg.get("exact_match", 0))
    log.info("")
    log.info("  --- Faithfulness ---")
    log.info("    Citation Accuracy : %.4f", agg.get("citation_accuracy", 0))
    log.info("    Citation Recall   : %.4f", agg.get("citation_recall", 0))
    log.info("    Hallucination Rate: %.4f", agg.get("hallucination_rate", 0))
    log.info("    Format Compliance : %.4f", agg.get("format_compliance", 0))
    log.info("")
    log.info("  --- Agent Behavior ---")
    log.info("    Avg Steps         : %.4f", agent_agg.get("avg_steps", 0))
    log.info("    Steps Distribution: %s", agent_agg.get("steps_distribution", {}))
    log.info("    Terminated finish : %d", agent_agg.get("terminated_by_finish", 0))
    log.info("    Terminated max_stp: %d", agent_agg.get("terminated_by_max_steps", 0))
    log.info("    Tool Call Counts  : %s", agent_agg.get("tool_call_counts", {}))
    if "recovery_rate" in agent_agg:
        log.info("")
        log.info("  --- Hard-Subset Recovery (n=%d) ---", agent_agg.get("hard_subset_n", 0))
        log.info("    Recovery Rate     : %.4f", agent_agg.get("recovery_rate", 0))
        log.info("    Hard Subset F1    : %.4f", agent_agg.get("hard_subset_f1", 0))
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# ETA helper
# ---------------------------------------------------------------------------

def _eta_str(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "ETA: ?"
    rate = done / elapsed  # samples per second
    remaining = (total - done) / rate
    return f"ETA: {timedelta(seconds=int(remaining))}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Seed
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    log = setup_logger(args)

    log.info("=" * 60)
    log.info("eval_agent starting")
    log.info("  command       : %s", " ".join(sys.argv))
    log.info("  input         : %s", args.input)
    log.info("  output        : %s", args.output)
    log.info("  model         : %s", args.model)
    log.info("  adapter       : %s", args.adapter or "None (zero-shot)")
    log.info("  output_format : %s", args.output_format)
    log.info("  max_steps     : %s", args.max_steps or "default")
    log.info("  default_k     : %s", args.default_k or "default")
    log.info("  limit         : %s", args.limit if args.limit > 0 else "all")
    log.info("  hard_subset   : %s", args.hard_subset or "None")
    log.info("  device        : %s", args.device)
    log.info("  seed          : %d", args.seed)
    log.info("=" * 60)

    # Load model
    model, tokenizer = load_llm(args, log)

    # Build AgentConfig
    cfg_kwargs: dict = {"device": args.device}
    if args.max_steps is not None:
        cfg_kwargs["max_steps"] = args.max_steps
    if args.default_k is not None:
        cfg_kwargs["default_k"] = args.default_k
    cfg = AgentConfig(**cfg_kwargs)

    # Build agent
    ctx = AgentContext()
    registry = build_default_registry(
        model=model,
        tokenizer=tokenizer,
        context=ctx,
        config=cfg,
        output_format=args.output_format,
    )
    agent = KGReActAgent(
        model=model,
        tokenizer=tokenizer,
        tool_registry=registry,
        config=cfg,
    )

    # Load test data
    log.info("Loading test data from: %s", args.input)
    records = load_jsonl(args.input)
    if args.limit > 0:
        records = records[: args.limit]
    log.info("Samples to evaluate: %d", len(records))

    # Load hard subset (optional)
    hard_subset: set | None = None
    if args.hard_subset:
        hard_subset = load_hard_subset(args.hard_subset)
        log.info("Hard subset loaded: %d questions", len(hard_subset))

    # Evaluation loop
    results: list[dict] = []
    start_time = time.time()

    for i, record in enumerate(records):
        question = record.get("question", "")
        golden = record.get("golden", [])
        mmr_paths = record.get("mmr_reason_paths", [])

        # Progress log every 50 samples
        if i > 0 and i % 50 == 0:
            elapsed = time.time() - start_time
            eta = _eta_str(elapsed, i, len(records))
            log.info(
                "Progress: %d/%d (%.1f%%)  elapsed=%.1fs  %s",
                i, len(records), 100.0 * i / len(records), elapsed, eta,
            )

        try:
            # Set shared context for this sample
            ctx.question = question
            ctx.current_record = record
            ctx.current_paths = []
            ctx.last_reasoning = {}

            # Run agent
            result: AgentResult = agent.run(question)

            # Defensive check: verify reasoning_trace length matches steps_taken
            if len(result.reasoning_trace) != result.steps_taken:
                log.debug("reasoning_trace length %d != steps_taken %d for question: %s",
                         len(result.reasoning_trace), result.steps_taken, question[:80])

            # Compute metrics
            golden_indices = label_golden_indices(mmr_paths, golden)
            path_entities = get_all_path_entities(mmr_paths)

            answer_metrics = compute_answer_metrics(result.answer, golden)
            faith_metrics = compute_faithfulness(
                cited_indices=set(result.cited_paths),
                golden_indices=golden_indices,
                pred_answers=result.answer,
                path_entities=path_entities,
            )

            # Determine format_ok: use parse_output to validate format consistently with eval_faithfulness
            format_ok = False
            raw_output = None
            for step in result.reasoning_trace:
                action = step.get("action") or ""
                if action.startswith("reason_and_cite"):
                    raw_output = step.get("observation", "")

            if raw_output:
                # Get the final answer string from the last reason_and_cite observation
                parsed = parse_output(raw_output, args.output_format)
                format_ok = parsed.get("format_ok", False)
            else:
                # No reason_and_cite observation found; agent jumped to finish without reasoning
                format_ok = False

            rec = {
                "question": question,
                "golden": golden,
                "pred_answer": result.answer,
                "cited_paths": result.cited_paths,
                "steps_taken": result.steps_taken,
                "terminated_by": result.terminated_by,
                "reasoning_trace": result.reasoning_trace,
                "final_verification": result.final_verification,
                "format_ok": format_ok,
                # Answer metrics
                "hit1": answer_metrics["hit1"],
                "hit_any": answer_metrics["hit_any"],
                "precision": round(answer_metrics["precision"], 4),
                "recall": round(answer_metrics["recall"], 4),
                "f1": round(answer_metrics["f1"], 4),
                "exact_match": answer_metrics["exact_match"],
                "tp": answer_metrics["tp"],
                "pred_n": answer_metrics["pred_n"],
                "gold_n": answer_metrics["gold_n"],
                # Faithfulness metrics
                "citation_accuracy": faith_metrics["citation_accuracy"],
                "citation_recall": faith_metrics["citation_recall"],
                "hallucination_rate": faith_metrics["hallucination_rate"],
                "hallucinated_entities": faith_metrics["hallucinated_entities"],
                # Passthrough metadata
                "hop": record.get("hop"),
                "mmr_answer_path_hit": record.get("mmr_answer_path_hit"),
            }
            results.append(rec)

            log.debug(
                "Sample %d/%d  hit1=%d  f1=%.4f  steps=%d  term=%s  q=%s",
                i + 1, len(records),
                answer_metrics["hit1"],
                answer_metrics["f1"],
                result.steps_taken,
                result.terminated_by,
                question[:60],
            )

        except Exception as exc:  # noqa: BLE001
            log.error(
                "Exception on sample %d/%d (q=%r): %s",
                i + 1, len(records), question[:60], exc,
                exc_info=True,
            )
            results.append(_failure_record(record, str(exc)))

    elapsed_total = time.time() - start_time
    log.info(
        "Evaluation complete: %d/%d samples in %.1fs (%.2f s/sample)",
        len(results), len(records), elapsed_total,
        elapsed_total / max(len(results), 1),
    )
    log.info("finish_time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Aggregate
    agg = aggregate(results)
    agent_agg = aggregate_agent_metrics(results, hard_subset=hard_subset)

    # Print summary
    print_summary(agg, agent_agg, log)

    # Save outputs
    save_results(results, agg, agent_agg, args, log)


if __name__ == "__main__":
    main()
