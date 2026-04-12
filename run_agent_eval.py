#!/usr/bin/env python3
# run_agent_eval.py
"""
PathfinderAgent evaluation entry point.

Processes a predict JSONL file through the PathfinderAgent pipeline and computes
the same Hit@1, Hits, F1 metrics as llm_infer / eval_faithfulness.

Usage:
  # Evaluate on groupAname_v2 ablation errors (path_hit_but_wrong subset)
  python run_agent_eval.py \\
      --input  data/output/WebQSP/ablation/groupAname_v2/beam20_lam0.2_v2_ft_eval_run0.jsonl \\
      --output data/output/WebQSP/pathfinder_logs/agent_eval_run0.jsonl \\
      --ckpt   data/ckpt/WebQSP/model-29-0.6411.pt \\
      --input_dir data/input/WebQSP

  # Limit to first 50 samples for a quick smoke test
  python run_agent_eval.py \\
      --input  data/output/WebQSP/ablation/groupAname_v2/beam20_lam0.2_v2_ft_eval_run0.jsonl \\
      --output data/output/WebQSP/pathfinder_logs/agent_eval_debug.jsonl \\
      --ckpt   data/ckpt/WebQSP/model-29-0.6411.pt \\
      --input_dir data/input/WebQSP \\
      --limit 50
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

# Environment setup (must be before unsloth / transformers)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("UNSLOTH_DISABLE_STATS", "1")

# Ensure workspace root is importable
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pathfinder_agent.agent import PathfinderAgent
from pathfinder_agent.tools.dynamic_retriever import TransferNetWrapper
from pathfinder_agent.config import LOG_DIR, LORA_ADAPTER_PATH


# ---------------------------------------------------------------------------
# Metrics (reuse from eval_faithfulness)
# ---------------------------------------------------------------------------

def _norm(s):
    return s.lower().strip()


def compute_metrics(pred: list, gold: list) -> dict:
    pred_set = {_norm(e) for e in pred if e.strip()}
    gold_set = {_norm(e) for e in gold if e.strip()}

    hit1    = int(bool(pred) and _norm(pred[0]) in gold_set)
    hit_any = int(bool(pred_set & gold_set))

    if not pred_set and not gold_set:
        return {"hit1": 1, "hit_any": 1, "f1": 1.0, "precision": 1.0, "recall": 1.0}
    if not pred_set or not gold_set:
        return {"hit1": hit1, "hit_any": hit_any, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    tp = len(pred_set & gold_set)
    p  = tp / len(pred_set)
    r  = tp / len(gold_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"hit1": hit1, "hit_any": hit_any, "f1": round(f1, 4),
            "precision": round(p, 4), "recall": round(r, 4)}


def aggregate(results: list) -> dict:
    n = len(results)
    if n == 0:
        return {}
    def mean(k):
        return round(sum(r.get(k, 0) for r in results) / n, 4)
    return {
        "n":        n,
        "hit1":     mean("hit1"),
        "hit_any":  mean("hit_any"),
        "macro_f1": mean("f1"),
        "macro_p":  mean("precision"),
        "macro_r":  mean("recall"),
    }


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(output_path: str) -> logging.Logger:
    log_path = os.path.splitext(os.path.abspath(output_path))[0] + ".log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("run_agent_eval")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
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

def parse_args():
    p = argparse.ArgumentParser(description="PathfinderAgent Evaluation")
    p.add_argument("--input",     required=True,  help="Input JSONL (predict or ablation)")
    p.add_argument("--output",    required=True,  help="Output JSONL (per-sample results)")
    p.add_argument("--ckpt",      required=True,  help="TransferNet checkpoint (.pt)")
    p.add_argument("--input_dir", required=True,  help="TransferNet data dir (WebQSP input_dir)")
    p.add_argument("--model",     default="unsloth/meta-llama-3.1-8b-instruct-bnb-4bit")
    p.add_argument("--adapter",   default=LORA_ADAPTER_PATH, help="LoRA adapter path")
    p.add_argument("--limit",     type=int, default=0, help="Evaluate first N samples (0=all)")
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# ETA helper
# ---------------------------------------------------------------------------

def _eta(elapsed, done, total):
    if done == 0:
        return "ETA: ?"
    return f"ETA: {timedelta(seconds=int((total - done) / (done / elapsed)))}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    log = setup_logger(args.output)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    log.info("=" * 60)
    log.info("PathfinderAgent Evaluation")
    log.info("  input    : %s", args.input)
    log.info("  output   : %s", args.output)
    log.info("  ckpt     : %s", args.ckpt)
    log.info("  adapter  : %s", args.adapter)
    log.info("=" * 60)

    # -- Load data --
    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    if args.limit > 0:
        records = records[:args.limit]
    log.info("Samples to evaluate: %d", len(records))

    # -- Initialize agent (model + adapter loaded inside __init__) --
    agent = PathfinderAgent(
        model_name=args.model,
        adapter_path=args.adapter,
        device=args.device,
    )

    # -- Build TransferNet wrapper --
    log.info("Loading TransferNet from %s ...", args.ckpt)
    transfernet = TransferNetWrapper(data_dir=args.input_dir, ckpt_path=args.ckpt)
    agent.transfernet_wrapper = transfernet
    log.info("TransferNet ready.")

    # -- Evaluation loop --
    results = []
    start_time = time.time()

    for i, record in enumerate(records):
        question     = record.get("question", "")
        golden       = record.get("golden", [])
        topics_field = record.get("topics", [])
        topic_entity = topics_field[0] if topics_field else ""

        if i > 0 and i % 20 == 0:
            elapsed = time.time() - start_time
            log.info("Progress: %d/%d (%.1f%%)  %s",
                     i, len(records), 100.0 * i / len(records),
                     _eta(elapsed, i, len(records)))

        try:
            pred_answers = agent.run(question, topic_entity)
            m = compute_metrics(pred_answers, golden)
            rec = {
                "question":     question,
                "topic_entity": topic_entity,
                "golden":       golden,
                "pred_answer":  pred_answers,
                **m,
            }
        except Exception as e:
            log.error("Error on sample %d (q=%r): %s", i + 1, question[:60], e, exc_info=True)
            rec = {
                "question": question,
                "topic_entity": topic_entity,
                "golden":   golden,
                "pred_answer": [],
                "hit1": 0, "hit_any": 0, "f1": 0.0,
                "precision": 0.0, "recall": 0.0,
                "error": str(e),
            }

        results.append(rec)
        log.debug("Sample %d: hit1=%d f1=%.4f pred=%s",
                  i + 1, rec["hit1"], rec["f1"], rec["pred_answer"])

    elapsed_total = time.time() - start_time
    log.info("Completed %d samples in %.1fs (%.2fs/sample)",
             len(results), elapsed_total, elapsed_total / max(len(results), 1))

    # -- Aggregate --
    agg = aggregate(results)
    log.info("=" * 60)
    log.info("  Hit@1      : %.4f", agg.get("hit1", 0))
    log.info("  Hit@Any    : %.4f", agg.get("hit_any", 0))
    log.info("  Macro F1   : %.4f", agg.get("macro_f1", 0))
    log.info("  Macro P/R  : %.4f / %.4f", agg.get("macro_p", 0), agg.get("macro_r", 0))
    log.info("=" * 60)

    # -- Save per-sample JSONL --
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Per-sample results -> %s", args.output)

    # -- Save summary JSON --
    summary_path = args.output.replace(".jsonl", "_summary.json")
    summary = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input":     args.input,
            "ckpt":      args.ckpt,
            "adapter":   args.adapter,
            "limit":     args.limit,
            "n_evaluated": len(results),
        },
        "metrics": agg,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("Summary -> %s", summary_path)


if __name__ == "__main__":
    main()
