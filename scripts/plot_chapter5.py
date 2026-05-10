"""Generate Chapter-5 system behavior figures."""

from __future__ import annotations

import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

EVAL_DIR = "data/output/WebQSP/checked_batch_agent/checked_batch_eval_20260505_0009"
OUTPUT_DIR = "docs/figures/chapter5"


def _load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def plot_batches_distribution(records: list[dict], output_path: str) -> None:
    counter = Counter(int(record.get("batches_used", 0)) for record in records)
    bins = sorted(counter)
    counts = [counter[b] for b in bins]
    total = sum(counts)
    pct = [c / total * 100 for c in counts]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(b) for b in bins], counts, color="#4C72B0")
    for bar, percent in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{percent:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Batches Used")
    ax.set_ylabel("Sample Count")
    ax.set_title("Batches Consumed Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_stop_reason(summary: dict, output_path: str) -> None:
    counts = summary.get("stop_reason_counts", {})
    labels = list(counts.keys())
    sizes = list(counts.values())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90,
           colors=["#55A868", "#C44E52", "#8172B2"][: len(labels)])
    ax.set_title("Stop Reason Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_latency(summary: dict, output_path: str) -> None:
    stages = ["Retrieval", "Answer", "Check"]
    values = [
        summary.get("avg_retrieval_elapsed_ms", 0.0),
        summary.get("avg_answer_elapsed_ms", 0.0),
        summary.get("avg_check_elapsed_ms", 0.0),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(stages, values, color=["#4C72B0", "#DD8452", "#55A868"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{value:.0f} ms", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title("Per-Stage Average Latency Breakdown")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = _load_jsonl(os.path.join(EVAL_DIR, "checked_batch_eval.jsonl"))
    with open(os.path.join(EVAL_DIR, "checked_batch_eval_summary.json"), "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    plot_batches_distribution(records, os.path.join(OUTPUT_DIR, "batches_distribution.png"))
    plot_stop_reason(summary, os.path.join(OUTPUT_DIR, "stop_reason.png"))
    plot_latency(summary, os.path.join(OUTPUT_DIR, "latency_breakdown.png"))
    print("Saved figures to", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
