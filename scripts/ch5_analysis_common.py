"""第五章适用性分析（A1–A3）的公共工具：数据加载、指标口径与配对 bootstrap。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROBE_PATH = "data/analysis/20260919_1700__ch5_applicability_probe/probe_results.json"
SEED = 17
N_BOOT = 2000


def load_probe(path: str | Path = PROBE_PATH) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: payload[name]["rows"] for name in ("webqsp", "cwq")}


def micro_f1(rows: list[dict[str, Any]], opened: bool) -> float:
    """全局汇总口径的 Micro-F1；gold 不随补全变化，两侧共用。"""
    tp = sum(row["tp_open" if opened else "tp_closed"] for row in rows)
    pred = sum(row["pred_open" if opened else "closed_n"] for row in rows)
    gold = sum(row["gold_n"] for row in rows)
    if not tp or not pred or not gold:
        return 0.0
    precision, recall = tp / pred, tp / gold
    return 2 * precision * recall / (precision + recall)


def macro_f1(rows: list[dict[str, Any]], opened: bool) -> float:
    key = "f1_open" if opened else "f1_closed"
    return sum(row[key] for row in rows) / len(rows) if rows else 0.0


def _micro_curve(tp: np.ndarray, pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, gold, out=np.zeros_like(tp), where=gold > 0)
    total = precision + recall
    return np.divide(2 * precision * recall, total, out=np.zeros_like(tp), where=total > 0)


def bootstrap_delta(
    rows: list[dict[str, Any]],
    n_boot: int = N_BOOT,
    seed: int = SEED,
    gate: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, float]:
    """逐样本配对重采样，返回相对「关补全」的 ΔMacro-F1 与 ΔMicro-F1 及 95% 区间。

    gate 为 None 时所有样本都取开补全口径（即标准 ΔF1）；
    传入判定函数时，仅由该函数判真的样本取开补全，其余回退到关补全。
    """
    if not rows:
        return {"n": 0, "macro": 0.0, "macro_lo": 0.0, "macro_hi": 0.0,
                "micro": 0.0, "micro_lo": 0.0, "micro_hi": 0.0}
    array = lambda key: np.array([float(row[key]) for row in rows])  # noqa: E731
    use_open = np.ones(len(rows), dtype=bool) if gate is None else np.array(
        [bool(gate(row)) for row in rows])
    f1_open, f1_closed = array("f1_open"), array("f1_closed")
    tp_open, tp_closed, pred_open = array("tp_open"), array("tp_closed"), array("pred_open")
    closed_n, gold = array("closed_n"), array("gold_n")
    chosen_f1 = np.where(use_open, f1_open, f1_closed)
    chosen_tp = np.where(use_open, tp_open, tp_closed)
    chosen_pred = np.where(use_open, pred_open, closed_n)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(rows), size=(n_boot, len(rows)))
    macro_curve = chosen_f1[picks].mean(axis=1) - f1_closed[picks].mean(axis=1)
    micro_curve = (
        _micro_curve(chosen_tp[picks].sum(axis=1), chosen_pred[picks].sum(axis=1), gold[picks].sum(axis=1))
        - _micro_curve(tp_closed[picks].sum(axis=1), closed_n[picks].sum(axis=1), gold[picks].sum(axis=1))
    )
    macro_point = chosen_f1.mean() - f1_closed.mean()
    total = np.array([gold.sum()])
    micro_point = float(
        _micro_curve(np.array([chosen_tp.sum()]), np.array([chosen_pred.sum()]), total)[0]
        - _micro_curve(np.array([tp_closed.sum()]), np.array([closed_n.sum()]), total)[0]
    )
    return {
        "n": len(rows),
        "macro": float(macro_point),
        "macro_lo": float(np.percentile(macro_curve, 2.5)),
        "macro_hi": float(np.percentile(macro_curve, 97.5)),
        "micro": float(micro_point),
        "micro_lo": float(np.percentile(micro_curve, 2.5)),
        "micro_hi": float(np.percentile(micro_curve, 97.5)),
    }


def bucketize(
    rows: list[dict[str, Any]], key: str, edges: list[float]
) -> list[tuple[str, list[dict[str, Any]]]]:
    """按 edges 升序分桶，返回 [(区间标签, 样本)]，空桶保留以维持跨数据集可比。"""
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = float(row.get(key, 0.0))
        buckets[sum(1 for edge in edges if value >= edge)].append(row)
    result = []
    for slot in range(len(edges) + 1):
        low = edges[slot - 1] if slot > 0 else None
        high = edges[slot] if slot < len(edges) else None
        if low is None:
            label = f"<{high:g}"
        elif high is None:
            label = f"≥{low:g}"
        else:
            label = f"[{low:g}, {high:g})"
        result.append((label, buckets[slot]))
    return result
