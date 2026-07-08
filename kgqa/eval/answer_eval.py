"""答案级评测：复用 oh_my_agent 指标，增加 group_by 分组视图。"""
from __future__ import annotations

from oh_my_agent.common.metrics import compute_answer_metrics, aggregate_metrics
from kgqa.types import MetricSpec


def answer_record(pred: list[str], gold: list[str], hop=None, format_ok: bool = True) -> dict:
    rec = dict(compute_answer_metrics(pred, gold))
    rec["hop"] = hop
    rec["format_ok"] = format_ok
    # aggregate_metrics 需要 citation/hallucination 字段（compute_answer_metrics 不产出）。
    # 答案级评测无引用数据，补中性默认值，避免聚合时 KeyError。
    rec.setdefault("citation_accuracy", 0.0)
    rec.setdefault("citation_recall", 0.0)
    rec.setdefault("hallucination_rate", 0.0)
    return rec


def answer_summary(records: list[dict], spec: MetricSpec) -> dict:
    overall = aggregate_metrics(records)
    by_hop: dict[str, dict] = {}
    if spec.group_by == "hop":
        groups: dict[str, list[dict]] = {}
        for rec in records:
            if rec.get("hop") is None:
                continue
            groups.setdefault(str(rec["hop"]), []).append(rec)
        by_hop = {hop: aggregate_metrics(recs) for hop, recs in sorted(groups.items())}
    return {"overall": overall, "by_hop": by_hop}
