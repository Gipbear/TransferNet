"""TransferNet 在线得分生产器。"""
from __future__ import annotations

from kgqa.backbone.base import ScoreProducer


def make_score_producer(
    dataset: str,
    *,
    bert_name: str | None = None,
    per_hop_limit: int = 0,
    limit: int = 0,
) -> ScoreProducer:
    if dataset == "webqsp":
        from kgqa.backbone.webqsp import WebQSPScoreProducer
        return WebQSPScoreProducer(bert_name=bert_name or "BAAI/bge-base-en-v1.5")
    if dataset == "metaqa":
        from kgqa.backbone.metaqa import MetaQAScoreProducer
        return MetaQAScoreProducer(per_hop_limit=per_hop_limit)
    if dataset == "cwq":
        from kgqa.backbone.cwq import CWQScoreProducer
        return CWQScoreProducer(bert_name=bert_name or "bert-base-cased", limit=limit)
    raise KeyError(f"未支持的 score producer: {dataset}")


__all__ = ["ScoreProducer", "make_score_producer"]
