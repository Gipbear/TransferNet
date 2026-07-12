"""数据集得分生产器工厂。"""

from __future__ import annotations

from kgqa.models.base import ScoreProducer


def make_score_producer(
    dataset: str,
    *,
    bert_name: str | None = None,
    per_hop_limit: int = 0,
    limit: int = 0,
) -> ScoreProducer:
    if dataset == "webqsp":
        from kgqa.models.webqsp import WebQSPScoreProducer
        return WebQSPScoreProducer(bert_name=bert_name or "bert-base-uncased")
    if dataset == "metaqa":
        from kgqa.models.metaqa import MetaQAScoreProducer
        return MetaQAScoreProducer(per_hop_limit=per_hop_limit)
    if dataset == "cwq":
        from kgqa.models.cwq import CWQScoreProducer
        return CWQScoreProducer(bert_name=bert_name or "bert-base-cased", limit=limit)
    raise KeyError(f"未支持的 score producer: {dataset}")


__all__ = ["make_score_producer"]
