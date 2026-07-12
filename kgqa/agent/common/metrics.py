"""兼容路径：答案与忠实度指标已迁至 :mod:`kgqa.core.answer_metrics`。"""
from kgqa.core.answer_metrics import (
    aggregate_metrics,
    cited_indices_for_answers,
    compute_answer_metrics,
    compute_faithfulness,
    label_golden_indices,
    llm_produced_answers,
    norm_entity,
)

__all__ = [
    "aggregate_metrics", "cited_indices_for_answers", "compute_answer_metrics",
    "compute_faithfulness", "label_golden_indices", "llm_produced_answers", "norm_entity",
]
