"""kgqa 测试合成夹具。"""
from __future__ import annotations

import torch
from kgqa.scores.base import SampleScore


def toy_sample_score() -> SampleScore:
    return SampleScore(
        question="toy question",
        topic_ids=[0],
        gold_ids=[1],
        hop_attn=torch.tensor([0.9, 0.1]),
        rel_probs=[torch.tensor([0.0, 0.8, 0.7]), torch.tensor([0.0, 0.0, 0.0])],
        ent_indices=[torch.tensor([1, 2]), torch.tensor([], dtype=torch.long)],
        ent_scores=[torch.tensor([0.6, 0.5]), torch.tensor([])],
        e_score_indices=torch.tensor([1, 2]),
        e_score_values=torch.tensor([0.95, 0.4]),
        sample_index=0,
    )
