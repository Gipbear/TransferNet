"""MetaQA 得分缓存加载：dump_scores 的 dict 缓存 → ScoreBundle（含 hop）。"""
from __future__ import annotations

import torch

from kgqa.scores.base import CacheMeta, SampleScore, ScoreBundle, ScoreLoader


class MetaQAScoreLoader(ScoreLoader):
    def load(self, cache_path: str) -> ScoreBundle:
        cache = torch.load(cache_path, weights_only=False)
        meta_d = cache["meta"]
        meta = CacheMeta(
            dataset=meta_d.get("dataset", "MetaQA"),
            split=meta_d.get("split", ""),
            id2ent=meta_d.get("id2ent", {}),
            id2rel=meta_d.get("id2rel", {}),
            num_samples=meta_d.get("num_samples", len(cache["samples"])),
            topk_entities=meta_d.get("topk_entities", 500),
            input_dir=meta_d.get("input_dir"),
            qa_file=meta_d.get("qa_file"),
        )
        samples = [
            SampleScore(
                question=s["question"],
                topic_ids=list(s["topic_ids"]),
                gold_ids=list(s["gold_ids"]),
                hop_attn=s["hop_attn"],
                rel_probs=s["rel_probs"],
                ent_indices=s["ent_indices"],
                ent_scores=s["ent_scores"],
                e_score_indices=s["e_score_indices"],
                e_score_values=s["e_score_values"],
                sample_index=i,
                hop=s.get("hop"),
                triples=s.get("triples"),
            )
            for i, s in enumerate(cache["samples"])
        ]
        return ScoreBundle(meta=meta, samples=samples)
