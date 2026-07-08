"""WebQSP 得分缓存加载：把 dump_scores.py 的 dict 缓存转成 ScoreBundle。"""
from __future__ import annotations

from scripts.offline_path_search import load_score_cache
from kgqa.scores.base import CacheMeta, SampleScore, ScoreBundle, ScoreLoader


class WebQSPScoreLoader(ScoreLoader):
    def load(self, cache_path: str) -> ScoreBundle:
        cache = load_score_cache(cache_path)
        meta_d = cache["meta"]
        meta = CacheMeta(
            dataset=meta_d.get("dataset", "WebQSP"),
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
                triples=s.get("triples"),
            )
            for i, s in enumerate(cache["samples"])
        ]
        return ScoreBundle(meta=meta, samples=samples)
