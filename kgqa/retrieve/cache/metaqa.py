"""MetaQA 得分缓存加载：dump_scores 的 dict 缓存 → ScoreBundle（含 hop）。"""
from __future__ import annotations

from kgqa.retrieve.cache.base import ScoreBundle, ScoreLoader, score_bundle_from_cache


class MetaQAScoreLoader(ScoreLoader):
    def load(self, cache_path: str) -> ScoreBundle:
        return score_bundle_from_cache(cache_path, "MetaQA")
