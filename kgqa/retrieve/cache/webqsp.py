"""WebQSP 得分缓存加载：把 dump_scores.py 的 dict 缓存转成 ScoreBundle。"""
from __future__ import annotations

from kgqa.retrieve.cache.base import ScoreBundle, ScoreLoader, score_bundle_from_cache


class WebQSPScoreLoader(ScoreLoader):
    def load(self, cache_path: str) -> ScoreBundle:
        return score_bundle_from_cache(cache_path, "WebQSP")
