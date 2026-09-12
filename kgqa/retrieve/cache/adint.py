"""ADInt 得分缓存加载器。"""

from kgqa.retrieve.cache.base import ScoreBundle, ScoreLoader, score_bundle_from_cache


class ADIntScoreLoader(ScoreLoader):
    def load(self, cache_path: str) -> ScoreBundle:
        return score_bundle_from_cache(cache_path, "ADInt")
