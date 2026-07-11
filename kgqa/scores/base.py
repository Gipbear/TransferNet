"""得分 dump/load 策略口（方案 C 发散点之二）+ 统一 SampleScore/CacheMeta/ScoreBundle。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class SampleScore:
    question: str
    topic_ids: list[int]
    gold_ids: list[int]
    hop_attn: Any
    rel_probs: list[Any]
    ent_indices: list[Any]
    ent_scores: list[Any]
    e_score_indices: Any
    e_score_values: Any
    sample_index: int = -1
    hop: Optional[int] = None
    triples: Optional[list[list[int]]] = None


@dataclass
class CacheMeta:
    dataset: str
    split: str
    id2ent: dict
    id2rel: dict
    num_samples: int
    topk_entities: int = 500
    input_dir: Optional[str] = None
    qa_file: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreBundle:
    meta: CacheMeta
    samples: list[SampleScore]


class ScoreLoader(ABC):
    @abstractmethod
    def load(self, cache_path: str) -> ScoreBundle: ...


class ScoreDumper(ABC):
    @abstractmethod
    def dump(self, bundle: ScoreBundle, out_path: str) -> None: ...


def load_score_cache(path: str) -> dict:
    """加载并校验统一的 PyTorch 得分缓存。"""
    cache = torch.load(path, map_location="cpu", weights_only=False)
    version = cache.get("version", 0)
    if version < 1:
        raise ValueError(f"不支持的缓存版本: {version}，请重新生成得分缓存。")
    return cache
