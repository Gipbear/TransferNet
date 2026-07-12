"""统一得分缓存的数据结构、校验与反序列化。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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


@dataclass
class ScoreBundle:
    meta: CacheMeta
    samples: list[SampleScore]


class ScoreLoader(ABC):
    @abstractmethod
    def load(self, cache_path: str) -> ScoreBundle: ...


def load_score_cache(path: str) -> dict:
    """加载并校验统一的 PyTorch 得分缓存。"""
    cache = torch.load(path, map_location="cpu", weights_only=False)
    version = cache.get("version", 0)
    if version < 1:
        raise ValueError(f"不支持的缓存版本: {version}，请重新生成得分缓存。")
    return cache


def score_bundle_from_cache(path: str, default_dataset: str) -> ScoreBundle:
    """将统一缓存格式还原为 ScoreBundle。"""
    cache = load_score_cache(path)
    meta_dict = cache["meta"]
    raw_samples = cache["samples"]
    meta = CacheMeta(
        dataset=meta_dict.get("dataset", default_dataset),
        split=meta_dict.get("split", ""),
        id2ent=meta_dict.get("id2ent", {}),
        id2rel=meta_dict.get("id2rel", {}),
        num_samples=meta_dict.get("num_samples", len(raw_samples)),
        topk_entities=meta_dict.get("topk_entities", 500),
        input_dir=meta_dict.get("input_dir"),
        qa_file=meta_dict.get("qa_file"),
    )
    samples = [
        SampleScore(
            question=sample["question"],
            topic_ids=list(sample["topic_ids"]),
            gold_ids=list(sample["gold_ids"]),
            hop_attn=sample["hop_attn"],
            rel_probs=sample["rel_probs"],
            ent_indices=sample["ent_indices"],
            ent_scores=sample["ent_scores"],
            e_score_indices=sample["e_score_indices"],
            e_score_values=sample["e_score_values"],
            sample_index=index,
            hop=sample.get("hop"),
            triples=sample.get("triples"),
        )
        for index, sample in enumerate(raw_samples)
    ]
    return ScoreBundle(meta=meta, samples=samples)
