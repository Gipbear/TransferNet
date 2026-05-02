"""Cached offline path retrieval service.

This module intentionally keeps only the single-sample retrieval path used by
``scripts.offline_path_search``: reconstruct sparse scores, expand candidates,
apply tail_blend scoring, then select paths with MMR.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from scripts.offline_path_search import (
    LogNormStrategy,
    _method_hop_numbers,
    _path_to_triples,
    candidate_to_tuple,
    final_ent_score_dict,
    load_score_cache,
    rebuild_valid_edges_dict,
    reconstruct_ent_dict,
    reconstruct_rel_dict,
    search_path_candidates,
    select_path_candidates,
)


def normalize_question(question: str) -> str:
    text = question.strip().lower()
    text = re.sub(r"\[(cls|sep|pad)\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class CachedRetrievalResult:
    question: str
    sample_index: int
    topics: list[str]
    hop: int
    mmr_reason_paths: list[dict[str, Any]]
    prediction: dict[str, float]
    elapsed_ms: float
    method: str
    alpha_final: float
    threshold: float
    beam_size: int
    lambda_val: float
    cache_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "sample_index": self.sample_index,
            "topics": self.topics,
            "hop": self.hop,
            "mmr_reason_paths": self.mmr_reason_paths,
            "prediction": self.prediction,
            "elapsed_ms": self.elapsed_ms,
            "method": self.method,
            "alpha_final": self.alpha_final,
            "threshold": self.threshold,
            "beam_size": self.beam_size,
            "lambda_val": self.lambda_val,
            "cache_path": self.cache_path,
        }


class CachedPathRetriever:
    def __init__(self, *, cache_path: str, input_dir: str):
        self.cache_path = str(Path(cache_path))
        self.input_dir = str(Path(input_dir))
        self.cache = load_score_cache(self.cache_path)
        self.samples = self.cache["samples"]
        self.meta = self.cache["meta"]
        self.id2ent = self.meta.get("id2ent", {})
        self.id2rel = self.meta.get("id2rel", {})
        self.valid_edges_dict = rebuild_valid_edges_dict(self.input_dir)
        self.question_index = self._build_question_index()

    def _build_question_index(self) -> dict[str, int]:
        index: dict[str, int] = {}
        for i, sample in enumerate(self.samples):
            key = normalize_question(sample["question"])
            index.setdefault(key, i)
        return index

    def retrieve(
        self,
        *,
        question: Optional[str] = None,
        sample_index: Optional[int] = None,
        topic_entities: Optional[list[str]] = None,
        method: str = "tail_blend",
        alpha_final: float = 1.0,
        threshold: float = 0.01,
        beam_size: int = 50,
        lambda_val: float = 0.5,
    ) -> CachedRetrievalResult:
        if method not in {"tail_blend", "baseline"}:
            raise ValueError(f"unknown method: {method}")
        if beam_size < 1:
            raise ValueError("beam_size must be >= 1")

        t0 = time.perf_counter()
        idx = self._resolve_sample_index(question, sample_index)
        sample = self.samples[idx]
        topics = self._topics(sample)
        if topic_entities is not None and set(topic_entities) != set(topics):
            raise ValueError(f"topic_entities mismatch: expected {topics}, got {topic_entities}")

        hop_num = int(sample["hop_attn"].argmax().item()) + 1
        hop_nums = _method_hop_numbers(method, hop_num, len(sample["rel_probs"]))
        rel_dicts, ent_dicts = self._reconstruct_scores(sample, threshold, max(hop_nums))

        scoring = LogNormStrategy()
        path_candidates = []
        final_scores = final_ent_score_dict(sample)
        for candidate_hop in hop_nums:
            path_candidates.extend(search_path_candidates(
                sample["topic_ids"],
                rel_dicts,
                ent_dicts,
                candidate_hop,
                self.valid_edges_dict,
                scoring,
                beam_size,
                final_ent_scores=final_scores,
                order_start=len(path_candidates),
            ))

        selected = select_path_candidates(
            path_candidates,
            beam_size,
            method=method,
            alpha_final=alpha_final,
            lambda_val=lambda_val,
        )
        paths = [candidate_to_tuple(candidate) for candidate in selected]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return CachedRetrievalResult(
            question=sample["question"],
            sample_index=idx,
            topics=topics,
            hop=hop_num,
            mmr_reason_paths=self._serialize_paths(paths),
            prediction=self._prediction(sample),
            elapsed_ms=round(elapsed_ms, 1),
            method=method,
            alpha_final=alpha_final,
            threshold=threshold,
            beam_size=beam_size,
            lambda_val=lambda_val,
            cache_path=self.cache_path,
        )

    def _resolve_sample_index(self, question: Optional[str], sample_index: Optional[int]) -> int:
        if sample_index is not None:
            if sample_index >= len(self.samples):
                raise IndexError(f"sample_index out of range: {sample_index}")
            return sample_index
        if not question:
            raise ValueError("question or sample_index is required")
        key = normalize_question(question)
        if key not in self.question_index:
            raise KeyError(f"question not found in cache: {question}")
        return self.question_index[key]

    def _reconstruct_scores(self, sample: dict, threshold: float, hop_count: int):
        rel_dicts, ent_dicts = [], []
        for t in range(hop_count):
            rel_dicts.append(reconstruct_rel_dict(sample["rel_probs"][t], threshold))
            ent_dicts.append(
                reconstruct_ent_dict(sample["ent_indices"][t], sample["ent_scores"][t], threshold)
            )
        return rel_dicts, ent_dicts

    def _topics(self, sample: dict) -> list[str]:
        return [self.id2ent.get(int(topic_id), str(topic_id)) for topic_id in sample["topic_ids"]]

    def _prediction(self, sample: dict) -> dict[str, float]:
        prediction = {}
        for idx, val in zip(sample["e_score_indices"].tolist(), sample["e_score_values"].tolist()):
            if float(val) >= 0.5:
                prediction[self.id2ent.get(int(idx), str(idx))] = round(float(val), 4)
        return prediction

    def _serialize_paths(self, paths: list[tuple[list[int], list[int], float]]):
        return [
            {
                "path": _path_to_triples(nodes, rels, self.id2ent, self.id2rel),
                "log_score": round(float(score), 6),
            }
            for nodes, rels, score in paths
        ]

    def info(self) -> dict[str, Any]:
        return {
            "cache_path": self.cache_path,
            "input_dir": self.input_dir,
            "dataset": self.meta.get("dataset"),
            "split": self.meta.get("split"),
            "qa_file": self.meta.get("qa_file"),
            "num_samples": len(self.samples),
            "num_steps": self.meta.get("num_steps"),
            "topk_entities": self.meta.get("topk_entities"),
            "entity_count": self.meta.get("num_entities"),
            "relation_count": self.meta.get("num_relations"),
            "edge_source_count": len(self.valid_edges_dict),
        }
