"""从训练问题和检索路径推断 CWQ 候选的主题锚点及主关系支持。"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

type RelationSignature = tuple[str, ...]

_STOPWORDS: Final = {"the", "of", "a", "an", "in", "at", "on", "to", "for", "and", "or", "is", "was", "were"}
_TOKEN_RE: Final = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class MainRelationExample:
    question: str
    signatures: tuple[RelationSignature, ...]


@dataclass(frozen=True, slots=True)
class PathEvidence:
    topic: str
    tail: str
    relations: RelationSignature


def predict_relation_signatures(
    training: Sequence[MainRelationExample],
    questions: Sequence[str],
    top_k: int,
    min_similarity: float,
) -> list[tuple[RelationSignature, ...] | None]:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    train_vectors = vectorizer.fit_transform(item.question for item in training)
    similarities = (vectorizer.transform(questions) @ train_vectors.T).toarray()
    predictions: list[tuple[RelationSignature, ...] | None] = []
    for row in similarities:
        if float(np.max(row)) < min_similarity:
            predictions.append(None)
            continue
        closest = np.argsort(-row)[:top_k]
        signatures = dict.fromkeys(
            signature
            for index in closest
            for signature in training[int(index)].signatures
        )
        predictions.append(tuple(signatures))
    return predictions


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower())) - _STOPWORDS


def _supported_topics(question: str, topics: tuple[str, ...]) -> set[str]:
    question_tokens = _tokens(question)
    overlap = {
        topic: len(question_tokens & _tokens(topic)) / max(1, len(_tokens(topic)))
        for topic in topics
    }
    strongest = max(overlap.values(), default=0.0)
    if strongest == 0:
        return set(topics)
    return {topic for topic, score in overlap.items() if score == strongest}


def supported_names(
    question: str,
    topics: tuple[str, ...],
    paths: tuple[PathEvidence, ...],
    signatures: tuple[RelationSignature, ...] | None,
) -> tuple[str, ...]:
    supported_topics = _supported_topics(question, topics)
    allowed_signatures = None if signatures is None else set(signatures)
    names = dict.fromkeys(
        path.tail
        for path in paths
        if path.topic in supported_topics
        and (allowed_signatures is None or path.relations in allowed_signatures)
    )
    return tuple(names)
