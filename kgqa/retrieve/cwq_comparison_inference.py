"""表示无测试标签的 CWQ 检索输入并预测问题类型。"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

type QuestionType = Literal["composition", "conjunction", "comparative", "superlative"]
type Resolution = Literal["resolved", "ambiguous", "unresolved"]


class ReasonPath(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    path: tuple[tuple[str, str, str], ...]


class InferenceRetrieval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    sample_index: int
    question: str
    topics: tuple[str, ...]
    mmr_reason_paths: tuple[ReasonPath, ...]


class SubgraphInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    tuples: list[list[int]]


class PairedQA(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    question: str
    entities: tuple[int, ...]
    subgraph: SubgraphInput


@dataclass(frozen=True, slots=True)
class TypeExample:
    question: str
    kind: QuestionType


@dataclass(frozen=True, slots=True)
class TypePrediction:
    kind: QuestionType
    confidence: float


@dataclass(frozen=True, slots=True)
class EntityLookup:
    aligned_ids: Mapping[int, tuple[int, ...]]
    rog_names: Mapping[int, str]
    mid_names: Mapping[int, str]


class Candidate(TypedDict):
    name: str
    mids: list[str]
    resolution: Resolution


def predict_question_types(
    training: Sequence[TypeExample], questions: Sequence[str],
) -> list[TypePrediction]:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vectors = vectorizer.fit_transform(item.question for item in training)
    model = LogisticRegression(max_iter=1000).fit(vectors, [item.kind for item in training])
    probabilities = model.predict_proba(vectorizer.transform(questions))
    predictions: list[TypePrediction] = []
    for row in probabilities:
        winning = int(row.argmax())
        predictions.append(TypePrediction(model.classes_[winning], float(row[winning])))
    return predictions


def build_candidates(retrieval: InferenceRetrieval, lookup: EntityLookup) -> list[Candidate]:
    rog_ids_by_name: defaultdict[str, list[int]] = defaultdict(list)
    for rog_id in lookup.aligned_ids:
        rog_ids_by_name[lookup.rog_names[rog_id]].append(rog_id)
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for reason_path in retrieval.mmr_reason_paths:
        if not reason_path.path:
            continue
        name = reason_path.path[-1][2]
        if name in seen:
            continue
        seen.add(name)
        mid_ids = sorted({
            mid_id
            for rog_id in rog_ids_by_name.get(name, ())
            for mid_id in lookup.aligned_ids[rog_id]
        })
        resolution: Resolution = "resolved" if len(mid_ids) == 1 else "ambiguous" if mid_ids else "unresolved"
        candidates.append({
            "name": name,
            "mids": [lookup.mid_names[mid_id] for mid_id in mid_ids],
            "resolution": resolution,
        })
    return candidates


def build_topics(topic_ids: tuple[int, ...], lookup: EntityLookup) -> list[Candidate]:
    topics: list[Candidate] = []
    for topic_id in topic_ids:
        mid_ids = lookup.aligned_ids.get(topic_id, ())
        resolution: Resolution = "resolved" if len(mid_ids) == 1 else "ambiguous" if mid_ids else "unresolved"
        topics.append({
            "name": lookup.rog_names[topic_id],
            "mids": [lookup.mid_names[mid_id] for mid_id in mid_ids],
            "resolution": resolution,
        })
    return topics
