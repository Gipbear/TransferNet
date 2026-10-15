"""用问题文本检索 CWQ 比较属性链并统计选择准确率。"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, assert_never

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from kgqa.retrieve.cwq_attribute_coverage import AttributeChain
from kgqa.retrieve.cwq_comparison import ComparisonKind

type Analyzer = Literal["word", "char_wb"]


@dataclass(frozen=True, slots=True)
class ComparisonQuestion:
    question: str
    kind: ComparisonKind
    chain: AttributeChain


@dataclass(frozen=True, slots=True)
class ChainRanking:
    chains: tuple[AttributeChain, ...]
    scores: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AccuracySlice:
    sample_count: int
    seen_label_count: int
    top1_hit_count: int
    top3_hit_count: int
    seen_top1_hit_count: int
    seen_top3_hit_count: int


@dataclass(frozen=True, slots=True)
class SelectorEvaluation:
    overall: AccuracySlice
    comparative: AccuracySlice
    superlative: AccuracySlice


def rank_by_frequency(train: Sequence[ComparisonQuestion], top_k: int = 3) -> ChainRanking:
    frequencies = Counter(sample.chain for sample in train)
    ordered = sorted(frequencies, key=lambda chain: (-frequencies[chain], chain))[:top_k]
    return ChainRanking(tuple(ordered), tuple(float(frequencies[chain]) for chain in ordered))


def rank_attribute_chains(
    train: Sequence[ComparisonQuestion],
    questions: Sequence[str],
    analyzer: Analyzer,
) -> list[ChainRanking]:
    """每条属性链取训练问题中的最大 TF-IDF 相似度，按分数排序。"""
    labels = sorted({sample.chain for sample in train})
    label_index = {chain: index for index, chain in enumerate(labels)}
    train_label_ids = np.fromiter((label_index[item.chain] for item in train), dtype=np.int64)
    frequencies = np.bincount(train_label_ids, minlength=len(labels))
    match analyzer:
        case "word":
            ngram_range = (1, 2)
        case "char_wb":
            ngram_range = (3, 5)
        case unreachable:
            assert_never(unreachable)
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, sublinear_tf=True)
    train_vectors = vectorizer.fit_transform(item.question for item in train)
    question_vectors = vectorizer.transform(questions)
    similarities = (question_vectors @ train_vectors.T).toarray()
    rankings: list[ChainRanking] = []
    for row in similarities:
        scores = np.zeros(len(labels), dtype=np.float64)
        np.maximum.at(scores, train_label_ids, row)
        ordered = np.lexsort((np.arange(len(labels)), -frequencies, -scores))[:3]
        rankings.append(
            ChainRanking(
                tuple(labels[index] for index in ordered),
                tuple(float(scores[index]) for index in ordered),
            )
        )
    return rankings


def evaluate_rankings(
    train: Sequence[ComparisonQuestion],
    dev: Sequence[ComparisonQuestion],
    rankings: Sequence[ChainRanking],
) -> SelectorEvaluation:
    known_chains = {sample.chain for sample in train}
    paired = list(zip(dev, rankings, strict=True))
    comparative = [(sample, ranking) for sample, ranking in paired if sample.kind == "comparative"]
    superlative = [(sample, ranking) for sample, ranking in paired if sample.kind == "superlative"]
    return SelectorEvaluation(
        _evaluate_slice(paired, known_chains),
        _evaluate_slice(comparative, known_chains),
        _evaluate_slice(superlative, known_chains),
    )


def _evaluate_slice(
    paired: Sequence[tuple[ComparisonQuestion, ChainRanking]],
    known_chains: set[AttributeChain],
) -> AccuracySlice:
    seen_count = top1_count = top3_count = seen_top1_count = seen_top3_count = 0
    for sample, ranking in paired:
        seen = sample.chain in known_chains
        top1_hit = bool(ranking.chains) and ranking.chains[0] == sample.chain
        top3_hit = sample.chain in ranking.chains[:3]
        seen_count += seen
        top1_count += top1_hit
        top3_count += top3_hit
        seen_top1_count += seen and top1_hit
        seen_top3_count += seen and top3_hit
    return AccuracySlice(
        len(paired), seen_count, top1_count, top3_count, seen_top1_count, seen_top3_count
    )
