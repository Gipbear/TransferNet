"""从训练 SPARQL 学习 CWQ 主题到答案的主关系并生成 Freebase 核验查询。"""
from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from kgqa.retrieve.cwq_attribute_evidence import CandidateValues

_FREEBASE_NS: Final = "http://rdf.freebase.com/ns/"
_TOPIC_RE: Final = re.compile(r"FILTER\s*\(\s*\?x\s*!=\s*ns:(m\.[\w]+)\s*\)", re.IGNORECASE)
_TRIPLE_RE: Final = re.compile(
    r"(\?[A-Za-z]\w*|ns:[mg]\.[\w]+)\s+ns:([\w.]+)\s+(\?[A-Za-z]\w*|ns:[mg]\.[\w]+)"
)
_TOKEN_RE: Final = re.compile(r"[a-z0-9]+")
_MIN_RELATION_SCORE: Final = 0.7
_MIN_RELATION_MARGIN: Final = 0.02
_MIN_CUE_SUPPORT: Final = 5
_MIN_CUE_PURITY: Final = 0.98


@dataclass(frozen=True, slots=True, order=True)
class RelationStep:
    predicate: str
    reverse: bool


@dataclass(frozen=True, slots=True, order=True)
class RelationPattern:
    steps: tuple[RelationStep, ...]


@dataclass(frozen=True, slots=True)
class RelationExample:
    question: str
    patterns: tuple[RelationPattern, ...]


@dataclass(frozen=True, slots=True)
class RelationRanking:
    patterns: tuple[RelationPattern, ...]
    scores: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CandidateRelationEvidence:
    sample_index: int
    topic_mid: str
    candidate_mid: str


@dataclass(frozen=True, slots=True)
class RelationCue:
    pattern: RelationPattern
    support: int


@dataclass(frozen=True, slots=True)
class RelationCueLexicon:
    entries: tuple[tuple[str, RelationCue], ...]


@dataclass(frozen=True, slots=True)
class _GraphEdge:
    source: str
    predicate: str
    target: str


def _shortest_patterns(edges: tuple[_GraphEdge, ...], topic: str) -> tuple[RelationPattern, ...]:
    queue = deque([(topic, ())])
    shortest_length: int | None = None
    patterns: list[RelationPattern] = []
    visited_depth = {topic: 0}
    while queue:
        node, steps = queue.popleft()
        if shortest_length is not None and len(steps) >= shortest_length:
            continue
        for edge in edges:
            if edge.source == node:
                next_node = edge.target
                step = RelationStep(edge.predicate, False)
            else:
                if edge.target != node:
                    continue
                next_node = edge.source
                step = RelationStep(edge.predicate, True)
            next_steps = (*steps, step)
            if next_node == "?x":
                shortest_length = len(next_steps)
                patterns.append(RelationPattern(next_steps))
                continue
            if len(next_steps) >= 3 or next_node == "?num":
                continue
            depth = len(next_steps)
            if visited_depth.get(next_node, depth + 1) < depth:
                continue
            visited_depth[next_node] = depth
            queue.append((next_node, next_steps))
    return tuple(dict.fromkeys(patterns))


def extract_main_relation_patterns(sparql: str) -> tuple[RelationPattern, ...]:
    edges = tuple(_GraphEdge(source, predicate, target) for source, predicate, target in _TRIPLE_RE.findall(sparql))
    patterns = (
        pattern
        for topic_mid in dict.fromkeys(_TOPIC_RE.findall(sparql))
        for pattern in _shortest_patterns(edges, f"ns:{topic_mid}")
    )
    return tuple(dict.fromkeys(patterns))


def rank_relation_patterns(
    training: Sequence[RelationExample], questions: Sequence[str], top_k: int = 3,
) -> list[RelationRanking]:
    labels = sorted({pattern for example in training for pattern in example.patterns})
    label_index = {pattern: index for index, pattern in enumerate(labels)}
    train_label_ids = [tuple(label_index[pattern] for pattern in example.patterns) for example in training]
    frequencies = Counter(label_id for label_ids in train_label_ids for label_id in label_ids)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    train_vectors = vectorizer.fit_transform(example.question for example in training)
    similarities = (vectorizer.transform(questions) @ train_vectors.T).toarray()
    rankings: list[RelationRanking] = []
    for row in similarities:
        scores = np.zeros(len(labels), dtype=np.float64)
        for example_index, label_ids in enumerate(train_label_ids):
            for label_id in label_ids:
                scores[label_id] = max(scores[label_id], row[example_index])
        ordered = sorted(range(len(labels)), key=lambda index: (-scores[index], -frequencies[index], index))[:top_k]
        rankings.append(RelationRanking(
            tuple(labels[index] for index in ordered), tuple(float(scores[index]) for index in ordered),
        ))
    return rankings


def _question_ngrams(question: str) -> frozenset[str]:
    tokens = _TOKEN_RE.findall(question.casefold())
    bigrams = (" ".join(tokens[index:index + 2]) for index in range(len(tokens) - 1))
    return frozenset((*tokens, *bigrams))


def learn_relation_cues(training: Sequence[RelationExample]) -> RelationCueLexicon:
    counts: defaultdict[str, Counter[RelationPattern]] = defaultdict(Counter)
    for example in training:
        if len(example.patterns) != 1:
            continue
        for ngram in _question_ngrams(example.question):
            counts[ngram][example.patterns[0]] += 1
    entries: list[tuple[str, RelationCue]] = []
    for ngram, pattern_counts in counts.items():
        pattern, support = pattern_counts.most_common(1)[0]
        total = pattern_counts.total()
        if support >= _MIN_CUE_SUPPORT and support / total >= _MIN_CUE_PURITY:
            entries.append((ngram, RelationCue(pattern, support)))
    return RelationCueLexicon(tuple(sorted(entries)))


def predict_relation_cue(question: str, lexicon: RelationCueLexicon) -> RelationPattern | None:
    cues = dict(lexicon.entries)
    votes: Counter[RelationPattern] = Counter()
    for ngram in _question_ngrams(question):
        cue = cues.get(ngram)
        if cue is not None:
            votes[cue.pattern] += cue.support
    if not votes:
        return None
    return min(votes, key=lambda pattern: (-votes[pattern], pattern))


def select_confident_pattern(
    ranking: RelationRanking, cue_pattern: RelationPattern | None = None,
) -> RelationPattern | None:
    if ranking.patterns and cue_pattern == ranking.patterns[0]:
        return cue_pattern
    if not ranking.patterns or ranking.scores[0] < _MIN_RELATION_SCORE:
        return None
    runner_up = ranking.scores[1] if len(ranking.scores) > 1 else 0.0
    if ranking.scores[0] - runner_up < _MIN_RELATION_MARGIN:
        return None
    return ranking.patterns[0]


def relation_filter_applies(supported_names: Sequence[str], cue_agreed: bool) -> bool:
    return bool(supported_names) or cue_agreed


def build_relation_query(
    pattern: RelationPattern, evidence: Sequence[CandidateRelationEvidence],
) -> str:
    rows = " ".join(
        f'(\"{item.sample_index}\" <{_FREEBASE_NS}{item.topic_mid}> <{_FREEBASE_NS}{item.candidate_mid}>)'
        for item in evidence
    )
    triples: list[str] = []
    current = "?topic"
    for index, step in enumerate(pattern.steps):
        next_node = "?candidate" if index == len(pattern.steps) - 1 else f"?step{index}"
        head, tail = (next_node, current) if step.reverse else (current, next_node)
        triples.append(f"{head} <{_FREEBASE_NS}{step.predicate}> {tail} .")
        current = next_node
    graph = " ".join(triples)
    return (
        "SELECT DISTINCT ?sample ?candidate WHERE { "
        f"VALUES (?sample ?topic ?candidate) {{ {rows} }} {graph} }}"
    )


def filter_supported_candidates(
    candidates: tuple[CandidateValues, ...], supported_names: frozenset[str] | None,
) -> tuple[CandidateValues, ...]:
    if supported_names is None:
        return candidates
    return tuple(candidate for candidate in candidates if candidate.name.strip().casefold() in supported_names)
