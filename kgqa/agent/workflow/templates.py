"""只从训练问题及训练 SPARQL 学习路由、属性链与多跳模板。"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.utils.validation import check_is_fitted

from kgqa.agent.workflow.models import RouteDecision
from kgqa.retrieve.cwq_comparison import ComparisonParseError, extract_comparison_spec
from kgqa.retrieve.cwq_comparison_condition import extract_threshold

TRIPLE_RE = re.compile(
    r'(\?[a-zA-Z0-9_]+|ns:[a-zA-Z0-9_.]+)\s+ns:([a-zA-Z0-9_.]+)\s+'
    r'(\?[a-zA-Z0-9_]+|ns:[a-zA-Z0-9_.]+|"[^"]*")'
)


class ChainIndex:
    def __init__(self, examples: list[tuple[str, tuple]], min_score: float = 0.0) -> None:
        self.labels = sorted({chain for _, chain in examples})
        label_ids = {chain: i for i, chain in enumerate(self.labels)}
        self.y = np.array([label_ids[chain] for _, chain in examples])
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))
        ])
        questions = [question for question, _ in examples]
        self.matrix = self.vectorizer.fit_transform(questions)
        self.min_score = min_score

    def rank(self, question: str, top_k: int = 5) -> list[tuple[tuple, float]]:
        similarities = (self.vectorizer.transform([question]) @ self.matrix.T).toarray()[0]
        scores = np.zeros(len(self.labels))
        np.maximum.at(scores, self.y, similarities)
        ordered = sorted(range(len(scores)), key=lambda i: (-scores[i], self.labels[i]))
        return [(self.labels[i], float(scores[i])) for i in ordered[:top_k] if scores[i] > self.min_score]


class TextClassifier:
    def __init__(self, examples: list[tuple[str, str]]) -> None:
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))
        ])
        matrix = self.vectorizer.fit_transform([question for question, _ in examples])
        self.model = LogisticRegression(C=10.0, max_iter=1000, random_state=17, class_weight="balanced")
        self.model.fit(matrix, [label for _, label in examples])

    @classmethod
    def from_fitted(cls, vectorizer: Any, model: Any) -> "TextClassifier":
        """装入离线拟合的特征与概率模型，沿用冻结模板的预测接口。"""
        if not callable(getattr(model, "predict_proba", None)):
            raise TypeError("模型必须提供 predict_proba；间隔分类器须先做概率校准")
        check_is_fitted(vectorizer)
        check_is_fitted(model)
        instance = cls.__new__(cls)
        instance.vectorizer, instance.model = vectorizer, model
        return instance

    def predict(self, question: str) -> tuple[str, float]:
        scores = self.model.predict_proba(self.vectorizer.transform([question]))[0]
        index = int(scores.argmax())
        return str(self.model.classes_[index]), float(scores[index])


def numeric_chain(sparql: str) -> tuple[str, ...] | None:
    outgoing, literals, found = {}, {}, None
    for head, relation, tail in TRIPLE_RE.findall(sparql):
        if head == "?x" and tail.startswith("?"):
            outgoing[tail] = relation
        elif tail.startswith('"') and head.startswith("?"):
            literals[head] = relation
        elif head == "?x" and tail.startswith('"'):
            found = (relation,)
    for variable, relation in outgoing.items():
        if variable in literals:
            found = (relation, literals[variable])
    return found


def relation_chains(sparql: str) -> tuple[tuple, ...]:
    graph = defaultdict(list)
    for head, relation, tail in TRIPLE_RE.findall(sparql):
        if relation.startswith(("type.", "common.", "base.")) or tail.startswith('"'):
            continue
        graph[head].append((tail, relation, False))
        graph[tail].append((head, relation, True))
    found = set()
    for start in sorted(node for node in graph if node.startswith("ns:")):
        frontier = [(start, (), frozenset([start]))]
        while frontier:
            node, path, visited = frontier.pop()
            for following, relation, reverse in graph[node]:
                if following in visited:
                    continue
                next_path = (*path, (relation, reverse))
                if following == "?x":
                    found.add(next_path)
                elif len(next_path) < 4:
                    frontier.append((following, next_path, visited | {following}))
    return tuple(sorted(found))


class TemplateLibrary:
    @classmethod
    def fit(cls, train_path: str | Path) -> "TemplateLibrary":
        records = json.loads(Path(train_path).read_text())
        self = cls()
        self.router = TextClassifier([(r["question"], r["compositionality_type"]) for r in records])
        self._fit_attributes(records)
        self.questions = [r["question"] for r in records]
        self.patterns = [relation_chains(r["sparql"]) for r in records]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.questions)
        self.training_n = len(records)
        return self

    def _fit_attributes(self, records: list[dict]) -> None:
        comparison, operators, numeric = defaultdict(list), defaultdict(list), []
        for record in records:
            kind, question, sparql = record["compositionality_type"], record["question"], record["sparql"]
            if kind == "conjunction":
                chain = numeric_chain(sparql)
                if chain:
                    numeric.append((question, chain))
            if kind not in ("comparative", "superlative"):
                continue
            try:
                spec = extract_comparison_spec(kind, sparql)
            except ComparisonParseError:
                continue
            comparison[kind].append((question, spec.attribute_chain))
            operators[kind].append((question, spec.operator))
        self.numeric_index = ChainIndex(numeric, 0.12)
        self.comparison_indices = {kind: ChainIndex(values) for kind, values in comparison.items()}
        self.operators = {kind: TextClassifier(values) for kind, values in operators.items()}

    def classify(self, question: str) -> RouteDecision:
        kind, confidence = self.router.predict(question)
        return RouteDecision(kind, confidence)

    def numeric_chains(self, question: str, top_k: int = 5) -> list[tuple]:
        return [chain for chain, _ in self.numeric_index.rank(question, top_k)]

    def comparison_plan(self, question: str, kind: str) -> dict:
        ranked = self.comparison_indices[kind].rank(question, 2)
        if not ranked:
            return {}
        operator, confidence = self.operators[kind].predict(question)
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1]
        candidate_chains = [list(item[0]) for item in ranked[:2]]
        return {"chain": list(ranked[0][0]), "candidate_chains": candidate_chains,
                "margin": margin, "operator": operator,
                "confidence": confidence, "threshold": extract_threshold(question)}

    def extract_patterns(self, question: str, k_neighbors: int = 32, p_limit: int = 10) -> list[dict]:
        scores = (self.vectorizer.transform([question]) @ self.matrix.T).toarray()[0]
        neighbors = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k_neighbors]
        ranked = {}
        for index in neighbors:
            for pattern in self.patterns[index]:
                if self.questions[index].strip().casefold() == question.strip().casefold():
                    continue
                ranked[pattern] = max(ranked.get(pattern, 0), float(scores[index]))
        result = []
        for minimum, maximum in ((1, 2), (3, 4)):
            ordered = sorted(ranked, key=lambda p: (-ranked[p], p))
            selected = [p for p in ordered if minimum <= len(p) <= maximum and ranked[p] > 0][:p_limit]
            for pattern in selected:
                result.append({"steps": [{"predicate": p, "reverse": r} for p, r in pattern],
                               "score": ranked[pattern]})
        return result

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> "TemplateLibrary":
        return joblib.load(path)
