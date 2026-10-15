"""从 CWQ 问题文本预测比较方向并提取显式阈值。"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from kgqa.retrieve.cwq_attribute_audit import parse_operator
from kgqa.retrieve.cwq_comparison import ComparisonKind, ComparisonOperator

_MONTH: Final = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_MONTH_DAY_RE: Final = re.compile(
    rf"\b(?P<month>{_MONTH})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
_DAY_MONTH_RE: Final = re.compile(
    rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH})\.?\s*,?\s*(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
_MONTH_YEAR_RE: Final = re.compile(rf"\b(?P<month>{_MONTH})\.?\s+(?:of\s+)?(?P<year>\d{{4}})\b", re.IGNORECASE)
_NUMERIC_DATE_RE: Final = re.compile(
    r"(?<!\w)(?P<first>\d{1,4})[-/](?P<second>\d{1,2})(?:[-/](?P<third>\d{2,4}))?(?!\w)"
)
_NUMBER_RE: Final = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w.])")
_CUE_RE: Final = re.compile(
    r"\b(?:before|after|prior to|earlier than|later than|greater than|less than|"
    r"fewer than|more than|higher than|lower than|smaller than|larger than|"
    r"under|over|since|until|from|beginning|starting|at least|at most)\b",
    re.IGNORECASE,
)
_MONTH_NUMBERS: Final = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


@dataclass(frozen=True, slots=True)
class OperatorExample:
    question: str
    kind: ComparisonKind
    operator: ComparisonOperator


@dataclass(frozen=True, slots=True)
class ComparisonQuery:
    question: str
    kind: ComparisonKind


@dataclass(frozen=True, slots=True)
class OperatorPrediction:
    operator: ComparisonOperator
    confidence: float


@dataclass(frozen=True, slots=True)
class _ThresholdCandidate:
    start: int
    end: int
    value: str


def predict_operators(
    training: Sequence[OperatorExample],
    queries: Sequence[ComparisonQuery],
) -> list[OperatorPrediction]:
    predictions: dict[int, OperatorPrediction] = {}
    for kind in ("comparative", "superlative"):
        examples = [item for item in training if item.kind == kind]
        selected = [(index, item.question) for index, item in enumerate(queries) if item.kind == kind]
        if not selected:
            continue
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        train_vectors = vectorizer.fit_transform(item.question for item in examples)
        model = LogisticRegression(max_iter=1000).fit(train_vectors, [item.operator for item in examples])
        probabilities = model.predict_proba(vectorizer.transform(question for _, question in selected))
        for (index, _), row in zip(selected, probabilities, strict=True):
            winning = int(row.argmax())
            predictions[index] = OperatorPrediction(parse_operator(str(model.classes_[winning])), float(row[winning]))
    return [predictions[index] for index in range(len(queries))]


def _normalized_date(year: int, month: int, day: int | None) -> str | None:
    try:
        date(year, month, day or 1)
    except ValueError:
        return None
    if day is None:
        return f"{year:04d}-{month:02d}"
    return f"{year:04d}-{month:02d}-{day:02d}"


def _named_date(match: re.Match[str]) -> str | None:
    month = _MONTH_NUMBERS[match.group("month")[:3].lower()]
    day_group = match.groupdict().get("day")
    day = int(day_group) if day_group is not None else None
    return _normalized_date(int(match.group("year")), month, day)


def _numeric_date(match: re.Match[str]) -> str | None:
    first, second, third = match.group("first", "second", "third")
    if third is None:
        if len(first) != 4:
            return None
        return _normalized_date(int(first), int(second), None)
    if len(first) == 4:
        year, month, day = int(first), int(second), int(third)
    else:
        year = int(third) if len(third) == 4 else 2000 + int(third)
        month, day = int(first), int(second)
        if month > 12 and day <= 12:
            month, day = day, month
    return _normalized_date(year, month, day)


def _threshold_candidates(question: str) -> list[_ThresholdCandidate]:
    candidates: list[_ThresholdCandidate] = []
    date_patterns = (
        (_MONTH_DAY_RE, _named_date),
        (_DAY_MONTH_RE, _named_date),
        (_MONTH_YEAR_RE, _named_date),
        (_NUMERIC_DATE_RE, _numeric_date),
    )
    for pattern, normalize in date_patterns:
        for match in pattern.finditer(question):
            if any(match.start() < item.end and item.start < match.end() for item in candidates):
                continue
            value = normalize(match)
            if value is not None:
                candidates.append(_ThresholdCandidate(match.start(), match.end(), value))
    for match in _NUMBER_RE.finditer(question):
        if any(match.start() < item.end and item.start < match.end() for item in candidates):
            continue
        candidates.append(_ThresholdCandidate(match.start(), match.end(), match.group().replace(",", "")))
    return sorted(candidates, key=lambda item: item.start)


def extract_threshold(question: str) -> str | None:
    candidates = _threshold_candidates(question)
    if len(candidates) == 1:
        return candidates[0].value
    cues = tuple(_CUE_RE.finditer(question))
    scored = [
        (candidate.start - cue.end(), candidate)
        for candidate in candidates
        for cue in cues
        if 0 <= candidate.start - cue.end() <= 50
    ]
    if not scored:
        return None
    best_distance = min(distance for distance, _ in scored)
    best_values = {candidate.value for distance, candidate in scored if distance == best_distance}
    return next(iter(best_values)) if len(best_values) == 1 else None
