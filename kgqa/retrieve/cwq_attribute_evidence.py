"""抽取 CWQ 比较属性证据并执行确定性比较。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Final, Literal, assert_never

from kgqa.retrieve.cwq_comparison import ComparisonOperator

type PathStatus = Literal["found", "missing_relation", "no_path"]
type ComparableKind = Literal["date", "number"]

_FREEBASE_NS: Final = "http://rdf.freebase.com/ns/"
_XSD_PREFIX: Final = "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"
_DATE_RE: Final = re.compile(
    r"^([+-]?\d{1,6})(?:-(\d{2})(?:-(\d{2}))?)?(?:T(\d{2}):(\d{2}):(\d{2}))?"
)
_TYPED_LITERAL_RE: Final = re.compile(r'^"(.*)"\^\^<[^>]+>$')


@dataclass(frozen=True, slots=True)
class AttributePathResult:
    status: PathStatus
    terminal_entity_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CandidateValues:
    name: str
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True, order=True)
class _ComparableValue:
    kind: ComparableKind
    value: Decimal


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    selected_names: tuple[str, ...]
    comparable_candidate_count: int
    gold_present: bool
    gold_has_value: bool
    oracle_hit: bool


@dataclass(frozen=True, slots=True)
class UnsupportedAttributeChainError(Exception):
    chain_length: int

    def __str__(self) -> str:
        return f"仅支持一跳或两跳属性链，收到 {self.chain_length} 跳"


@dataclass(frozen=True, slots=True)
class MissingWhereClauseError(Exception):
    def __str__(self) -> str:
        return "官方 SPARQL 缺少 WHERE 子句"


def trace_attribute_chain(
    tuples: list[list[int]],
    relation_ids: dict[str, int],
    subject_id: int,
    chain: tuple[str, ...],
) -> AttributePathResult:
    frontier = {subject_id}
    for predicate in chain:
        relation_id = relation_ids.get(predicate)
        if relation_id is None:
            return AttributePathResult("missing_relation", ())
        next_frontier = {
            tail
            for head, relation, tail in tuples
            if head in frontier and relation == relation_id
        }
        if not next_frontier:
            return AttributePathResult("no_path", ())
        frontier = next_frontier
    return AttributePathResult("found", tuple(sorted(frontier)))


def build_sparql_query(chain: tuple[str, ...], mids: tuple[str, ...]) -> str:
    values = " ".join(f"<{_FREEBASE_NS}{mid}>" for mid in mids)
    if len(chain) == 1:
        pattern = f"?subject <{_FREEBASE_NS}{chain[0]}> ?value ."
    elif len(chain) == 2:
        pattern = (
            f"?subject <{_FREEBASE_NS}{chain[0]}> ?intermediate . "
            f"?intermediate <{_FREEBASE_NS}{chain[1]}> ?value ."
        )
    else:
        raise UnsupportedAttributeChainError(len(chain))
    return f"SELECT DISTINCT ?subject ?value WHERE {{ VALUES ?subject {{ {values} }} {pattern} }}"


def build_candidate_constrained_query(sparql: str, mids: tuple[str, ...]) -> str:
    query = re.sub(r"(?<=\s)OR(?=\s)", "||", sparql)
    query = re.sub(r"\)(?=\?x\s+ns:)", ") .\n", query)
    query = query.replace("xsd:datetime", "xsd:dateTime")
    if "xsd:" in query and "PREFIX xsd:" not in query:
        query = f"{_XSD_PREFIX}\n{query}"
    where_match = re.search(r"\bWHERE\s*\{", query, flags=re.IGNORECASE)
    if where_match is None:
        raise MissingWhereClauseError()
    values = " ".join(f"<{_FREEBASE_NS}{mid}>" for mid in mids)
    clause = f"\nVALUES ?x {{ {values} }}"
    return f"{query[:where_match.end()]}{clause}{query[where_match.end():]}"


def _clean_literal(raw: str) -> str:
    value = raw.strip()
    typed = _TYPED_LITERAL_RE.fullmatch(value)
    if typed is not None:
        return typed.group(1)
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def _parse_date(raw: str) -> _ComparableValue | None:
    match = _DATE_RE.match(raw)
    if match is None or match.end() != len(raw.rstrip("Z")):
        return None
    parts = [int(value) if value is not None else 0 for value in match.groups()]
    year, month, day, hour, minute, second = parts
    magnitude = abs(year) * 10**10 + month * 10**8 + day * 10**6 + hour * 10**4 + minute * 10**2 + second
    signed = -magnitude if year < 0 else magnitude
    return _ComparableValue("date", Decimal(signed))


def _parse_comparable(raw: str, datatype: str | None) -> _ComparableValue | None:
    value = _clean_literal(raw)
    if datatype == "dateTime" or "-" in value[1:]:
        parsed_date = _parse_date(value)
        if parsed_date is not None:
            return parsed_date
    try:
        return _ComparableValue("number", Decimal(value))
    except InvalidOperation:
        return None


def _matches(operator: ComparisonOperator, value: Decimal, threshold: Decimal) -> bool:
    match operator:
        case "lt":
            return value < threshold
        case "le":
            return value <= threshold
        case "gt":
            return value > threshold
        case "ge":
            return value >= threshold
        case "argmin" | "argmax":
            return False
        case unreachable:
            assert_never(unreachable)


def _normalized_names(names: tuple[str, ...]) -> set[str]:
    return {name.strip().casefold() for name in names}


def evaluate_candidates(
    operator: ComparisonOperator,
    threshold: str | None,
    datatype: str | None,
    candidates: tuple[CandidateValues, ...],
    gold_names: tuple[str, ...],
) -> EvaluationResult:
    parsed = {
        candidate.name: tuple(
            value
            for raw in candidate.values
            if (value := _parse_comparable(raw, datatype)) is not None
        )
        for candidate in candidates
    }
    comparable = {name: values for name, values in parsed.items() if values}
    match operator:
        case "lt" | "le" | "gt" | "ge":
            parsed_threshold = _parse_comparable(threshold or "", datatype)
            selected = () if parsed_threshold is None else tuple(
                name
                for name, values in comparable.items()
                if any(
                    value.kind == parsed_threshold.kind
                    and _matches(operator, value.value, parsed_threshold.value)
                    for value in values
                )
            )
        case "argmin" | "argmax":
            extrema = {
                name: (min(values) if operator == "argmin" else max(values))
                for name, values in comparable.items()
            }
            target = None if not extrema else (
                min(extrema.values()) if operator == "argmin" else max(extrema.values())
            )
            selected = tuple(name for name, value in extrema.items() if value == target)
        case unreachable:
            assert_never(unreachable)
    normalized_gold = _normalized_names(gold_names)
    candidate_names = _normalized_names(tuple(candidate.name for candidate in candidates))
    comparable_names = _normalized_names(tuple(comparable))
    selected_names = _normalized_names(selected)
    return EvaluationResult(
        selected,
        len(comparable),
        bool(normalized_gold & candidate_names),
        bool(normalized_gold & comparable_names),
        bool(normalized_gold & selected_names),
    )
