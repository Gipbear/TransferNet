"""解析 CWQ 比较约束并建立样本内实体对齐。"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Literal, assert_never

from kgqa.retrieve.cwq_mid_alignment import AlignmentMismatchError

type ComparisonKind = Literal["comparative", "superlative"]
type ComparisonOperator = Literal["lt", "le", "gt", "ge", "argmin", "argmax"]
type TriplePattern = tuple[str, str, str]

_TRIPLE_RE: Final = re.compile(
    r"(\?[A-Za-z]\w*|ns:[mg]\.[\w]+)\s+ns:([\w.]+)\s+(\?[A-Za-z]\w*|ns:[mg]\.[\w]+)"
)
_FILTER_RE: Final = re.compile(
    r"FILTER\s*\(\s*(?:xsd:\w+\s*\(\s*)?(\?num)\s*\)?\s*(<=|>=|<|>)\s*([^\)]+?)\s*\)",
    re.IGNORECASE,
)
_ORDER_RE: Final = re.compile(
    r"ORDER BY\s+(DESC\s*\()?\s*(?:xsd:integer\s*\(\s*)?(\?num)",
    re.IGNORECASE,
)
_TYPED_LITERAL_RE: Final = re.compile(r'^"([^"]+)"\^\^xsd:(\w+)$')
_INTEGER_RE: Final = re.compile(r"^[+-]?\d+$")
_DECIMAL_RE: Final = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+)$")


class ComparisonParseError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ComparisonSpec:
    operator: ComparisonOperator
    attribute_chain: tuple[str, ...]
    threshold: str | None
    threshold_datatype: str | None


def _trace_attribute_chain(sparql: str, target_variable: str) -> tuple[str, ...]:
    triples: list[TriplePattern] = _TRIPLE_RE.findall(sparql)
    current = target_variable
    seen: set[str] = set()
    reversed_chain: list[str] = []
    while current != "?x" and current not in seen:
        seen.add(current)
        matches = [triple for triple in triples if triple[2] == current]
        if not matches:
            raise ComparisonParseError(f"无法从 {target_variable} 回溯到 ?x")
        source, relation, _ = matches[-1]
        reversed_chain.append(relation)
        current = source
    if current != "?x":
        raise ComparisonParseError(f"属性链存在环: {target_variable}")
    return tuple(reversed(reversed_chain))


def _parse_threshold(raw_threshold: str) -> tuple[str, str]:
    raw = raw_threshold.strip()
    typed_match = _TYPED_LITERAL_RE.fullmatch(raw)
    if typed_match is not None:
        return typed_match.group(1), typed_match.group(2)
    if _INTEGER_RE.fullmatch(raw):
        return raw, "integer"
    if _DECIMAL_RE.fullmatch(raw):
        return raw, "decimal"
    return raw, "expression"


def _parse_filter_operator(symbol: str) -> ComparisonOperator:
    operators: dict[str, ComparisonOperator] = {
        "<": "lt",
        "<=": "le",
        ">": "gt",
        ">=": "ge",
    }
    if symbol not in operators:
        raise ComparisonParseError(f"未知比较运算符: {symbol}")
    return operators[symbol]


def extract_comparison_spec(compositionality_type: ComparisonKind, sparql: str) -> ComparisonSpec:
    match compositionality_type:
        case "comparative":
            matches = _FILTER_RE.findall(sparql)
            if not matches:
                raise ComparisonParseError("comparative SPARQL 缺少 ?num FILTER")
            variable, symbol, raw_threshold = matches[-1]
            threshold, datatype = _parse_threshold(raw_threshold)
            operator = _parse_filter_operator(symbol)
        case "superlative":
            match = _ORDER_RE.search(sparql)
            if match is None:
                raise ComparisonParseError("superlative SPARQL 缺少 ORDER BY ?num")
            direction, variable = match.groups()
            operator = "argmax" if direction is not None else "argmin"
            threshold = None
            datatype = None
        case unreachable:
            assert_never(unreachable)
    chain = _trace_attribute_chain(sparql, variable)
    return ComparisonSpec(operator, chain, threshold, datatype)


def align_sample_entity_ids(
    rog_triples: list[list[int]],
    mid_triples: list[list[int]],
    row_index: int,
) -> dict[int, tuple[int, ...]]:
    if len(rog_triples) != len(mid_triples):
        raise AlignmentMismatchError(row_index, "三元组数量不一致")
    aligned: defaultdict[int, set[int]] = defaultdict(set)
    for rog, mid in zip(rog_triples, mid_triples):
        if rog[1] != mid[1]:
            raise AlignmentMismatchError(row_index, "关系序列不一致")
        aligned[rog[0]].add(mid[0])
        aligned[rog[2]].add(mid[2])
    return {rog_id: tuple(sorted(mid_ids)) for rog_id, mid_ids in aligned.items()}


def rerank_merge_answers(selected: list[str], original: list[str]) -> list[str]:
    selected_keys = {answer.strip().casefold() for answer in selected}
    original_keys = {answer.strip().casefold() for answer in original}
    ordered = [answer for answer in original if answer.strip().casefold() in selected_keys]
    ordered.extend(answer for answer in selected if answer.strip().casefold() not in original_keys)
    ordered.extend(answer for answer in original if answer.strip().casefold() not in selected_keys)
    merged: list[str] = []
    seen: set[str] = set()
    for answer in ordered:
        key = answer.strip().casefold()
        if key not in seen:
            seen.add(key)
            merged.append(answer)
    return merged
