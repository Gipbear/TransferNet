"""汇总 CWQ 本地子图与外部属性证据。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from kgqa.retrieve.cwq_attribute_evidence import (
    CandidateValues,
    EvaluationResult,
    evaluate_candidates,
    trace_attribute_chain,
)
from kgqa.retrieve.cwq_comparison import ComparisonOperator


class CandidateInput(TypedDict):
    name: str
    mid_entity_ids: list[int]
    mids: list[str]


class SampleInput(TypedDict):
    sample_index: int
    id: str
    question: str
    compositionality_type: str
    operator: str
    threshold: str | None
    threshold_datatype: str | None
    attribute_chain: list[str]
    candidates: list[CandidateInput]


class _RetrievalInput(TypedDict):
    sample_index: int
    golden: list[str]


class _SubgraphInput(TypedDict):
    tuples: list[list[int]]


class _QaInput(TypedDict):
    subgraph: _SubgraphInput


@dataclass(frozen=True, slots=True)
class AuditConfig:
    samples: Path
    qa: Path
    retrieval: Path
    relations: Path
    output_dir: Path
    endpoint: str


@dataclass(frozen=True, slots=True)
class LocalSampleAudit:
    sample_index: int
    relation_vocab_complete: bool
    first_hop_candidate_count: int
    complete_value_candidate_count: int


@dataclass(frozen=True, slots=True)
class RemoteSampleAudit:
    sample_index: int
    sample_id: str
    question: str
    compositionality_type: str
    operator: str
    threshold: str | None
    attribute_chain: tuple[str, ...]
    golden: tuple[str, ...]
    candidate_values: tuple[CandidateValues, ...]
    evaluation: EvaluationResult


@dataclass(frozen=True, slots=True)
class StrictSampleAudit:
    sample_index: int
    sample_id: str
    question: str
    compositionality_type: str
    operator: str
    golden: tuple[str, ...]
    selected_mids: tuple[str, ...]
    selected_names: tuple[str, ...]
    oracle_hit: bool


@dataclass(frozen=True, slots=True)
class InvalidOperatorError(Exception):
    operator: str

    def __str__(self) -> str:
        return f"未知比较运算符: {self.operator}"


def parse_operator(raw: str) -> ComparisonOperator:
    operators: dict[str, ComparisonOperator] = {
        "lt": "lt",
        "le": "le",
        "gt": "gt",
        "ge": "ge",
        "argmin": "argmin",
        "argmax": "argmax",
    }
    if raw not in operators:
        raise InvalidOperatorError(raw)
    return operators[raw]


def load_samples(path: Path) -> list[SampleInput]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_goldens(path: Path, selected: set[int]) -> dict[int, tuple[str, ...]]:
    goldens: dict[int, tuple[str, ...]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record: _RetrievalInput = json.loads(line)
            if record["sample_index"] in selected:
                goldens[record["sample_index"]] = tuple(record["golden"])
    return goldens


def read_relation_ids(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8") as handle:
        return {line.rstrip("\n"): index for index, line in enumerate(handle)}


def audit_local_subgraphs(
    config: AuditConfig,
    samples_by_index: dict[int, SampleInput],
    relation_ids: dict[str, int],
) -> tuple[LocalSampleAudit, ...]:
    audits: list[LocalSampleAudit] = []
    with config.qa.open(encoding="utf-8") as handle:
        for sample_index, line in enumerate(handle):
            sample = samples_by_index.get(sample_index)
            if sample is None:
                continue
            qa: _QaInput = json.loads(line)
            tuples = qa["subgraph"]["tuples"]
            chain = tuple(sample["attribute_chain"])
            first_hop_count = 0
            complete_count = 0
            for candidate in sample["candidates"]:
                first_hop_found = False
                complete_found = False
                for entity_id in candidate["mid_entity_ids"]:
                    first_hop = trace_attribute_chain(tuples, relation_ids, entity_id, chain[:1])
                    complete = trace_attribute_chain(tuples, relation_ids, entity_id, chain)
                    first_hop_found = first_hop_found or first_hop.status == "found"
                    complete_found = complete_found or complete.status == "found"
                first_hop_count += first_hop_found
                complete_count += complete_found
            audits.append(LocalSampleAudit(
                sample_index,
                all(predicate in relation_ids for predicate in chain),
                first_hop_count,
                complete_count,
            ))
    return tuple(audits)


def build_remote_audits(
    samples: list[SampleInput],
    goldens: dict[int, tuple[str, ...]],
    values_by_chain: dict[tuple[str, ...], dict[str, tuple[str, ...]]],
) -> tuple[RemoteSampleAudit, ...]:
    audits: list[RemoteSampleAudit] = []
    for sample in samples:
        chain = tuple(sample["attribute_chain"])
        chain_values = values_by_chain[chain]
        candidate_values = tuple(
            CandidateValues(
                candidate["name"],
                tuple(value for mid in candidate["mids"] for value in chain_values.get(mid, ())),
            )
            for candidate in sample["candidates"]
        )
        golden = goldens[sample["sample_index"]]
        evaluation = evaluate_candidates(
            parse_operator(sample["operator"]),
            sample["threshold"],
            sample["threshold_datatype"],
            candidate_values,
            golden,
        )
        audits.append(RemoteSampleAudit(
            sample["sample_index"],
            sample["id"],
            sample["question"],
            sample["compositionality_type"],
            sample["operator"],
            sample["threshold"],
            chain,
            golden,
            candidate_values,
            evaluation,
        ))
    return tuple(audits)


def build_strict_audits(
    samples: list[SampleInput],
    goldens: dict[int, tuple[str, ...]],
    selections: dict[int, tuple[str, ...]],
) -> tuple[StrictSampleAudit, ...]:
    audits: list[StrictSampleAudit] = []
    for sample in samples:
        selected_mids = selections[sample["sample_index"]]
        selected_mid_set = set(selected_mids)
        selected_names = tuple(
            candidate["name"]
            for candidate in sample["candidates"]
            if selected_mid_set.intersection(candidate["mids"])
        )
        golden = goldens[sample["sample_index"]]
        normalized_gold = {name.strip().casefold() for name in golden}
        normalized_selected = {name.strip().casefold() for name in selected_names}
        audits.append(StrictSampleAudit(
            sample["sample_index"],
            sample["id"],
            sample["question"],
            sample["compositionality_type"],
            sample["operator"],
            golden,
            selected_mids,
            selected_names,
            bool(normalized_gold & normalized_selected),
        ))
    return tuple(audits)

