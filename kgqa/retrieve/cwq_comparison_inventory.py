"""汇总 CWQ 比较题候选实体、MID 和属性链。"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Literal, TypedDict

from kgqa.retrieve.cwq_comparison import (
    ComparisonKind,
    ComparisonSpec,
    align_sample_entity_ids,
    extract_comparison_spec,
)
from kgqa.retrieve.cwq_mid_alignment import AlignmentMismatchError

type Resolution = Literal["resolved", "ambiguous", "unresolved"]


class _OfficialRecord(TypedDict):
    ID: str
    question: str
    compositionality_type: str
    sparql: str


class _SubgraphRecord(TypedDict):
    tuples: list[list[int]]


class _QaRecord(TypedDict):
    id: str
    subgraph: _SubgraphRecord


class _ReasonPath(TypedDict):
    path: list[list[str]]


class _RetrievalRecord(TypedDict):
    sample_index: int
    mmr_reason_paths: list[_ReasonPath]


class _PredictionRecord(TypedDict):
    sample_index: int
    hit_any: int


@dataclass(frozen=True, slots=True)
class InventoryConfig:
    official_test: Path
    rog_qa: Path
    mid_qa: Path
    retrieval: Path
    predictions: Path
    rog_entities: Path
    mid_entities: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class CandidateIds:
    name: str
    rog_entity_ids: tuple[int, ...]
    mid_entity_ids: tuple[int, ...]
    resolution: Resolution


@dataclass(frozen=True, slots=True)
class SampleDraft:
    sample_index: int
    sample_id: str
    question: str
    kind: ComparisonKind
    hit_any: int
    spec: ComparisonSpec
    candidates: tuple[CandidateIds, ...]


@dataclass(frozen=True, slots=True)
class InventorySummary:
    sample_count: int
    comparative_count: int
    superlative_count: int
    hit_any_error_count: int
    hit_any_error_candidate_count: int
    hit_any_error_ambiguous_candidate_count: int
    candidate_count: int
    resolved_candidate_count: int
    ambiguous_candidate_count: int
    unresolved_candidate_count: int
    all_candidates_resolved_sample_count: int
    hit_any_error_all_candidates_mapped_sample_count: int
    hit_any_error_all_candidates_resolved_sample_count: int
    unique_mid_count: int
    unique_hit_any_error_mid_count: int
    unique_attribute_chain_count: int
    unique_predicate_count: int


@dataclass(frozen=True, slots=True)
class InventoryResult:
    drafts: tuple[SampleDraft, ...]
    summary: InventorySummary


def _comparison_kind(raw: str) -> ComparisonKind | None:
    if raw == "comparative" or raw == "superlative":
        return raw
    return None


def _load_retrieval(path: Path) -> list[_RetrievalRecord]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _load_predictions(path: Path) -> list[_PredictionRecord]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_selected_vocab(path: Path, selected_ids: set[int]) -> dict[int, str]:
    names: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for entity_id, line in enumerate(handle):
            if entity_id in selected_ids:
                names[entity_id] = line.rstrip("\n")
    return names


def _candidate_names(record: _RetrievalRecord) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for reason_path in record["mmr_reason_paths"]:
        edges = reason_path["path"]
        if not edges:
            continue
        name = edges[-1][2]
        if name not in seen:
            seen.add(name)
            names.append(name)
    return tuple(names)


def _resolve_candidates(
    retrieval: _RetrievalRecord,
    aligned: dict[int, tuple[int, ...]],
    rog_names: dict[int, str],
) -> tuple[CandidateIds, ...]:
    rog_ids_by_name: defaultdict[str, list[int]] = defaultdict(list)
    for rog_entity_id in aligned:
        rog_ids_by_name[rog_names[rog_entity_id]].append(rog_entity_id)
    candidates: list[CandidateIds] = []
    for name in _candidate_names(retrieval):
        rog_ids = tuple(sorted(rog_ids_by_name.get(name, [])))
        mid_ids = tuple(sorted({mid_id for rog_id in rog_ids for mid_id in aligned[rog_id]}))
        if len(mid_ids) == 1:
            resolution: Resolution = "resolved"
        elif mid_ids:
            resolution = "ambiguous"
        else:
            resolution = "unresolved"
        candidates.append(CandidateIds(name, rog_ids, mid_ids, resolution))
    return tuple(candidates)


def _aligned_samples(
    config: InventoryConfig,
    records: dict[int, tuple[_OfficialRecord, ComparisonKind]],
) -> tuple[dict[int, dict[int, tuple[int, ...]]], set[int]]:
    alignments: dict[int, dict[int, tuple[int, ...]]] = {}
    rog_entity_ids: set[int] = set()
    with (
        config.rog_qa.open(encoding="utf-8") as rog_handle,
        config.mid_qa.open(encoding="utf-8") as mid_handle,
    ):
        for sample_index, pair in enumerate(zip_longest(rog_handle, mid_handle)):
            rog_line, mid_line = pair
            if rog_line is None or mid_line is None:
                raise AlignmentMismatchError(sample_index + 1, "文件行数不一致")
            if sample_index not in records:
                continue
            rog_record: _QaRecord = json.loads(rog_line)
            mid_record: _QaRecord = json.loads(mid_line)
            official, _ = records[sample_index]
            if rog_record["id"] != official["ID"] or mid_record["id"] != official["ID"]:
                raise AlignmentMismatchError(sample_index + 1, "official ID 与子图不一致")
            aligned = align_sample_entity_ids(
                rog_record["subgraph"]["tuples"],
                mid_record["subgraph"]["tuples"],
                sample_index + 1,
            )
            alignments[sample_index] = aligned
            rog_entity_ids.update(aligned)
    return alignments, rog_entity_ids


def _summarize(drafts: list[SampleDraft]) -> InventorySummary:
    resolutions = Counter(candidate.resolution for draft in drafts for candidate in draft.candidates)
    unique_mid_ids = {
        mid_id
        for draft in drafts
        for candidate in draft.candidates
        for mid_id in candidate.mid_entity_ids
    }
    chains = {draft.spec.attribute_chain for draft in drafts}
    predicates = {predicate for chain in chains for predicate in chain}
    error_drafts = [draft for draft in drafts if draft.hit_any == 0]
    error_candidates = [candidate for draft in error_drafts for candidate in draft.candidates]
    error_mid_ids = {mid_id for candidate in error_candidates for mid_id in candidate.mid_entity_ids}
    return InventorySummary(
        sample_count=len(drafts),
        comparative_count=sum(draft.kind == "comparative" for draft in drafts),
        superlative_count=sum(draft.kind == "superlative" for draft in drafts),
        hit_any_error_count=len(error_drafts),
        hit_any_error_candidate_count=len(error_candidates),
        hit_any_error_ambiguous_candidate_count=sum(
            candidate.resolution == "ambiguous" for candidate in error_candidates
        ),
        candidate_count=sum(len(draft.candidates) for draft in drafts),
        resolved_candidate_count=resolutions["resolved"],
        ambiguous_candidate_count=resolutions["ambiguous"],
        unresolved_candidate_count=resolutions["unresolved"],
        all_candidates_resolved_sample_count=sum(
            bool(draft.candidates) and all(candidate.resolution == "resolved" for candidate in draft.candidates)
            for draft in drafts
        ),
        hit_any_error_all_candidates_mapped_sample_count=sum(
            bool(draft.candidates) and all(candidate.resolution != "unresolved" for candidate in draft.candidates)
            for draft in error_drafts
        ),
        hit_any_error_all_candidates_resolved_sample_count=sum(
            bool(draft.candidates) and all(candidate.resolution == "resolved" for candidate in draft.candidates)
            for draft in error_drafts
        ),
        unique_mid_count=len(unique_mid_ids),
        unique_hit_any_error_mid_count=len(error_mid_ids),
        unique_attribute_chain_count=len(chains),
        unique_predicate_count=len(predicates),
    )


def build_inventory(config: InventoryConfig) -> InventoryResult:
    official: list[_OfficialRecord] = json.loads(config.official_test.read_text(encoding="utf-8"))
    comparison_records: dict[int, tuple[_OfficialRecord, ComparisonKind]] = {}
    for sample_index, record in enumerate(official):
        kind = _comparison_kind(record["compositionality_type"])
        if kind is not None:
            comparison_records[sample_index] = (record, kind)

    retrieval_by_index = {
        record["sample_index"]: record
        for record in _load_retrieval(config.retrieval)
    }
    hit_any_by_index = {
        record["sample_index"]: record["hit_any"]
        for record in _load_predictions(config.predictions)
    }
    alignments, rog_entity_ids = _aligned_samples(config, comparison_records)
    rog_names = read_selected_vocab(config.rog_entities, rog_entity_ids)
    drafts: list[SampleDraft] = []
    for sample_index, pair in comparison_records.items():
        record, kind = pair
        candidates = _resolve_candidates(
            retrieval_by_index[sample_index],
            alignments[sample_index],
            rog_names,
        )
        drafts.append(
            SampleDraft(
                sample_index,
                record["ID"],
                record["question"],
                kind,
                hit_any_by_index[sample_index],
                extract_comparison_spec(kind, record["sparql"]),
                candidates,
            )
        )
    return InventoryResult(tuple(drafts), _summarize(drafts))
