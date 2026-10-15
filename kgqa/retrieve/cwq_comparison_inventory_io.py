"""写出 CWQ 比较题候选 MID 与谓词清单。"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import TypedDict

from kgqa.retrieve.cwq_comparison_inventory import (
    InventoryConfig,
    InventoryResult,
    SampleDraft,
    read_selected_vocab,
)


class _CandidateOutput(TypedDict):
    name: str
    rog_entity_ids: list[int]
    mid_entity_ids: list[int]
    mids: list[str]
    resolution: str


class _SampleOutput(TypedDict):
    sample_index: int
    id: str
    question: str
    compositionality_type: str
    hit_any: int
    operator: str
    threshold: str | None
    threshold_datatype: str | None
    attribute_chain: list[str]
    candidates: list[_CandidateOutput]


class _ChainOutput(TypedDict):
    attribute_chain: list[str]
    comparative_count: int
    superlative_count: int
    total_count: int


class _PredicateInventory(TypedDict):
    operator_counts: dict[str, int]
    chains: list[_ChainOutput]
    first_hop_predicates: list[str]
    terminal_value_predicates: list[str]
    all_predicates: list[str]


def _serialize_sample(draft: SampleDraft, mid_names: dict[int, str]) -> _SampleOutput:
    candidates: list[_CandidateOutput] = []
    for candidate in draft.candidates:
        candidates.append({
            "name": candidate.name,
            "rog_entity_ids": list(candidate.rog_entity_ids),
            "mid_entity_ids": list(candidate.mid_entity_ids),
            "mids": [mid_names[mid_id] for mid_id in candidate.mid_entity_ids],
            "resolution": candidate.resolution,
        })
    return {
        "sample_index": draft.sample_index,
        "id": draft.sample_id,
        "question": draft.question,
        "compositionality_type": draft.kind,
        "hit_any": draft.hit_any,
        "operator": draft.spec.operator,
        "threshold": draft.spec.threshold,
        "threshold_datatype": draft.spec.threshold_datatype,
        "attribute_chain": list(draft.spec.attribute_chain),
        "candidates": candidates,
    }


def _build_predicate_inventory(drafts: tuple[SampleDraft, ...]) -> _PredicateInventory:
    chain_counts: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    operators: Counter[str] = Counter()
    for draft in drafts:
        chain_counts[draft.spec.attribute_chain][draft.kind] += 1
        operators[draft.spec.operator] += 1
    chains: list[_ChainOutput] = []
    for chain, counts in sorted(chain_counts.items(), key=lambda item: (-sum(item[1].values()), item[0])):
        chains.append({
            "attribute_chain": list(chain),
            "comparative_count": counts["comparative"],
            "superlative_count": counts["superlative"],
            "total_count": sum(counts.values()),
        })
    return {
        "operator_counts": dict(sorted(operators.items())),
        "chains": chains,
        "first_hop_predicates": sorted({chain[0] for chain in chain_counts}),
        "terminal_value_predicates": sorted({chain[-1] for chain in chain_counts}),
        "all_predicates": sorted({predicate for chain in chain_counts for predicate in chain}),
    }


def _write_samples(config: InventoryConfig, result: InventoryResult, mid_names: dict[int, str]) -> None:
    with (config.output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for draft in result.drafts:
            record = _serialize_sample(draft, mid_names)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_error_samples(config: InventoryConfig, result: InventoryResult, mid_names: dict[int, str]) -> None:
    extraction_path = config.output_dir / "error_samples_for_literal_extraction.jsonl"
    resolved_path = config.output_dir / "error_samples_unique_resolution.jsonl"
    with (
        extraction_path.open("w", encoding="utf-8") as extraction_handle,
        resolved_path.open("w", encoding="utf-8") as resolved_handle,
    ):
        for draft in result.drafts:
            if draft.hit_any != 0 or not draft.candidates:
                continue
            if any(candidate.resolution == "unresolved" for candidate in draft.candidates):
                continue
            line = json.dumps(_serialize_sample(draft, mid_names), ensure_ascii=False) + "\n"
            extraction_handle.write(line)
            if all(candidate.resolution == "resolved" for candidate in draft.candidates):
                resolved_handle.write(line)


def write_inventory(config: InventoryConfig, result: InventoryResult) -> None:
    mid_ids = {
        mid_id
        for draft in result.drafts
        for candidate in draft.candidates
        for mid_id in candidate.mid_entity_ids
    }
    mid_names = read_selected_vocab(config.mid_entities, mid_ids)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_samples(config, result, mid_names)
    _write_error_samples(config, result, mid_names)
    mids = sorted(mid_names.values())
    (config.output_dir / "candidate_mids.txt").write_text("\n".join(mids) + "\n", encoding="utf-8")
    error_mid_ids = {
        mid_id
        for draft in result.drafts
        if draft.hit_any == 0
        for candidate in draft.candidates
        for mid_id in candidate.mid_entity_ids
    }
    error_mids = sorted(mid_names[mid_id] for mid_id in error_mid_ids)
    (config.output_dir / "error_candidate_mids.txt").write_text(
        "\n".join(error_mids) + "\n",
        encoding="utf-8",
    )
    inventory = _build_predicate_inventory(result.drafts)
    (config.output_dir / "predicate_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (config.output_dir / "summary.json").write_text(
        json.dumps(asdict(result.summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
