"""统计 CWQ 训练属性链对测试比较题的覆盖率。"""
from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict, assert_never

from kgqa.retrieve.cwq_comparison import extract_comparison_spec

type AttributeChain = tuple[str, ...]


class OfficialRecord(TypedDict):
    compositionality_type: Literal["comparative", "superlative", "composition", "conjunction"]
    sparql: str


class ComparisonRecord(TypedDict):
    compositionality_type: str
    operator: str
    attribute_chain: list[str]


@dataclass(frozen=True, slots=True)
class ChainFrequency:
    chain: AttributeChain
    count: int


@dataclass(frozen=True, slots=True)
class OfficialChainInventory:
    sample_count: int
    comparison_sample_count: int
    chains: tuple[ChainFrequency, ...]


@dataclass(frozen=True, slots=True)
class CoverageSlice:
    sample_count: int
    covered_sample_count: int
    unique_chain_count: int
    covered_unique_chain_count: int


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    overall: CoverageSlice
    comparative: CoverageSlice
    superlative: CoverageSlice


@dataclass(frozen=True, slots=True)
class CoverageAudit:
    train: OfficialChainInventory
    dev: OfficialChainInventory
    train_only: CoverageSummary
    train_dev: CoverageSummary
    unseen_train_dev: tuple[ChainFrequency, ...]


def build_coverage_audit(
    train_records: Sequence[OfficialRecord],
    dev_records: Sequence[OfficialRecord],
    test_samples: Sequence[ComparisonRecord],
) -> CoverageAudit:
    train = build_official_chain_inventory(train_records)
    dev = build_official_chain_inventory(dev_records)
    train_chains = {item.chain for item in train.chains}
    train_dev_chains = train_chains | {item.chain for item in dev.chains}
    return CoverageAudit(
        train,
        dev,
        summarize_chain_coverage(train_chains, test_samples),
        summarize_chain_coverage(train_dev_chains, test_samples),
        list_unseen_chains(train_dev_chains, test_samples),
    )


def build_official_chain_inventory(records: Sequence[OfficialRecord]) -> OfficialChainInventory:
    counts: Counter[AttributeChain] = Counter()
    comparison_sample_count = 0
    for record in records:
        match record["compositionality_type"]:
            case "comparative" | "superlative" as kind:
                spec = extract_comparison_spec(kind, record["sparql"])
                counts[spec.attribute_chain] += 1
                comparison_sample_count += 1
            case "composition" | "conjunction":
                continue
            case unreachable:
                assert_never(unreachable)
    return OfficialChainInventory(
        len(records),
        comparison_sample_count,
        _sorted_chain_frequencies(counts),
    )


def summarize_chain_coverage(
    known_chains: Collection[AttributeChain],
    samples: Sequence[ComparisonRecord],
) -> CoverageSummary:
    known = set(known_chains)
    comparative = [sample for sample in samples if sample["compositionality_type"] == "comparative"]
    superlative = [sample for sample in samples if sample["compositionality_type"] == "superlative"]
    return CoverageSummary(
        _summarize_slice(known, samples),
        _summarize_slice(known, comparative),
        _summarize_slice(known, superlative),
    )


def list_unseen_chains(
    known_chains: Collection[AttributeChain],
    samples: Sequence[ComparisonRecord],
) -> tuple[ChainFrequency, ...]:
    known = set(known_chains)
    counts = Counter(
        tuple(sample["attribute_chain"])
        for sample in samples
        if tuple(sample["attribute_chain"]) not in known
    )
    return _sorted_chain_frequencies(counts)


def _summarize_slice(known_chains: set[AttributeChain], samples: Sequence[ComparisonRecord]) -> CoverageSlice:
    chains = [tuple(sample["attribute_chain"]) for sample in samples]
    unique_chains = set(chains)
    return CoverageSlice(
        len(samples),
        sum(chain in known_chains for chain in chains),
        len(unique_chains),
        len(unique_chains & known_chains),
    )


def _sorted_chain_frequencies(counts: Counter[AttributeChain]) -> tuple[ChainFrequency, ...]:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(ChainFrequency(chain, count) for chain, count in ordered)
