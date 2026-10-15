"""对齐 RoG-CWQ 与 MID 版 CWQ 子图实体。"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path
from typing import NamedTuple, TypedDict

import numpy as np
from numpy.typing import NDArray

UNSEEN_MID = -1
AMBIGUOUS_MID = -2


class AlignmentMismatchError(Exception):
    def __init__(self, row_index: int, reason: str) -> None:
        self.row_index = row_index
        self.reason = reason
        super().__init__(f"第 {row_index} 行无法对齐: {reason}")


@dataclass(frozen=True, slots=True)
class AlignmentStats:
    row_count: int
    endpoint_observation_count: int
    observed_rog_entity_count: int
    unique_mapping_count: int
    ambiguous_mapping_count: int
    unseen_rog_entity_count: int
    rog_qa_sha256: str
    mid_qa_sha256: str


@dataclass(frozen=True, slots=True)
class MidMapping:
    mid_by_rog: NDArray[np.int64]
    evidence_count: NDArray[np.uint32]
    ambiguous: dict[int, dict[int, int]]
    stats: AlignmentStats


class _SubgraphRecord(TypedDict):
    tuples: list[list[int]]


class _QaRecord(TypedDict):
    id: str
    question: str
    subgraph: _SubgraphRecord


class _MappingState(NamedTuple):
    mid_by_rog: NDArray[np.int64]
    evidence_count: NDArray[np.uint32]
    ambiguous: dict[int, dict[int, int]]


def _observe_mapping(
    rog_entity_id: int,
    mid_entity_id: int,
    state: _MappingState,
) -> None:
    if rog_entity_id < 0 or rog_entity_id >= len(state.mid_by_rog):
        raise AlignmentMismatchError(-1, f"RoG entity id 越界: {rog_entity_id}")

    state.evidence_count[rog_entity_id] += 1
    candidates = state.ambiguous.get(rog_entity_id)
    if candidates is not None:
        candidates[mid_entity_id] = candidates.get(mid_entity_id, 0) + 1
        return

    current_mid = int(state.mid_by_rog[rog_entity_id])
    if current_mid == UNSEEN_MID:
        state.mid_by_rog[rog_entity_id] = mid_entity_id
        return
    if current_mid == mid_entity_id:
        return

    prior_count = int(state.evidence_count[rog_entity_id]) - 1
    state.ambiguous[rog_entity_id] = {current_mid: prior_count, mid_entity_id: 1}
    state.mid_by_rog[rog_entity_id] = AMBIGUOUS_MID


def _aligned_triples(
    rog_record: _QaRecord,
    mid_record: _QaRecord,
    row_index: int,
) -> tuple[list[list[int]], list[list[int]]]:
    if rog_record["id"] != mid_record["id"]:
        raise AlignmentMismatchError(row_index, "sample id 不一致")
    if rog_record["question"] != mid_record["question"]:
        raise AlignmentMismatchError(row_index, "question 不一致")

    rog_triples = rog_record["subgraph"]["tuples"]
    mid_triples = mid_record["subgraph"]["tuples"]
    if len(rog_triples) != len(mid_triples):
        raise AlignmentMismatchError(row_index, "三元组数量不一致")
    if any(rog[1] != mid[1] for rog, mid in zip(rog_triples, mid_triples)):
        raise AlignmentMismatchError(row_index, "关系序列不一致")
    return rog_triples, mid_triples


def build_mid_mapping(rog_qa_path: Path, mid_qa_path: Path, rog_entity_count: int) -> MidMapping:
    mid_by_rog = np.full(rog_entity_count, UNSEEN_MID, dtype=np.int64)
    evidence_count = np.zeros(rog_entity_count, dtype=np.uint32)
    ambiguous: dict[int, dict[int, int]] = {}
    state = _MappingState(mid_by_rog, evidence_count, ambiguous)
    rog_hash = hashlib.sha256()
    mid_hash = hashlib.sha256()
    row_count = 0

    with rog_qa_path.open("rb") as rog_handle, mid_qa_path.open("rb") as mid_handle:
        for row_index, pair in enumerate(zip_longest(rog_handle, mid_handle), start=1):
            rog_line, mid_line = pair
            if rog_line is None or mid_line is None:
                raise AlignmentMismatchError(row_index, "文件行数不一致")
            rog_hash.update(rog_line)
            mid_hash.update(mid_line)
            rog_record = json.loads(rog_line)
            mid_record = json.loads(mid_line)
            rog_triples, mid_triples = _aligned_triples(rog_record, mid_record, row_index)
            for rog, mid in zip(rog_triples, mid_triples):
                _observe_mapping(rog[0], mid[0], state)
                _observe_mapping(rog[2], mid[2], state)
            row_count = row_index

    unique_mapping_count = int(np.count_nonzero(mid_by_rog >= 0))
    ambiguous_mapping_count = len(ambiguous)
    observed_count = unique_mapping_count + ambiguous_mapping_count
    stats = AlignmentStats(
        row_count=row_count,
        endpoint_observation_count=int(evidence_count.sum()),
        observed_rog_entity_count=observed_count,
        unique_mapping_count=unique_mapping_count,
        ambiguous_mapping_count=ambiguous_mapping_count,
        unseen_rog_entity_count=rog_entity_count - observed_count,
        rog_qa_sha256=rog_hash.hexdigest(),
        mid_qa_sha256=mid_hash.hexdigest(),
    )
    return MidMapping(mid_by_rog, evidence_count, ambiguous, stats)


def write_mid_mapping(mapping: MidMapping, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "mid_mapping.npz",
        mid_by_rog=mapping.mid_by_rog,
        evidence_count=mapping.evidence_count,
    )
    with gzip.open(output_dir / "ambiguous_mappings.jsonl.gz", "wt", encoding="utf-8") as handle:
        for rog_entity_id, candidates in sorted(mapping.ambiguous.items()):
            record = {
                "rog_entity_id": rog_entity_id,
                "candidates": [
                    {"mid_entity_id": mid_entity_id, "evidence_count": count}
                    for mid_entity_id, count in sorted(candidates.items())
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = json.dumps(asdict(mapping.stats), ensure_ascii=False, indent=2, sort_keys=True)
    (output_dir / "summary.json").write_text(summary + "\n", encoding="utf-8")
