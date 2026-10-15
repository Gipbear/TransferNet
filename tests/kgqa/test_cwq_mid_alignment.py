"""RoG-CWQ 与 MID 版 CWQ 子图对齐测试。"""
from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from typing import TypedDict

import numpy as np

from kgqa.retrieve.cwq_mid_alignment import (
    AlignmentMismatchError,
    build_mid_mapping,
    write_mid_mapping,
)


class _SubgraphRecord(TypedDict):
    tuples: list[list[int]]
    entities: list[int]


class _QaRecord(TypedDict):
    id: str
    question: str
    entities: list[int]
    answers: list[str]
    subgraph: _SubgraphRecord


def _write_jsonl(path: Path, records: list[_QaRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _record(sample_id: str, question: str, tuples: list[list[int]]) -> _QaRecord:
    return {
        "id": sample_id,
        "question": question,
        "entities": [],
        "answers": [],
        "subgraph": {"tuples": tuples, "entities": []},
    }


class TestCwqMidAlignment(unittest.TestCase):
    def test_build_marks_conflicting_mid_assignments_as_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rog_path = root / "rog.jsonl"
            mid_path = root / "mid.jsonl"
            _write_jsonl(rog_path, [_record("q1", "question", [[0, 7, 1], [0, 7, 1]])])
            _write_jsonl(mid_path, [_record("q1", "question", [[10, 7, 11], [10, 7, 12]])])

            mapping = build_mid_mapping(rog_path, mid_path, rog_entity_count=3)

        self.assertEqual(mapping.mid_by_rog.tolist(), [10, -2, -1])
        self.assertEqual(mapping.evidence_count.tolist(), [2, 2, 0])
        self.assertEqual(mapping.ambiguous, {1: {11: 1, 12: 1}})
        self.assertEqual(mapping.stats.unique_mapping_count, 1)
        self.assertEqual(mapping.stats.ambiguous_mapping_count, 1)

    def test_build_rejects_relation_sequence_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rog_path = root / "rog.jsonl"
            mid_path = root / "mid.jsonl"
            _write_jsonl(rog_path, [_record("q1", "question", [[0, 7, 1]])])
            _write_jsonl(mid_path, [_record("q1", "question", [[10, 8, 11]])])

            with self.assertRaises(AlignmentMismatchError):
                build_mid_mapping(rog_path, mid_path, rog_entity_count=2)

    def test_write_materializes_reusable_mapping_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rog_path = root / "rog.jsonl"
            mid_path = root / "mid.jsonl"
            output_dir = root / "output"
            _write_jsonl(rog_path, [_record("q1", "question", [[0, 7, 1], [0, 7, 1]])])
            _write_jsonl(mid_path, [_record("q1", "question", [[10, 7, 11], [10, 7, 12]])])
            mapping = build_mid_mapping(rog_path, mid_path, rog_entity_count=3)

            write_mid_mapping(mapping, output_dir)

            with np.load(output_dir / "mid_mapping.npz") as arrays:
                self.assertEqual(arrays["mid_by_rog"].tolist(), [10, -2, -1])
            with gzip.open(output_dir / "ambiguous_mappings.jsonl.gz", "rt", encoding="utf-8") as handle:
                ambiguous = json.loads(handle.readline())
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(ambiguous["rog_entity_id"], 1)
        self.assertEqual(summary["ambiguous_mapping_count"], 1)


if __name__ == "__main__":
    unittest.main()
