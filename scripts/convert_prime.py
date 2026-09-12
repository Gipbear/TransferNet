#!/usr/bin/env python3

"""把本地 Hugging Face Prime 子集转换为 TransferNet 原生 ID 布局。"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypedDict

import torch

from Prime.graph import build_adjacency, target_distances

MAX_HOP: Final = 3
DEV_RATIO: Final = 0.1
SPLIT_SEED: Final = 17


class TopicEntity(TypedDict):
    topic_name: str
    topic_id: str


class SourceQA(TypedDict):
    question_id: str
    question: str
    answer: dict[str, str]
    topic_entity: list[TopicEntity]


class ConvertedQA(TypedDict):
    question_id: str
    question: str
    topic_entity: str
    answers: list[str]
    hop: int


@dataclass(frozen=True, slots=True)
class ConversionSummary:
    entities: int
    relations: int
    triples: int
    qa_rows: dict[str, int]
    dropped_rows: dict[str, int]


@dataclass(frozen=True, slots=True)
class NonContiguousEntityIdsError(ValueError):
    entity_count: int

    def __str__(self) -> str:
        return f"Prime entity IDs must be contiguous from zero; found {self.entity_count} entities"


def _load_pickle(archive: zipfile.ZipFile, member: str):
    return pickle.loads(archive.read(member))


def _load_qa(path: Path) -> list[SourceQA]:
    with path.open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def _relation_ids(raw: dict) -> dict[str, int]:
    first_key = next(iter(raw))
    if isinstance(first_key, str):
        return {str(name): int(relation_id) for name, relation_id in raw.items()}
    return {str(name): int(relation_id) for relation_id, name in raw.items()}


def _copy_graph(archive: zipfile.ZipFile, output_dir: Path) -> None:
    graph_dir = output_dir / "graph"
    graph_dir.mkdir(parents=True)
    for filename in ("edge_index.pt", "edge_types.pt", "node_types.pt"):
        with archive.open(f"prime/{filename}") as source, (graph_dir / filename).open("wb") as output:
            shutil.copyfileobj(source, output)


def _convert_rows(rows: list[SourceQA], adjacency) -> tuple[list[ConvertedQA], dict[str, int]]:
    grouped: dict[int, list[SourceQA]] = defaultdict(list)
    dropped = {"unlinked": 0, "hop0": 0, "beyond_max_hop": 0}
    for row in rows:
        if len(row["topic_entity"]) != 1:
            dropped["unlinked"] += 1
            continue
        grouped[int(row["topic_entity"][0]["topic_id"])].append(row)
    converted: list[ConvertedQA] = []
    for topic_id, topic_rows in grouped.items():
        targets = {int(answer) for row in topic_rows for answer in row["answer"]}
        distances = target_distances(adjacency, topic_id, targets, MAX_HOP)
        for row in topic_rows:
            answer_ids = [int(answer) for answer in row["answer"]]
            hops = [distances.get(answer) for answer in answer_ids]
            if any(hop is None for hop in hops):
                dropped["beyond_max_hop"] += 1
                continue
            if any(hop == 0 for hop in hops):
                dropped["hop0"] += 1
                continue
            converted.append(
                {
                    "question_id": str(row["question_id"]),
                    "question": " ".join(row["question"].split()),
                    "topic_entity": str(topic_id),
                    "answers": [str(answer) for answer in answer_ids],
                    "hop": max(int(hop) for hop in hops if hop is not None),
                }
            )
    converted.sort(key=lambda row: row["question_id"])
    return converted, dropped


def _split_train_dev(rows: list[ConvertedQA]) -> tuple[list[ConvertedQA], list[ConvertedQA]]:
    grouped: dict[int, list[ConvertedQA]] = defaultdict(list)
    for row in rows:
        grouped[row["hop"]].append(row)
    rng = random.Random(SPLIT_SEED)
    dev_ids: set[str] = set()
    for hop_rows in grouped.values():
        shuffled = sorted(hop_rows, key=lambda row: row["question_id"])
        rng.shuffle(shuffled)
        dev_ids.update(row["question_id"] for row in shuffled[:int(len(shuffled) * DEV_RATIO)])
    train = [row for row in rows if row["question_id"] not in dev_ids]
    dev = [row for row in rows if row["question_id"] in dev_ids]
    return train, dev


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def convert_dataset(source_dir: Path, metadata_zip: Path, output_dir: Path) -> ConversionSummary:
    """转换 Prime 图、实体元数据和问答划分。"""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    with zipfile.ZipFile(metadata_zip) as archive:
        _copy_graph(archive, output_dir)
        nodes = _load_pickle(archive, "prime/node_info.pkl")
        relations = _relation_ids(_load_pickle(archive, "prime/edge_type_dict.pkl"))
    entity_ids = sorted(nodes)
    if entity_ids != list(range(len(nodes))):
        raise NonContiguousEntityIdsError(len(nodes))
    names = [str(nodes[entity_id]["name"]) for entity_id in entity_ids]
    _write_json(output_dir / "entity_names.json", names)
    _write_json(output_dir / "graph/edge_type_dict.json", relations)
    edge_index = torch.load(output_dir / "graph/edge_index.pt", weights_only=True).long()
    adjacency = build_adjacency(edge_index, len(names))
    train_rows, train_dropped = _convert_rows(
        _load_qa(source_dir / "Prime_train_QA_with_topic_entity.json"), adjacency
    )
    test_rows, test_dropped = _convert_rows(
        _load_qa(source_dir / "Prime_test_QA_with_topic_entity.json"), adjacency
    )
    train, dev = _split_train_dev(train_rows)
    for split, rows in (("train", train), ("dev", dev), ("test", test_rows)):
        _write_json(output_dir / f"{split}.json", rows)
    dropped = {key: train_dropped[key] + test_dropped[key] for key in train_dropped}
    summary = ConversionSummary(
        entities=len(names),
        relations=len(relations),
        triples=edge_index.shape[1],
        qa_rows={"train": len(train), "dev": len(dev), "test": len(test_rows)},
        dropped_rows=dropped,
    )
    manifest = {
        "format": "prime-native-id-v1",
        "entity_identifier": "original Prime integer ID serialized as string",
        "max_hop": MAX_HOP,
        "graph_direction": "preserved exactly; no synthetic reverse edges",
        "dev_split": {"source": "filtered train", "ratio": DEV_RATIO, "seed": SPLIT_SEED, "stratify": "hop"},
        "summary": asdict(summary),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Prime for three-hop TransferNet training.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--metadata-zip", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data/input/Prime"), type=Path)
    args = parser.parse_args()
    print(json.dumps(asdict(convert_dataset(args.source_dir, args.metadata_zip, args.output_dir)), sort_keys=True))


if __name__ == "__main__":
    main()
