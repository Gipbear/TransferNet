#!/usr/bin/env python3

"""将 ADInt 转换为实体名称原生的 MetaQA 风格数据布局。"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypedDict

MAX_HOP: Final = 2
DEV_RATIO: Final = 0.1
SPLIT_SEED: Final = 17


class NodeInfo(TypedDict):
    name: str
    description: str
    source: str
    type: str


class TopicEntity(TypedDict):
    topic_name: str
    topic_id: int


class SourceQA(TypedDict):
    question_id: str
    question: str
    answer: dict[str, str]
    topic_entity: TopicEntity | str


class ConvertedQA(TypedDict):
    question_id: str
    question: str
    topic_entity: str
    answers: list[str]
    hop: int | None


@dataclass(frozen=True, slots=True)
class ConversionSummary:
    entities: int
    relations: int
    triples: int
    qa_rows: dict[str, int]
    dropped_rows: dict[str, int]


def _load_pickle(metadata_zip: Path, member: str):
    with zipfile.ZipFile(metadata_zip) as archive:
        return pickle.loads(archive.read(member))


def _clean_text(value: str) -> str:
    return " ".join(value.replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def _load_qa(path: Path) -> list[SourceQA]:
    rows: list[SourceQA] = []
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_entity_info(output_dir: Path, node_info: dict[int, NodeInfo], names: list[str]) -> None:
    with (output_dir / "entity_info.jsonl").open("w", encoding="utf-8") as output:
        for entity_id, name in enumerate(names):
            info = node_info[entity_id]
            record = {
                "name": name,
                "description": _clean_text(info["description"]),
                "source": info["source"],
                "type": _clean_text(info["type"]),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_graph(
    source_dir: Path,
    output_dir: Path,
    names: list[str],
    relations: dict[int, str],
) -> tuple[int, list[list[int]]]:
    adjacency = [[] for _ in names]
    triple_count = 0
    with (
        (source_dir / "ADint_KG.txt").open(encoding="utf-8-sig") as source,
        (output_dir / "kb/kb.tsv").open("w", encoding="utf-8") as output,
    ):
        for line_number, line in enumerate(source, 1):
            parts = line.rstrip().split("\t")
            if len(parts) != 3:
                raise ValueError(f"invalid triple at line {line_number}")
            head, relation, tail = (int(part) for part in parts)
            adjacency[head].append(tail)
            output.write(f"{names[head]}\t{relations[relation]}\t{names[tail]}\n")
            triple_count += 1
    return triple_count, adjacency


def _answer_hops(topic: int, answers: list[int], adjacency: list[list[int]]) -> list[int] | None:
    unresolved = set(answers)
    distances: dict[int, int] = {}
    if topic in unresolved:
        distances[topic] = 0
        unresolved.remove(topic)
    frontier = {topic}
    seen = {topic}
    for hop in range(1, MAX_HOP + 1):
        next_frontier: set[int] = set()
        for entity in frontier:
            next_frontier.update(adjacency[entity])
        frontier = next_frontier - seen
        seen.update(frontier)
        reached = unresolved & frontier
        for answer in reached:
            distances[answer] = hop
        unresolved -= reached
        if not unresolved:
            break
    if unresolved:
        return None
    return [distances[answer] for answer in answers]


def _convert_qa(
    rows: list[SourceQA],
    names: list[str],
    adjacency: list[list[int]],
) -> tuple[list[ConvertedQA], dict[str, int]]:
    converted: list[ConvertedQA] = []
    dropped = {"hop0": 0, "unlinked": 0, "beyond_max_hop": 0}
    for row in rows:
        topic = row["topic_entity"]
        if isinstance(topic, str):
            dropped["unlinked"] += 1
            continue
        topic_id = int(topic["topic_id"])
        answer_ids = [int(answer_id) for answer_id in row["answer"]]
        answer_hops = _answer_hops(topic_id, answer_ids, adjacency)
        if answer_hops is None:
            dropped["beyond_max_hop"] += 1
            continue
        if 0 in answer_hops:
            dropped["hop0"] += 1
            continue
        converted.append(
            {
                "question_id": str(row["question_id"]),
                "question": _clean_text(row["question"]).strip('"'),
                "topic_entity": names[topic_id],
                "answers": [names[answer_id] for answer_id in answer_ids],
                "hop": max(answer_hops),
            }
        )
    return converted, dropped


def _convert_test_qa(
    rows: list[SourceQA],
    names: list[str],
    adjacency: list[list[int]],
) -> list[ConvertedQA]:
    converted: list[ConvertedQA] = []
    for row in rows:
        topic = row["topic_entity"]
        answer_ids = [int(answer_id) for answer_id in row["answer"]]
        topic_name = (
            _clean_text(topic)
            if isinstance(topic, str)
            else names[int(topic["topic_id"])]
        )
        hop: int | None = None
        if not isinstance(topic, str):
            answer_hops = _answer_hops(int(topic["topic_id"]), answer_ids, adjacency)
            if answer_hops is not None:
                hop = max(answer_hops)
        converted.append(
            {
                "question_id": str(row["question_id"]),
                "question": _clean_text(row["question"]).strip('"'),
                "topic_entity": topic_name,
                "answers": [names[answer_id] for answer_id in answer_ids],
                "hop": hop,
            }
        )
    return converted


def _split_train_dev(rows: list[ConvertedQA]) -> tuple[list[ConvertedQA], list[ConvertedQA]]:
    grouped: dict[int, list[ConvertedQA]] = defaultdict(list)
    for row in rows:
        hop = row["hop"]
        if hop is None:
            raise ValueError("training rows must have a reachable hop")
        grouped[hop].append(row)
    dev_ids: set[str] = set()
    rng = random.Random(SPLIT_SEED)
    for hop_rows in grouped.values():
        shuffled = sorted(hop_rows, key=lambda row: row["question_id"])
        rng.shuffle(shuffled)
        dev_size = int(len(shuffled) * DEV_RATIO)
        dev_ids.update(row["question_id"] for row in shuffled[:dev_size])
    train = [row for row in rows if row["question_id"] not in dev_ids]
    dev = [row for row in rows if row["question_id"] in dev_ids]
    return train, dev


def _write_json(path: Path, rows: list[ConvertedQA]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")


def convert_dataset(source_dir: Path, metadata_zip: Path, output_dir: Path) -> ConversionSummary:
    """转换 ADInt 图谱及问答数据。"""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    (output_dir / "kb").mkdir(parents=True)
    node_info: dict[int, NodeInfo] = _load_pickle(metadata_zip, "ADint/node_info.pkl")
    edge_types: dict[int, str] = _load_pickle(metadata_zip, "ADint/edge_type_dict.pkl")
    entity_ids = sorted(node_info)
    if entity_ids != list(range(len(node_info))):
        raise ValueError("ADInt entity IDs must be contiguous from zero")
    names = [_clean_text(node_info[entity_id]["name"]) for entity_id in entity_ids]
    if len(names) != len(set(names)):
        raise ValueError("ADInt entity names must remain unique after whitespace normalization")
    relations = {relation_id: _clean_text(edge_types[relation_id]) for relation_id in sorted(edge_types)}
    _write_entity_info(output_dir, node_info, names)
    triple_count, adjacency = _write_graph(source_dir, output_dir, names, relations)
    source_train = _load_qa(source_dir / "ADint_train_with_topic_entity.json")
    source_test = _load_qa(source_dir / "ADint_test_with_topic_entity.json")
    converted_train, train_dropped = _convert_qa(source_train, names, adjacency)
    converted_test = _convert_test_qa(source_test, names, adjacency)
    train, dev = _split_train_dev(converted_train)
    _write_json(output_dir / "train.json", train)
    _write_json(output_dir / "dev.json", dev)
    _write_json(output_dir / "test.json", converted_test)
    summary = ConversionSummary(
        entities=len(names),
        relations=len(relations),
        triples=triple_count,
        qa_rows={"train": len(train), "dev": len(dev), "test": len(converted_test)},
        dropped_rows=train_dropped,
    )
    manifest = {
        "format": "metaqa-style-name-native-v2",
        "entity_identifier": "canonical ADInt entity name",
        "max_hop": MAX_HOP,
        "dev_split": {"source": "filtered train", "ratio": DEV_RATIO, "seed": SPLIT_SEED, "stratify": "hop"},
        "train_filters": [
            "drop unlinked topic entities",
            "drop hop-0 leakage",
            "require every answer within max_hop",
        ],
        "test_policy": "preserve every source row; unresolved topic or answer reachability uses hop=null",
        "source_dir": str(source_dir.resolve()),
        "metadata_zip": str(metadata_zip.resolve()),
        "summary": asdict(summary),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ADInt to a name-native two-hop layout.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--metadata-zip", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data/input/ADInt"), type=Path)
    args = parser.parse_args()
    summary = convert_dataset(args.source_dir, args.metadata_zip, args.output_dir)
    print(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
