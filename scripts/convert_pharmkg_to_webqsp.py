#!/usr/bin/env python3

"""将 RiTeK pharmKG 转换为 WebQSP 风格的全局图和 QA 文件。"""

from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

SPLITS = ("train", "dev", "test")
QA_SOURCE_NAMES = {
    "train": "pharmKG_train_QA_with_topic_entity.json",
    "dev": "pharmKG_dev_with_topic_entity.json",
    "test": "pharmKG_test_QA_with_topic_entity.json",
}


@dataclass(frozen=True, slots=True)
class ConversionSummary:
    entities: int
    relations: int
    triples: int
    qa_rows: dict[str, int]


class NodeInfo(TypedDict):
    name: str
    description: str | None
    source: str
    type: str


def _load_pickle(metadata_zip: Path, member: str):
    with zipfile.ZipFile(metadata_zip) as archive:
        return pickle.loads(archive.read(member))


def _entity_token(entity_id: int | str) -> str:
    return f"p.{entity_id}"


def _normalize_type(value: str) -> str:
    normalized = value.strip().lower()
    return normalized or "unknown"


def _clean_question(value: str) -> str:
    question = " ".join(value.replace("\r", " ").replace("\n", " ").split())
    if len(question) >= 2 and question[0] == question[-1] == '"':
        return question[1:-1].strip()
    return question


def _clean_relation(value: str) -> str:
    return " ".join(value.replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def _write_entities(output_dir: Path, node_info: dict[int, NodeInfo]) -> None:
    entity_ids = sorted(node_info)
    if entity_ids != list(range(len(node_info))):
        raise ValueError("pharmKG entity IDs must be contiguous from zero")

    fb_dir = output_dir / "fbwq_full"
    entity_names: dict[str, str] = {}
    with (
        (fb_dir / "entities.dict").open("w", encoding="utf-8") as entities,
        (fb_dir / "mid2name.txt").open("w", encoding="utf-8") as names,
        (fb_dir / "entity_info.jsonl").open("w", encoding="utf-8") as info_output,
    ):
        for entity_id in entity_ids:
            info = node_info[entity_id]
            token = _entity_token(entity_id)
            name = " ".join(info["name"].split())
            entity_names[token] = name
            entities.write(f"{token}\t{entity_id}\n")
            names.write(f"{token}\t{name}\n")
            record = {
                "id": entity_id,
                "token": token,
                "name": name,
                "description": " ".join((info["description"] or "").split()),
                "source": info["source"],
                "type": _normalize_type(info["type"]),
                "type_raw": info["type"],
            }
            info_output.write(json.dumps(record, ensure_ascii=False) + "\n")
    (fb_dir / "entities_names.json").write_text(
        json.dumps(entity_names, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def _write_relations(output_dir: Path, edge_types: dict[int, str]) -> dict[int, str]:
    relation_ids = sorted(edge_types)
    if relation_ids != list(range(len(edge_types))):
        raise ValueError("pharmKG relation IDs must be contiguous from zero")
    relations = {relation_id: _clean_relation(edge_types[relation_id]) for relation_id in relation_ids}
    if len(set(relations.values())) != len(relations):
        raise ValueError("pharmKG relation names must be unique")

    fb_dir = output_dir / "fbwq_full"
    glosses: dict[str, str] = {}
    with (fb_dir / "relations.dict").open("w", encoding="utf-8") as output:
        for relation_id, relation in relations.items():
            reverse = f"{relation}_reverse"
            output.write(f"{relation}\t{2 * relation_id}\n")
            output.write(f"{reverse}\t{2 * relation_id + 1}\n")
            glosses[relation] = relation
            glosses[reverse] = f"reverse of {relation}"
    (fb_dir / "relation_gloss_positive.json").write_text(
        json.dumps(glosses, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return relations


def _write_graph(source_dir: Path, output_dir: Path, relations: dict[int, str]) -> int:
    triples = 0
    source_path = source_dir / "Pharm_KG.txt"
    output_path = output_dir / "fbwq_full/train.txt"
    with source_path.open(encoding="utf-8-sig") as source, output_path.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(source, 1):
            parts = line.rstrip().split("\t")
            if len(parts) != 3:
                raise ValueError(f"invalid triple at {source_path}:{line_number}")
            head, relation_id_text, tail = parts
            relation = relations[int(relation_id_text)]
            output.write(f"{_entity_token(head)}\t{relation}\t{_entity_token(tail)}\n")
            triples += 1
    return triples


def _write_qa(source_dir: Path, output_dir: Path, split: str) -> int:
    source_path = source_dir / QA_SOURCE_NAMES[split]
    output_path = output_dir / f"QA_data/PharmKG/qa_{split}_pharmkg.txt"
    rows = 0
    with source_path.open(encoding="utf-8-sig") as source, output_path.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = _clean_question(row["question"])
            topic = _entity_token(row["topic_entity"]["topic_id"])
            answers = list(dict.fromkeys(_entity_token(entity_id) for entity_id in row["answer"]))
            if not answers:
                raise ValueError(f"missing answers at {source_path}:{line_number}")
            output.write(f"{question} [{topic}]\t{'|'.join(answers)}\n")
            rows += 1
    return rows


def convert_dataset(source_dir: Path, metadata_zip: Path, output_dir: Path) -> ConversionSummary:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    (output_dir / "fbwq_full").mkdir(parents=True)
    (output_dir / "QA_data/PharmKG").mkdir(parents=True)

    node_info = _load_pickle(metadata_zip, "pharmKG/node_info.pkl")
    edge_types = _load_pickle(metadata_zip, "pharmKG/edge_type_dict.pkl")
    _write_entities(output_dir, node_info)
    relations = _write_relations(output_dir, edge_types)
    triples = _write_graph(source_dir, output_dir, relations)
    qa_rows = {split: _write_qa(source_dir, output_dir, split) for split in SPLITS}
    summary = ConversionSummary(
        entities=len(node_info), relations=len(relations), triples=triples, qa_rows=qa_rows
    )
    manifest = {
        "format": "webqsp-compatible-global-kg-v1",
        "entity_token": "p.<original_pharmkg_entity_id>",
        "source_dir": str(source_dir.resolve()),
        "metadata_zip": str(metadata_zip.resolve()),
        "summary": asdict(summary),
        "notes": [
            "fbwq_full/train.txt contains the original directed pharmKG edges.",
            "relations.dict defines forward and _reverse relation IDs for GlobalKG reconstruction.",
            "hop1_neighbors is omitted because it stores zero-based source triple row indices.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RiTeK pharmKG to the WebQSP-style local layout.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--metadata-zip", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data/input/PharmKG"), type=Path)
    args = parser.parse_args()
    summary = convert_dataset(args.source_dir, args.metadata_zip, args.output_dir)
    print(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
