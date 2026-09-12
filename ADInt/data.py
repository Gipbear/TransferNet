"""加载名称原生的 ADInt 图谱和问答数据。"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from transformers import AutoTokenizer

from utils.huggingface import from_pretrained_local_first
from utils.misc import invert_dict


class QARecord(TypedDict):
    question_id: str
    question: str
    topic_entity: str
    answers: list[str]
    hop: int | None


@dataclass(frozen=True, slots=True)
class GraphData:
    ent2id: dict[str, int]
    rel2id: dict[str, int]
    triples: torch.Tensor
    adjacency: list[list[int]]


@dataclass(frozen=True, slots=True)
class LoaderSpec:
    input_dir: Path
    split: str
    bert_name: str
    batch_size: int
    training: bool = False
    qa_file: Path | None = None
    limit: int = 0


EncodedRow = tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]


def _index(mapping: dict[str, int], value: str) -> int:
    if value not in mapping:
        mapping[value] = len(mapping)
    return mapping[value]


def load_graph(input_dir: str) -> GraphData:
    ent2id: dict[str, int] = {}
    rel2id: dict[str, int] = {}
    triples: list[tuple[int, int, int]] = []
    graph_path = Path(input_dir) / "kb/kb.tsv"
    with graph_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"invalid ADInt triple at {graph_path}:{line_number}")
            subject, relation, obj = parts
            subject_id = _index(ent2id, subject)
            object_id = _index(ent2id, obj)
            relation_id = _index(rel2id, relation)
            reverse_id = _index(rel2id, f"{relation}_reverse")
            triples.append((subject_id, relation_id, object_id))
            triples.append((object_id, reverse_id, subject_id))
    triple_tensor = torch.tensor(triples, dtype=torch.long)
    adjacency = [[] for _ in ent2id]
    for subject, _, obj in triples:
        adjacency[subject].append(obj)
    return GraphData(ent2id=ent2id, rel2id=rel2id, triples=triple_tensor, adjacency=adjacency)


def _entity_range(topic_id: int, adjacency: list[list[int]]) -> torch.Tensor:
    candidates = set(adjacency[topic_id])
    for entity_id in tuple(candidates):
        candidates.update(adjacency[entity_id])
    candidates.discard(topic_id)
    return torch.tensor(sorted(candidates), dtype=torch.long)


def _topic_tensors(topic: str, graph: GraphData) -> tuple[torch.Tensor, torch.Tensor]:
    topic_id = graph.ent2id.get(topic)
    if topic_id is None:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty.clone()
    return torch.tensor([topic_id], dtype=torch.long), _entity_range(topic_id, graph.adjacency)


def _load_rows(path: Path) -> list[QARecord]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _tokenize_rows(rows: list[QARecord], tokenizer) -> list[dict[str, torch.Tensor]]:
    questions = [row["question"] for row in rows]
    tokenized: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(questions), 512):
        encoded = tokenizer(
            questions[start:start + 512],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch_size = encoded["input_ids"].shape[0]
        for offset in range(batch_size):
            tokenized.append({key: value[offset:offset + 1] for key, value in encoded.items()})
    return tokenized


def collate(batch: list[EncodedRow]):
    columns = list(zip(*batch))
    topics, questions, answers, entity_ranges = columns
    question_batch = {key: torch.cat([question[key] for question in questions]) for key in questions[0]}
    return list(topics), question_batch, list(answers), list(entity_ranges)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[EncodedRow]):
        self.rows = rows

    def __getitem__(self, index: int) -> EncodedRow:
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, spec: LoaderSpec, graph: GraphData):
        rows = _load_rows(spec.qa_file or spec.input_dir / f"{spec.split}.json")
        if spec.limit:
            rows = rows[:spec.limit]
        self.tokenizer = from_pretrained_local_first(AutoTokenizer, spec.bert_name)
        encoded_questions = _tokenize_rows(rows, self.tokenizer)
        encoded_rows: list[EncodedRow] = []
        for row, question in zip(rows, encoded_questions):
            topic_ids, entity_range = _topic_tensors(row["topic_entity"], graph)
            answer_ids = torch.tensor([graph.ent2id[answer] for answer in row["answers"]], dtype=torch.long)
            encoded_rows.append(
                (
                    topic_ids,
                    question,
                    answer_ids,
                    entity_range,
                )
            )
        self.ent2id = graph.ent2id
        self.rel2id = graph.rel2id
        self.id2ent = invert_dict(graph.ent2id)
        self.id2rel = invert_dict(graph.rel2id)
        self.qa_text = [row["question"] for row in rows]
        self.hops = [row["hop"] for row in rows]
        super().__init__(
            Dataset(encoded_rows),
            batch_size=spec.batch_size,
            shuffle=spec.training,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
        )


def load_data(input_dir: str, bert_name: str, batch_size: int):
    """加载 ADInt 的图谱及训练、验证、测试集合。"""
    graph = load_graph(input_dir)
    root = Path(input_dir)
    train_loader = DataLoader(
        LoaderSpec(root, "train", bert_name, batch_size, training=True),
        graph,
    )
    dev_loader = DataLoader(LoaderSpec(root, "dev", bert_name, batch_size), graph)
    test_loader = DataLoader(LoaderSpec(root, "test", bert_name, batch_size), graph)
    return (
        graph.ent2id,
        graph.rel2id,
        graph.triples,
        train_loader,
        dev_loader,
        test_loader,
    )
