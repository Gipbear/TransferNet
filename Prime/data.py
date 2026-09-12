"""加载 Prime 原生 ID 图谱和问答数据。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import torch
from transformers import AutoTokenizer

from Prime.graph import PrimeGraph, load_graph, reachable_entities
from utils.huggingface import from_pretrained_local_first


class QARecord(TypedDict):
    question_id: str
    question: str
    topic_entity: str
    answers: list[str]
    hop: int


@dataclass(frozen=True, slots=True)
class LoaderSpec:
    input_dir: Path
    split: str
    bert_name: str
    batch_size: int
    training: bool = False
    qa_file: Path | None = None
    limit: int = 0


@dataclass(slots=True)  # noqa: MUTABLE_OK
class RangeCache:
    """按主题缓存大规模三跳候选，避免每轮训练重复展开图。"""

    graph: PrimeGraph
    values: dict[int, torch.Tensor] = field(default_factory=dict)

    def get(self, topic_id: int) -> torch.Tensor:
        if topic_id not in self.values:
            reachable = reachable_entities(self.graph.adjacency, topic_id, max_hop=3)
            self.values[topic_id] = torch.from_numpy(reachable)
        return self.values[topic_id]


EncodedQuestion = dict[str, torch.Tensor]
DatasetRow = tuple[int, EncodedQuestion, torch.Tensor]
Batch = tuple[list[torch.Tensor], EncodedQuestion, list[torch.Tensor], list[torch.Tensor]]


def _load_rows(path: Path) -> list[QARecord]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tokenize(rows: list[QARecord], tokenizer) -> list[EncodedQuestion]:
    encoded_rows: list[EncodedQuestion] = []
    for start in range(0, len(rows), 512):
        encoded = tokenizer(
            [row["question"] for row in rows[start:start + 512]],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        for offset in range(encoded["input_ids"].shape[0]):
            encoded_rows.append({key: value[offset:offset + 1] for key, value in encoded.items()})
    return encoded_rows


class Dataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[DatasetRow], ranges: RangeCache):
        self.rows = rows
        self.ranges = ranges

    def __getitem__(self, index: int):
        topic_id, question, answers = self.rows[index]
        topic = torch.tensor([topic_id], dtype=torch.long)
        return topic, question, answers, self.ranges.get(topic_id)

    def __len__(self) -> int:
        return len(self.rows)


def collate(rows) -> Batch:
    topics, questions, answers, entity_ranges = zip(*rows)
    question_batch = {key: torch.cat([question[key] for question in questions]) for key in questions[0]}
    return list(topics), question_batch, list(answers), list(entity_ranges)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, spec: LoaderSpec, graph: PrimeGraph, ranges: RangeCache):
        qa_rows = _load_rows(spec.qa_file or spec.input_dir / f"{spec.split}.json")
        if spec.limit:
            qa_rows = qa_rows[:spec.limit]
        tokenizer = from_pretrained_local_first(AutoTokenizer, spec.bert_name)
        questions = _tokenize(qa_rows, tokenizer)
        rows = [
            (
                int(row["topic_entity"]),
                question,
                torch.tensor([int(answer) for answer in row["answers"]], dtype=torch.long),
            )
            for row, question in zip(qa_rows, questions)
        ]
        self.ent2id = {str(index): index for index in range(len(graph.entity_names))}
        self.rel2id = graph.relation_ids
        self.id2ent = {index: name for index, name in enumerate(graph.entity_names)}
        self.id2rel = {relation_id: relation for relation, relation_id in graph.relation_ids.items()}
        self.qa_text = [row["question"] for row in qa_rows]
        self.hops = [row["hop"] for row in qa_rows]
        super().__init__(
            Dataset(rows, ranges),
            batch_size=spec.batch_size,
            shuffle=spec.training,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
        )


def load_data(input_dir: str, bert_name: str, batch_size: int):
    """加载 Prime 图谱及训练、验证、测试集合。"""
    graph = load_graph(input_dir)
    root = Path(input_dir)
    ranges = RangeCache(graph)
    train = DataLoader(LoaderSpec(root, "train", bert_name, batch_size, training=True), graph, ranges)
    dev = DataLoader(LoaderSpec(root, "dev", bert_name, batch_size), graph, ranges)
    test = DataLoader(LoaderSpec(root, "test", bert_name, batch_size), graph, ranges)
    ent2id = {str(index): index for index in range(len(graph.entity_names))}
    return ent2id, graph.relation_ids, graph.triples, train, dev, test
