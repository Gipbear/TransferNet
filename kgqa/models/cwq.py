"""CWQ 在线得分生产（前向逻辑迁移自 CompWebQ/predict.py，子图逐样本内嵌）。

不走 CompWebQ.data.load_data（它会 tokenize train 2.6GB 并整体 pickle），
直接读 entities.txt/relations.txt 建词表、仅对 qa_file 构造 DataLoader。
"""
from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

import torch

from utils.misc import batch_device, invert_dict
from utils.path_utils import filter_tensor
from CompWebQ.data import DataLoader
from CompWebQ.model import TransferNet
from kgqa.models.base import ScoreProducer
from kgqa.scores.base import CacheMeta, SampleScore, ScoreBundle


def _read_vocab(path: str) -> dict[str, int]:
    vocab: dict[str, int] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            vocab[line.strip()] = len(vocab)
    return vocab


def _valid_lines(qa_file: str, limit: int = 0) -> list[str]:
    """取非空子图样本的原始行（与 CompWebQ DataLoader 跳过空子图的规则对齐）。"""
    lines: list[str] = []
    with open(qa_file, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            if not json.loads(line).get("subgraph", {}).get("tuples"):
                continue
            lines.append(line)
            if limit and len(lines) >= limit:
                break
    return lines


class CWQScoreProducer(ScoreProducer):
    def __init__(self, bert_name: str = "bert-base-cased", num_steps: int = 2,
                 num_ways: int = 1, limit: int = 0):
        self.bert_name = bert_name
        self.num_steps = num_steps
        self.num_ways = num_ways
        self.limit = limit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ckpt_path: str | None = None

    def load_checkpoint(self, ckpt_path: str) -> None:
        self._ckpt_path = ckpt_path

    def produce(self, input_dir: str, qa_file: str, *, split: str = "test",
                batch_size: int = 16, topk: int = 500) -> ScoreBundle:
        assert self._ckpt_path, "先调用 load_checkpoint()"
        ent2id = _read_vocab(os.path.join(input_dir, "entities.txt"))
        rel2id = _read_vocab(os.path.join(input_dir, "relations.txt"))

        lines = _valid_lines(qa_file, self.limit)
        raw_questions = [json.loads(l)["question"].strip() for l in lines]
        if self.limit:
            # 小子集截断成临时文件，DataLoader 免读全量 358MB
            fd, qa_path = tempfile.mkstemp(suffix=".jsonl")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.writelines(lines)
        else:
            qa_path = qa_file
        try:
            loader = DataLoader(qa_path, self.bert_name, ent2id, rel2id, batch_size)
        finally:
            if self.limit:
                os.unlink(qa_path)

        args = SimpleNamespace(bert_name=self.bert_name, num_steps=self.num_steps,
                               num_ways=self.num_ways)
        model = TransferNet(args, ent2id, rel2id)
        model.load_state_dict(torch.load(self._ckpt_path, map_location="cpu"), strict=False)
        model = model.to(self.device)
        model.eval()

        samples: list[SampleScore] = []
        with torch.no_grad():
            for batch in loader:
                outputs = model(*batch_device(batch, self.device))
                e_score = outputs["e_score"].cpu()
                hop_attn = outputs["hop_attn"].cpu()
                rel_probs = [t.cpu() for t in outputs["rel_probs"]]
                ent_probs = [t.cpu() for t in outputs["ent_probs"]]
                num_steps = len(rel_probs)
                for i in range(e_score.shape[0]):
                    topic_ids = [int(x) for (x, _) in filter_tensor(batch[0][i], 1)]
                    gold_ids = [int(x) for (x, _) in filter_tensor(batch[2][i], 0.5)]
                    ent_idx_hop, ent_sc_hop = [], []
                    for t in range(num_steps):
                        vec = ent_probs[t][i]
                        k = min(topk, vec.shape[0])
                        vals, idxs = vec.topk(k)
                        mask = vals > 0
                        ent_idx_hop.append(idxs[mask])
                        ent_sc_hop.append(vals[mask])
                    ev = e_score[i]
                    k = min(topk, ev.shape[0])
                    evals, eidxs = ev.topk(k)
                    emask = evals > 0
                    samples.append(SampleScore(
                        question=raw_questions[len(samples)],
                        topic_ids=topic_ids, gold_ids=gold_ids,
                        hop_attn=hop_attn[i].clone(),
                        rel_probs=[rel_probs[t][i].clone() for t in range(num_steps)],
                        ent_indices=ent_idx_hop, ent_scores=ent_sc_hop,
                        e_score_indices=eidxs[emask], e_score_values=evals[emask],
                        sample_index=len(samples),
                        triples=batch[3][i].tolist(),
                    ))
        meta = CacheMeta(dataset="CWQ", split=split, id2ent=invert_dict(ent2id),
                         id2rel=invert_dict(rel2id), num_samples=len(samples),
                         topk_entities=topk, input_dir=input_dir, qa_file=qa_file)
        return ScoreBundle(meta=meta, samples=samples)
