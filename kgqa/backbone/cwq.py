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
from tqdm import tqdm

from utils.misc import batch_device, invert_dict
from CompWebQ.data import DataLoader
from CompWebQ.model import TransferNet
from kgqa.backbone.base import ScoreProducer
from kgqa.core.contracts import CacheMeta, SampleScore, ScoreBundle


def _read_vocab(path: str) -> dict[str, int]:
    vocab: dict[str, int] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            vocab[line.strip()] = len(vocab)
    return vocab


def _read_rel_vocab(path: str, add_rev: bool = False) -> dict[str, int]:
    """关系词表；add_rev 时按原文件顺序追加 ``_rev`` 项。

    与 CompWebQ/data.py load_data 的扩展方式逐位一致（正向 0..n-1，反向 n..2n-1），
    训练用 --rev 得到的 ckpt 只有在同样扩展后关系分类器的输出维度才对得上。
    """
    rel2id = _read_vocab(path)
    if add_rev:
        for rel in list(rel2id):
            rel2id[rel + "_rev"] = len(rel2id)
    return rel2id


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


def _assert_relation_vocab(state: dict, rel2id: dict[str, int], rev: bool) -> None:
    """核对 ckpt 的关系分类器输出维度与当前关系词表一致。

    load_state_dict(strict=False) 会静默跳过 shape 不匹配的参数：用 --rev 训练的
    ckpt 配上未扩展的词表时，关系分类器保持随机初始化而不报错，得分全是噪声。
    """
    key = "rel-way_0.weight"
    if key not in state:
        return
    ckpt_num_rel = state[key].shape[0]
    if ckpt_num_rel == len(rel2id):
        return
    hint = "去掉 --rev" if rev else "加上 --rev"
    raise ValueError(
        f"ckpt 关系数 {ckpt_num_rel} 与词表 {len(rel2id)} 不一致"
        f"（当前 rev={rev}）；该 ckpt 多半需要{hint}")


class CWQScoreProducer(ScoreProducer):
    def __init__(self, bert_name: str = "bert-base-cased", num_steps: int = 2,
                 num_ways: int = 1, limit: int = 0, rev: bool = False):
        if num_ways != 1:
            raise ValueError(
                f"num_ways={num_ways} 不能用于生成得分缓存：TransferNet.forward 在 way 循环内"
                "重新赋值 rel_probs/ent_probs，返回的只是最后一个 way 的分布"
                "（只有 e_score 做了跨 way 的 prod）。MMR beam search 吃的正是 ent_probs，"
                "缓存会静默变成半个模型的结果")
        self.bert_name = bert_name
        self.num_steps = num_steps
        self.num_ways = num_ways
        self.limit = limit
        self.rev = rev
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ckpt_path: str | None = None

    def load_checkpoint(self, ckpt_path: str) -> None:
        self._ckpt_path = ckpt_path

    def produce(self, input_dir: str, qa_file: str, *, split: str = "test",
                batch_size: int = 16, topk: int = 500,
                show_progress: bool = True, progress_callback=None) -> ScoreBundle:
        assert self._ckpt_path, "先调用 load_checkpoint()"
        ent2id = _read_vocab(os.path.join(input_dir, "entities.txt"))
        rel2id = _read_rel_vocab(os.path.join(input_dir, "relations.txt"), self.rev)

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
            loader = DataLoader(qa_path, self.bert_name, ent2id, rel2id, batch_size,
                                add_rev=self.rev)
        finally:
            if self.limit:
                os.unlink(qa_path)

        args = SimpleNamespace(bert_name=self.bert_name, num_steps=self.num_steps,
                               num_ways=self.num_ways)
        state = torch.load(self._ckpt_path, map_location="cpu", weights_only=True)
        _assert_relation_vocab(state, rel2id, self.rev)
        model = TransferNet(args, ent2id, rel2id)
        model.load_state_dict(state, strict=False)
        model = model.to(self.device)
        model.eval()

        samples: list[SampleScore] = []
        with torch.no_grad(), tqdm(total=len(raw_questions), desc=f"CWQ {split} 得分",
                                   unit="题", dynamic_ncols=True,
                                   disable=not show_progress) as progress:
            for batch in loader:
                outputs = model(*batch_device(batch, self.device))
                e_score = outputs["e_score"].cpu()
                hop_attn = outputs["hop_attn"].cpu()
                rel_probs = [t.cpu() for t in outputs["rel_probs"]]
                ent_probs = [t.cpu() for t in outputs["ent_probs"]]
                num_steps = len(rel_probs)
                for i in range(e_score.shape[0]):
                    topic_ids = [int(x) for x in batch[0][i].tolist()]
                    gold_ids = [int(x) for x in batch[2][i].tolist()]
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
                progress.update(e_score.shape[0])
                if progress_callback:
                    progress_callback(len(samples), len(raw_questions))
        meta = CacheMeta(dataset="CWQ", split=split, id2ent=invert_dict(ent2id),
                         id2rel=invert_dict(rel2id), num_samples=len(samples),
                         topk_entities=topk, input_dir=input_dir, qa_file=qa_file)
        return ScoreBundle(meta=meta, samples=samples)
