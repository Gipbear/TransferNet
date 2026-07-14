"""MetaQA_KB 在线得分生产（前向逻辑迁移自 MetaQA_KB/predict.py）。"""
from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import torch
from tqdm import tqdm

from utils.misc import idx_to_one_hot
from utils.path_utils import filter_tensor
from MetaQA_KB.data import DataLoader
from MetaQA_KB.model import TransferNet
from kgqa.backbone.base import ScoreProducer
from kgqa.core.contracts import CacheMeta, SampleScore, ScoreBundle


class MetaQAScoreProducer(ScoreProducer):
    def __init__(self, num_steps: int = 3, dim_word: int = 300, dim_hidden: int = 1024,
                 aux_hop: int = 1, per_hop_limit: int = 0):
        self.num_steps = num_steps
        self.dim_word = dim_word
        self.dim_hidden = dim_hidden
        self.aux_hop = aux_hop
        self.per_hop_limit = per_hop_limit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ckpt_path: str | None = None

    def load_checkpoint(self, ckpt_path: str) -> None:
        self._ckpt_path = ckpt_path

    def produce(self, input_dir: str, qa_file: str, *, split: str = "test",
                batch_size: int = 64, topk: int = 500,
                show_progress: bool = True, progress_callback=None) -> ScoreBundle:
        assert self._ckpt_path, "先调用 load_checkpoint()"
        import os
        vocab_json = os.path.join(input_dir, "vocab.json")
        loader = DataLoader(vocab_json, qa_file, batch_size)
        vocab = loader.vocab
        num_ent = len(vocab["entity2id"])

        # TransferNet.__init__(args, dim_word, dim_hidden, vocab)；args 需 num_steps/aux_hop/input_dir
        args = SimpleNamespace(num_steps=self.num_steps, aux_hop=self.aux_hop,
                               input_dir=input_dir)
        model = TransferNet(args, self.dim_word, self.dim_hidden, vocab)
        model.load_state_dict(
            torch.load(self._ckpt_path, map_location="cpu", weights_only=True), strict=False)
        model = model.to(self.device)
        model.kg.Msubj = model.kg.Msubj.to(self.device)
        model.kg.Mobj = model.kg.Mobj.to(self.device)
        model.kg.Mrel = model.kg.Mrel.to(self.device)
        model.eval()

        kept = defaultdict(int)
        samples: list[SampleScore] = []
        total = len(loader.dataset)
        with torch.no_grad(), tqdm(total=total, desc=f"MetaQA {split} 得分",
                                   unit="题", dynamic_ncols=True,
                                   disable=not show_progress) as progress:
            for batch in loader:
                questions, topic_entities, answers, hops = batch
                topic_onehot = idx_to_one_hot(topic_entities, num_ent).to(self.device)
                answers_onehot = idx_to_one_hot(answers, num_ent)
                answers_onehot[:, 0] = 0  # 排除 DUMMY_ENTITY
                outputs = model(questions.to(self.device), topic_onehot)
                e_score = outputs["e_score"].cpu()
                rel_probs = [t.cpu() for t in outputs["rel_probs"]]
                ent_probs = [t.cpu() for t in outputs["ent_probs"]]
                hops_list = hops.tolist()
                for i in range(e_score.shape[0]):
                    hop = int(hops_list[i])
                    if self.per_hop_limit and kept[hop] >= self.per_hop_limit:
                        continue
                    kept[hop] += 1
                    topic_ids = [int(x) for (x, _) in filter_tensor(topic_onehot[i].cpu(), 1)]
                    gold_ids = answers_onehot[i].gt(0.5).nonzero().squeeze(1).tolist()
                    question_str = " ".join(
                        vocab["id2word"][w] for w in questions[i].cpu().tolist() if w > 0)
                    ent_idx_hop, ent_sc_hop = [], []
                    for t in range(self.num_steps):
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
                    hop_attn = torch.zeros(self.num_steps)
                    hop_attn[hop - 1] = 1.0
                    samples.append(SampleScore(
                        question=question_str,
                        topic_ids=topic_ids, gold_ids=[int(g) for g in gold_ids],
                        hop_attn=hop_attn,
                        rel_probs=[rel_probs[t][i].clone() for t in range(self.num_steps)],
                        ent_indices=ent_idx_hop, ent_scores=ent_sc_hop,
                        e_score_indices=eidxs[emask], e_score_values=evals[emask],
                        sample_index=len(samples), hop=hop,
                    ))
                progress.update(e_score.shape[0])
                if progress_callback:
                    progress_callback(min(progress.n, total), total)
                if self.per_hop_limit and all(
                        kept[h] >= self.per_hop_limit for h in range(1, self.num_steps + 1)):
                    break  # hop 分块有序，三跳配额都满即可提前停

        meta = CacheMeta(dataset="MetaQA", split=split, id2ent=vocab["id2entity"],
                         id2rel=vocab["id2relation"], num_samples=len(samples),
                         topk_entities=topk, input_dir=input_dir, qa_file=qa_file)
        return ScoreBundle(meta=meta, samples=samples)
