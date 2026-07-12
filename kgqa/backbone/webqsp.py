"""WebQSP 在线得分生产（前向逻辑迁移自 WebQSP/dump_scores.py）。"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from tqdm import tqdm

from utils.misc import batch_device
from WebQSP.data import DataLoader, load_data
from WebQSP.model import TransferNet
from WebQSP.predict import id_score_pairs
from kgqa.backbone.base import ScoreProducer
from kgqa.core.contracts import CacheMeta, SampleScore, ScoreBundle


class WebQSPScoreProducer(ScoreProducer):
    def __init__(self, bert_name: str = "BAAI/bge-base-en-v1.5"):
        self.bert_name = bert_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ckpt_path: str | None = None

    def load_checkpoint(self, ckpt_path: str) -> None:
        self._ckpt_path = ckpt_path

    def produce(self, input_dir: str, qa_file: str, *, split: str = "test",
                batch_size: int = 16, topk: int = 500,
                show_progress: bool = True, progress_callback=None) -> ScoreBundle:
        assert self._ckpt_path, "先调用 load_checkpoint()"
        # qa_file 按调用方给定的路径（相对 CWD 或绝对）直接使用，不再拼 input_dir。
        ent2id, rel2id, triples, _train, _val = load_data(input_dir, self.bert_name, batch_size)
        loader = DataLoader(input_dir, qa_file, self.bert_name, ent2id, rel2id, batch_size)
        args = SimpleNamespace(bert_name=self.bert_name)  # TransferNet 仅需 args.bert_name
        model = TransferNet(args, ent2id, rel2id, triples)
        model.load_state_dict(
            torch.load(self._ckpt_path, map_location="cpu", weights_only=True), strict=False)
        model = model.to(self.device)
        for attr in ("Msubj", "Mobj", "Mrel"):
            setattr(model, attr, getattr(model, attr).to(self.device))
        model.eval()

        raw_questions = getattr(loader, "qa_text", None)
        assert raw_questions is not None, "DataLoader 缺 qa_text"

        samples: list[SampleScore] = []
        with torch.no_grad(), tqdm(total=len(raw_questions), desc=f"WebQSP {split} 得分",
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
                    topic_ids = [x for (x, _) in id_score_pairs(batch[0][i], 1)]
                    gold_ids = [x for (x, _) in id_score_pairs(batch[2][i], 1)]
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
                    ))
                progress.update(e_score.shape[0])
                if progress_callback:
                    progress_callback(len(samples), len(raw_questions))
        meta = CacheMeta(dataset="WebQSP", split=split, id2ent=loader.id2ent,
                         id2rel=loader.id2rel, num_samples=len(samples),
                         topk_entities=topk, input_dir=input_dir, qa_file=qa_file)
        return ScoreBundle(meta=meta, samples=samples)
