"""PharmKG TransferNet 在线得分生产器。"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from tqdm import tqdm

from PharmKG.data import DataLoader, load_graph
from PharmKG.model import TransferNet
from PharmKG.predict import id_score_pairs
from kgqa.backbone.base import ScoreProducer
from kgqa.core.contracts import CacheMeta, SampleScore, ScoreBundle
from utils.misc import batch_device


class PharmKGScoreProducer(ScoreProducer):
    def __init__(self, bert_name: str = "BAAI/bge-base-en-v1.5"):
        self.bert_name = bert_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ckpt_path: str | None = None

    def load_checkpoint(self, ckpt_path: str) -> None:
        self._ckpt_path = ckpt_path

    def produce(self, input_dir: str, qa_file: str, *, split: str = "test",
                batch_size: int = 16, topk: int = 500,
                show_progress: bool = True, progress_callback=None) -> ScoreBundle:
        if not self._ckpt_path:
            raise RuntimeError("先调用 load_checkpoint()")
        ent2id, rel2id, triples = load_graph(input_dir)
        loader = DataLoader(input_dir, qa_file, self.bert_name, ent2id, rel2id, batch_size)
        args = SimpleNamespace(bert_name=self.bert_name)
        model = TransferNet(args, ent2id, rel2id, triples)
        model.load_state_dict(
            torch.load(self._ckpt_path, map_location="cpu", weights_only=True), strict=False,
        )
        model = model.to(self.device)
        for attr in ("Msubj", "Mobj", "Mrel"):
            setattr(model, attr, getattr(model, attr).to(self.device))
        model.eval()

        raw_questions = getattr(loader, "qa_text", None)
        assert raw_questions is not None, "DataLoader 缺 qa_text"
        samples: list[SampleScore] = []
        with torch.no_grad(), tqdm(
            total=len(raw_questions),
            desc=f"PharmKG {split} 得分",
            unit="题",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as progress:
            for batch in loader:
                outputs = model(*batch_device(batch, self.device))
                e_score = outputs["e_score"].cpu()
                hop_attn = outputs["hop_attn"].cpu()
                rel_probs = [values.cpu() for values in outputs["rel_probs"]]
                ent_probs = [values.cpu() for values in outputs["ent_probs"]]
                num_steps = len(rel_probs)
                for index in range(e_score.shape[0]):
                    topic_ids = [entity_id for entity_id, _ in id_score_pairs(batch[0][index], 1)]
                    gold_ids = [entity_id for entity_id, _ in id_score_pairs(batch[2][index], 1)]
                    ent_indices = []
                    ent_scores = []
                    for step in range(num_steps):
                        values = ent_probs[step][index]
                        count = min(topk, values.shape[0])
                        scores, indices = values.topk(count)
                        mask = scores > 0
                        ent_indices.append(indices[mask])
                        ent_scores.append(scores[mask])
                    final_values = e_score[index]
                    count = min(topk, final_values.shape[0])
                    final_scores, final_indices = final_values.topk(count)
                    final_mask = final_scores > 0
                    samples.append(SampleScore(
                        question=raw_questions[len(samples)],
                        topic_ids=topic_ids,
                        gold_ids=gold_ids,
                        hop_attn=hop_attn[index].clone(),
                        rel_probs=[rel_probs[step][index].clone() for step in range(num_steps)],
                        ent_indices=ent_indices,
                        ent_scores=ent_scores,
                        e_score_indices=final_indices[final_mask],
                        e_score_values=final_scores[final_mask],
                        sample_index=len(samples),
                    ))
                progress.update(e_score.shape[0])
                if progress_callback:
                    progress_callback(len(samples), len(raw_questions))
        meta = CacheMeta(
            dataset="PharmKG",
            split=split,
            id2ent=loader.id2ent,
            id2rel=loader.id2rel,
            num_samples=len(samples),
            topk_entities=topk,
            input_dir=input_dir,
            qa_file=qa_file,
        )
        return ScoreBundle(meta=meta, samples=samples)
