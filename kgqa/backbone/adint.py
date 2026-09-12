"""ADInt TransferNet 在线得分生产器。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

from ADInt.data import DataLoader, LoaderSpec, load_graph
from PharmKG.model import TransferNet
from PharmKG.predict import id_score_pairs
from kgqa.backbone.base import ScoreProducer
from kgqa.core.contracts import CacheMeta, SampleScore, ScoreBundle
from utils.misc import batch_device


class ADIntScoreProducer(ScoreProducer):
    def __init__(self, bert_name: str = "BAAI/bge-base-en-v1.5", limit: int = 0):
        self.bert_name = bert_name
        self.limit = limit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ckpt_path: str | None = None

    def load_checkpoint(self, ckpt_path: str) -> None:
        self._ckpt_path = ckpt_path

    def produce(
        self,
        input_dir: str,
        qa_file: str,
        *,
        split: str = "test",
        batch_size: int = 16,
        topk: int = 500,
        show_progress: bool = True,
        progress_callback=None,
    ) -> ScoreBundle:
        if not self._ckpt_path:
            raise RuntimeError("先调用 load_checkpoint()")
        graph = load_graph(input_dir)
        loader = DataLoader(
            LoaderSpec(input_dir=Path(input_dir), split=split,
                       bert_name=self.bert_name, batch_size=batch_size,
                       qa_file=Path(qa_file), limit=self.limit),
            graph,
        )
        model = TransferNet(SimpleNamespace(bert_name=self.bert_name), graph.ent2id, graph.rel2id, graph.triples)
        state = torch.load(self._ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        model = model.to(self.device)
        for attribute in ("Msubj", "Mobj", "Mrel"):
            setattr(model, attribute, getattr(model, attribute).to(self.device))
        model.eval()

        samples: list[SampleScore] = []
        with torch.no_grad(), tqdm(
            total=len(loader.dataset),
            desc=f"ADInt {split} 得分",
            unit="题",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as progress:
            for batch in loader:
                outputs = model(*batch_device(batch, self.device))
                self._append_batch(samples, batch, outputs, loader, topk)
                progress.update(len(batch[0]))
                if progress_callback:
                    progress_callback(len(samples), len(loader.dataset))
        meta = CacheMeta(
            dataset="ADInt",
            split=split,
            id2ent=loader.id2ent,
            id2rel=loader.id2rel,
            num_samples=len(samples),
            topk_entities=topk,
            input_dir=input_dir,
            qa_file=qa_file,
        )
        return ScoreBundle(meta=meta, samples=samples)

    @staticmethod
    def _topk(values: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
        count = min(topk, values.shape[0])
        scores, indices = values.topk(count)
        mask = scores > 0
        return indices[mask], scores[mask]

    def _append_batch(self, samples, batch, outputs, loader, topk: int) -> None:
        e_score = outputs["e_score"].cpu()
        hop_attn = outputs["hop_attn"].cpu()
        rel_probs = [values.cpu() for values in outputs["rel_probs"]]
        ent_probs = [values.cpu() for values in outputs["ent_probs"]]
        for row_index in range(e_score.shape[0]):
            sample_index = len(samples)
            ent_topk = [self._topk(values[row_index], topk) for values in ent_probs]
            final_indices, final_scores = self._topk(e_score[row_index], topk)
            samples.append(SampleScore(
                question=loader.qa_text[sample_index],
                topic_ids=[entity_id for entity_id, _ in id_score_pairs(batch[0][row_index], 1)],
                gold_ids=[entity_id for entity_id, _ in id_score_pairs(batch[2][row_index], 1)],
                hop_attn=hop_attn[row_index].clone(),
                rel_probs=[values[row_index].clone() for values in rel_probs],
                ent_indices=[pair[0] for pair in ent_topk],
                ent_scores=[pair[1] for pair in ent_topk],
                e_score_indices=final_indices,
                e_score_values=final_scores,
                sample_index=sample_index,
                hop=loader.hops[sample_index],
            ))
