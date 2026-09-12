"""Prime 三跳 TransferNet 在线得分生产器。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

from PharmKG.model import TransferNet
from PharmKG.predict import id_score_pairs
from Prime.data import DataLoader, LoaderSpec, RangeCache
from Prime.graph import load_graph
from kgqa.backbone.base import ScoreProducer
from kgqa.core.contracts import CacheMeta, SampleScore, ScoreBundle
from utils.misc import batch_device


class PrimeCheckpointNotLoadedError(RuntimeError):
    """表示 Prime 得分生产器尚未绑定 checkpoint。"""


class PrimeScoreProducer(ScoreProducer):
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
        batch_size: int = 4,
        topk: int = 500,
        show_progress: bool = True,
        progress_callback=None,
    ) -> ScoreBundle:
        if not self._ckpt_path:
            raise PrimeCheckpointNotLoadedError
        graph = load_graph(input_dir)
        loader = DataLoader(
            LoaderSpec(Path(input_dir), split, self.bert_name, batch_size, qa_file=Path(qa_file), limit=self.limit),
            graph,
            RangeCache(graph),
        )
        args = SimpleNamespace(bert_name=self.bert_name, num_steps=3)
        model = TransferNet(args, loader.ent2id, loader.rel2id, graph.triples)
        model.load_state_dict(torch.load(self._ckpt_path, map_location="cpu", weights_only=True))
        model = model.to(self.device)
        for attribute in ("Msubj", "Mobj", "Mrel"):
            setattr(model, attribute, getattr(model, attribute).to(self.device))
        model.eval()
        samples: list[SampleScore] = []
        with torch.no_grad(), tqdm(
            total=len(loader.dataset),
            desc=f"Prime {split} 得分",
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
            dataset="Prime",
            split=split,
            id2ent={index: str(index) for index in range(len(loader.ent2id))},
            id2rel=loader.id2rel,
            num_samples=len(samples),
            topk_entities=topk,
            input_dir=input_dir,
            qa_file=qa_file,
        )
        return ScoreBundle(meta=meta, samples=samples)

    @staticmethod
    def _topk(values: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
        scores, indices = values.topk(min(topk, values.shape[0]))
        positive = scores > 0
        return indices[positive], scores[positive]

    def _append_batch(self, samples, batch, outputs, loader, topk: int) -> None:
        entity_scores = outputs["e_score"].cpu()
        hop_attn = outputs["hop_attn"].cpu()
        relation_scores = [values.cpu() for values in outputs["rel_probs"]]
        step_entity_scores = [values.cpu() for values in outputs["ent_probs"]]
        for row_index in range(entity_scores.shape[0]):
            sample_index = len(samples)
            step_topk = [self._topk(values[row_index], topk) for values in step_entity_scores]
            final_indices, final_values = self._topk(entity_scores[row_index], topk)
            samples.append(SampleScore(
                question=loader.qa_text[sample_index],
                topic_ids=[entity_id for entity_id, _ in id_score_pairs(batch[0][row_index], 1)],
                gold_ids=[entity_id for entity_id, _ in id_score_pairs(batch[2][row_index], 1)],
                hop_attn=hop_attn[row_index].clone(),
                rel_probs=[values[row_index].clone() for values in relation_scores],
                ent_indices=[pair[0] for pair in step_topk],
                ent_scores=[pair[1] for pair in step_topk],
                e_score_indices=final_indices,
                e_score_values=final_values,
                sample_index=sample_index,
                hop=loader.hops[sample_index],
            ))
