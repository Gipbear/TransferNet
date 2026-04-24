"""Reusable TransferNet MMR path retrieval service.

The offline ``predict.py`` scripts mix three concerns: dataset/model loading,
batch evaluation, and path serialization.  This module keeps only the reusable
single-query path retrieval pieces so an HTTP server can load the model once and
serve many requests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from utils.path_utils import build_valid_edges_dict, filter_tensor, mmr_diversity_beam_search


@dataclass
class RetrievalResult:
    question: str
    topics: list[str]
    hop: int
    mmr_reason_paths: list[dict[str, Any]]
    prediction: dict[str, float]
    elapsed_ms: float
    raw_topics: list[str]
    raw_mmr_reason_paths: list[dict[str, Any]]
    raw_prediction: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "topics": self.topics,
            "hop": self.hop,
            "mmr_reason_paths": self.mmr_reason_paths,
            "prediction": self.prediction,
            "elapsed_ms": self.elapsed_ms,
            "raw_topics": self.raw_topics,
            "raw_mmr_reason_paths": self.raw_mmr_reason_paths,
            "raw_prediction": self.raw_prediction,
        }


def _load_entity_label(path: Optional[str]) -> Optional[dict[str, str]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"entity label file not found: {path}")
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                mapping[parts[0].strip()] = parts[1].strip()
    return mapping


def _safe_torch_load(path: str, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class TransferNetPathRetriever:
    """Single-query MMR path retriever backed by a loaded TransferNet model."""

    def __init__(
        self,
        *,
        dataset: str,
        input_dir: str,
        ckpt: str,
        device: Optional[str] = None,
        bert_name: Optional[str] = None,
        num_steps: int = 3,
        dim_word: int = 300,
        dim_hidden: int = 1024,
        aux_hop: int = 1,
        entity_label: Optional[str] = None,
    ):
        dataset_key = dataset.lower()
        if dataset_key in {"webqsp", "web"}:
            dataset_key = "webqsp"
        elif dataset_key in {"metaqa", "metaqa_kb", "meta"}:
            dataset_key = "metaqa"
        else:
            raise ValueError(
                "path server currently supports dataset='webqsp' or 'metaqa'. "
                "CompWebQ needs per-sample subgraphs and is not exposed as a "
                "single-query server yet."
            )

        self.dataset = dataset_key
        self.input_dir = input_dir
        self.ckpt = ckpt
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mid2label = _load_entity_label(entity_label)

        if self.dataset == "webqsp":
            self._load_webqsp(bert_name or "bert-base-uncased")
        else:
            self._load_metaqa(num_steps, dim_word, dim_hidden, aux_hop)

        state = _safe_torch_load(ckpt, self.device)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print("Missing keys: {}".format("; ".join(missing)), flush=True)
        if unexpected:
            print("Unexpected keys: {}".format("; ".join(unexpected)), flush=True)

        self.model = self.model.to(self.device)
        self._move_sparse_matrices()
        self.model.eval()

    # ------------------------------------------------------------------
    # Dataset loaders
    # ------------------------------------------------------------------

    def _load_webqsp(self, bert_name: str) -> None:
        from WebQSP.data import load_data
        from WebQSP.model import TransferNet

        args = argparse.Namespace(bert_name=bert_name)
        ent2id, rel2id, triples, _train_loader, val_loader = load_data(
            self.input_dir, bert_name, 16
        )
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.id2ent = val_loader.id2ent
        self.id2rel = val_loader.id2rel
        self.tokenizer = val_loader.tokenizer
        self.model = TransferNet(args, ent2id, rel2id, triples)

        triples_list = [[int(s), int(r), int(o)] for s, r, o in triples.tolist()]
        self.valid_edges_dict = build_valid_edges_dict(triples_list)

    def _load_metaqa(
        self,
        num_steps: int,
        dim_word: int,
        dim_hidden: int,
        aux_hop: int,
    ) -> None:
        from MetaQA_KB.data import load_vocab
        from MetaQA_KB.model import TransferNet

        args = argparse.Namespace(
            input_dir=self.input_dir,
            num_steps=num_steps,
            dim_word=dim_word,
            dim_hidden=dim_hidden,
            aux_hop=aux_hop,
        )
        self.vocab = load_vocab(os.path.join(self.input_dir, "vocab.json"))
        self.ent2id = self.vocab["entity2id"]
        self.rel2id = self.vocab["relation2id"]
        self.id2ent = self.vocab["id2entity"]
        self.id2rel = self.vocab["id2relation"]
        self.model = TransferNet(args, dim_word, dim_hidden, self.vocab)

        msubj = np.load(os.path.join(self.input_dir, "Msubj.npy"))
        mobj = np.load(os.path.join(self.input_dir, "Mobj.npy"))
        mrel = np.load(os.path.join(self.input_dir, "Mrel.npy"))
        triples = np.stack([msubj[:, 1], mrel[:, 1], mobj[:, 1]], axis=1).tolist()
        self.valid_edges_dict = build_valid_edges_dict(
            [[int(s), int(r), int(o)] for s, r, o in triples]
        )

    def _move_sparse_matrices(self) -> None:
        if self.dataset == "webqsp":
            self.model.Msubj = self.model.Msubj.to(self.device)
            self.model.Mobj = self.model.Mobj.to(self.device)
            self.model.Mrel = self.model.Mrel.to(self.device)
        elif self.dataset == "metaqa":
            self.model.kg.Msubj = self.model.kg.Msubj.to(self.device)
            self.model.kg.Mobj = self.model.kg.Mobj.to(self.device)
            self.model.kg.Mrel = self.model.kg.Mrel.to(self.device)

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def _resolve(self, entity: str) -> str:
        if self.mid2label:
            return self.mid2label.get(entity, entity)
        return entity

    def _topic_ids(self, topic_entities: list[str]) -> list[int]:
        ids: list[int] = []
        missing: list[str] = []
        for entity in topic_entities:
            if entity in self.ent2id:
                ids.append(int(self.ent2id[entity]))
            else:
                missing.append(entity)
        if missing:
            raise KeyError(f"unknown topic_entities: {missing}")
        if not ids:
            raise ValueError("topic_entities must contain at least one entity id/name")
        return ids

    def _encode_webqsp(self, question: str, topic_ids: list[int]) -> tuple[torch.Tensor, dict]:
        heads = torch.zeros((1, len(self.ent2id)), dtype=torch.float32)
        heads[0, topic_ids] = 1.0
        questions = self.tokenizer(
            question.strip(),
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return heads.to(self.device), {k: v.to(self.device) for k, v in questions.items()}

    def _encode_metaqa(self, question: str, topic_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        words = self._tokenize_metaqa(question)
        word2id = self.vocab["word2id"]
        unk = word2id.get("<UNK>", 1)
        pad = word2id.get("<PAD>", 0)
        ids = [word2id.get(word, unk) for word in words] or [unk]
        question_tensor = torch.tensor([ids], dtype=torch.long)
        if question_tensor.size(1) == 0:
            question_tensor = torch.tensor([[pad]], dtype=torch.long)
        heads = torch.zeros((1, len(self.ent2id)), dtype=torch.float32)
        heads[0, topic_ids] = 1.0
        return question_tensor.to(self.device), heads.to(self.device)

    @staticmethod
    def _tokenize_metaqa(question: str) -> list[str]:
        text = question.lower()
        try:
            from nltk import word_tokenize

            return word_tokenize(text)
        except Exception:
            return re.findall(r"\w+|[^\w\s]", text)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        *,
        question: str,
        topic_entities: list[str],
        hop: Optional[int] = None,
        beam_size: int = 20,
        lambda_val: float = 0.2,
        prediction_threshold: float = 0.9,
    ) -> RetrievalResult:
        if beam_size < 1:
            raise ValueError("beam_size must be >= 1")
        if hop is not None and hop < 1:
            raise ValueError("hop must be >= 1")

        t0 = time.perf_counter()
        topic_ids = self._topic_ids(topic_entities)

        with torch.no_grad():
            if self.dataset == "webqsp":
                heads, questions = self._encode_webqsp(question, topic_ids)
                outputs = self.model(heads, questions)
                hop_count = int(hop or (outputs["hop_attn"][0].argmax().item() + 1))
            else:
                questions, heads = self._encode_metaqa(question, topic_ids)
                outputs = self.model(questions, heads)
                hop_count = int(hop or self.model.num_steps)

            rel_probs_cpu = [tensor[0].detach().cpu() for tensor in outputs["rel_probs"]]
            ent_probs_cpu = [tensor[0].detach().cpu() for tensor in outputs["ent_probs"]]
            e_score = outputs["e_score"][0].detach().cpu()

        max_hop = len(rel_probs_cpu)
        if hop_count > max_hop:
            raise ValueError(f"hop={hop_count} exceeds model num_steps={max_hop}")

        topic_scores = [(topic_id, 1.0) for topic_id in topic_ids]
        single_outputs = {"rel_probs": rel_probs_cpu, "ent_probs": ent_probs_cpu}
        precomputed = [
            (
                dict(filter_tensor(single_outputs["rel_probs"][t], 0.01)),
                dict(filter_tensor(single_outputs["ent_probs"][t], 0.01)),
            )
            for t in range(hop_count)
        ]
        mmr_paths = mmr_diversity_beam_search(
            single_outputs,
            self.valid_edges_dict,
            topic_scores,
            hop_count,
            K=beam_size,
            lambda_val=lambda_val,
            precomputed_dicts=precomputed,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return RetrievalResult(
            question=question,
            topics=[self._resolve(self.id2ent[topic_id]) for topic_id in topic_ids],
            hop=hop_count,
            mmr_reason_paths=self._serialize_paths(mmr_paths),
            prediction=self._serialize_prediction(e_score, prediction_threshold),
            elapsed_ms=round(elapsed_ms, 1),
            raw_topics=[self.id2ent[topic_id] for topic_id in topic_ids],
            raw_mmr_reason_paths=self._serialize_paths_raw(mmr_paths),
            raw_prediction=self._serialize_prediction_raw(e_score, prediction_threshold),
        )

    def _serialize_paths(
        self, paths: list[tuple[list[int], list[int], float]]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for nodes, rels, score in paths:
            rows.append(
                {
                    "path": [
                        [
                            self._resolve(self.id2ent[nodes[k]]),
                            self.id2rel[rels[k]],
                            self._resolve(self.id2ent[nodes[k + 1]]),
                        ]
                        for k in range(len(rels))
                    ],
                    "log_score": round(float(score), 6),
                }
            )
        return rows

    def _serialize_paths_raw(
        self, paths: list[tuple[list[int], list[int], float]]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for nodes, rels, score in paths:
            rows.append(
                {
                    "path": [
                        [
                            self.id2ent[nodes[k]],
                            self.id2rel[rels[k]],
                            self.id2ent[nodes[k + 1]],
                        ]
                        for k in range(len(rels))
                    ],
                    "log_score": round(float(score), 6),
                }
            )
        return rows

    def _serialize_prediction(
        self, e_score: torch.Tensor, threshold: float
    ) -> dict[str, float]:
        return {
            self._resolve(self.id2ent[entity_id]): float(f"{score:.3f}")
            for entity_id, score in filter_tensor(e_score, threshold)
        }

    def _serialize_prediction_raw(
        self, e_score: torch.Tensor, threshold: float
    ) -> dict[str, float]:
        return {
            self.id2ent[entity_id]: float(f"{score:.3f}")
            for entity_id, score in filter_tensor(e_score, threshold)
        }

    def info(self) -> dict[str, Any]:
        vram_mb = None
        if torch.cuda.is_available():
            vram_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)
        return {
            "dataset": self.dataset,
            "input_dir": self.input_dir,
            "ckpt": self.ckpt,
            "device": str(self.device),
            "entity_count": len(self.ent2id),
            "relation_count": len(self.rel2id),
            "edge_source_count": len(self.valid_edges_dict),
            "vram_allocated_mb": vram_mb,
        }
