"""常驻加载 TransferNet，对无标签外部子图评分并调用原生 RSDPR。"""

from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from CompWebQ.model import TransferNet
from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.interfaces import GraphScorer
from kgqa.core.contracts import SampleScore
from kgqa.retrieve.engine import retrieve_one
from kgqa.retrieve.graph.global_kg import GlobalKG
from utils.huggingface import from_pretrained_local_first


class TransferNetGraphScorer(GraphScorer):
    def __init__(self, ckpt: str, relations: str, bert_name: str, device: str = "cpu") -> None:
        forward = Path(relations).read_text().splitlines()
        self.rel2id = {name: i for i, name in enumerate([*forward, *[r + "_rev" for r in forward]])}
        self.id2rel = {index: name for name, index in self.rel2id.items()}
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.num_steps = int(state["hop-way_0.weight"].shape[0])
        if state["rel-way_0.weight"].shape[0] != len(self.rel2id):
            raise ValueError("checkpoint 关系词表维度不一致")
        args = SimpleNamespace(bert_name=bert_name, num_steps=self.num_steps, num_ways=1)
        self.model = TransferNet(args, {"placeholder": 0}, self.rel2id)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.tokenizer = from_pretrained_local_first(AutoTokenizer, bert_name)
        self.device = device
        self.lock = Lock()

    def score(self, question: str, topics: tuple[str, ...], edges: list[list[str]], config: WorkflowConfig) -> dict:
        known = [edge for edge in edges if edge[1] in self.rel2id]
        unknown = sorted({edge[1] for edge in edges if edge[1] not in self.rel2id})
        nodes = sorted(set(topics) | {node for h, _, t in known for node in (h, t)})
        if not known or not nodes:
            return {"paths": [], "unknown_relations": unknown, "num_steps": self.num_steps}
        ent2id = {node: i for i, node in enumerate(nodes)}
        triples = []
        for head, relation, tail in known:
            triples.append([ent2id[head], self.rel2id[relation], ent2id[tail]])
            reverse = relation[:-4] if relation.endswith("_rev") else relation + "_rev"
            triples.append([ent2id[tail], self.rel2id[reverse], ent2id[head]])
        sample = self._forward(question, topics, ent2id, triples)
        result = retrieve_one(
            sample, GlobalKG.from_triples(triples), dict(enumerate(nodes)), self.id2rel,
            beam_size=20, lambda_val=config.lambda_val, eta=config.eta,
            threshold=0.01, penalty_mode=config.penalty_mode,
        )
        return {"paths": result.paths, "unknown_relations": unknown, "num_steps": self.num_steps,
                "score_source": "transfernet"}

    def _forward(self, question: str, topics: tuple, ent2id: dict, triples: list) -> SampleScore:
        encoded = self.tokenizer(question.strip(), max_length=64, padding="max_length", return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        heads = torch.zeros((1, len(ent2id)), device=self.device)
        topic_ids = [ent2id[topic] for topic in topics if topic in ent2id]
        heads[0, topic_ids] = 1
        tensor = torch.tensor(triples, device=self.device)
        with self.lock, torch.no_grad():
            self.model.num_ents = len(ent2id)
            output = self.model(heads, encoded, triples=[tensor])
        ent_indices, ent_scores = [], []
        for probabilities in output["ent_probs"]:
            row = probabilities[0].cpu()
            selected = (row > 0).nonzero().flatten()
            ent_indices.append(selected)
            ent_scores.append(row[selected])
        final = output["e_score"][0].cpu()
        selected = (final > 0).nonzero().flatten()
        return SampleScore(question, topic_ids, [], output["hop_attn"][0].cpu(),
                           [p[0].cpu() for p in output["rel_probs"]], ent_indices, ent_scores,
                           selected, final[selected], triples=triples)
