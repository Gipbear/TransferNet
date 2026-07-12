import json
import os
import tempfile
import unittest
from unittest import mock

import torch

from kgqa.retrieve.graph.global_kg import GlobalKG
from kgqa.retrieve import engine
from kgqa.core.contracts import RetrieveResult


class _Sample:
    """最小 SampleScoreLike:单跳,两个候选尾(同 test_engine)。"""
    question = "toy question"
    topic_ids = [0]
    gold_ids = [1]
    hop_attn = torch.tensor([0.9, 0.1])
    rel_probs = [torch.tensor([0.0, 0.8, 0.7]), torch.tensor([0.0, 0.0, 0.0])]
    ent_indices = [torch.tensor([1, 2]), torch.tensor([], dtype=torch.long)]
    ent_scores = [torch.tensor([0.6, 0.5]), torch.tensor([])]
    e_score_indices = torch.tensor([1, 2])
    e_score_values = torch.tensor([0.95, 0.4])


class TestEngineGolden(unittest.TestCase):
    def setUp(self):
        self.kg = GlobalKG.from_triples([[0, 1, 1], [0, 2, 2]])
        self.id2rel = {1: "rel.one", 2: "rel.two"}

    def test_retrieve_one_carries_golden(self):
        """golden = gold_ids 经 id2ent 还原,与 topics/paths 同一实体空间。"""
        id2ent = {0: "m.topic", 1: "m.gold", 2: "m.other"}
        r = engine.retrieve_one(
            _Sample(), self.kg, id2ent, self.id2rel,
            beam_size=10, threshold=0.01, lambda_val=0.2,
        )
        self.assertEqual(r.golden, ["m.gold"])

    def test_golden_falls_back_to_str_id(self):
        """id2ent 缺条目时回退 str(id),不丢样本。"""
        id2ent = {0: "m.topic", 2: "m.other"}
        r = engine.retrieve_one(
            _Sample(), self.kg, id2ent, self.id2rel,
            beam_size=10, threshold=0.01, lambda_val=0.2,
        )
        self.assertEqual(r.golden, ["1"])


class TestCliWritesGolden(unittest.TestCase):
    def test_output_jsonl_contains_golden(self):
        from kgqa.retrieve.cli import retrieve as retrieve_cli

        result = RetrieveResult(
            question="q", topics=["m.topic"], hop=1,
            paths=[{"path": [["m.topic", "rel.one", "m.gold"]], "log_score": -0.1}],
            prediction={"m.gold": 0.95}, elapsed_ms=0.1, sample_index=0,
            golden=["m.gold"],
        )

        class _FakeBackend:
            def retrieve_all(self, *, limit=0, **params):
                return [result]

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.jsonl")
            args = retrieve_cli.build_parser().parse_args([
                "--dataset", "webqsp", "--backend", "offline",
                "--cache", "unused", "--input_dir", "unused",
                "--output", out,
            ])
            with mock.patch.object(retrieve_cli, "build_backend",
                                   return_value=_FakeBackend()):
                retrieve_cli.run_retrieval(args)
            with open(out, encoding="utf-8") as fh:
                rec = json.loads(fh.readline())
        self.assertEqual(rec["golden"], ["m.gold"])
        self.assertEqual(rec["mmr_reason_paths"], result.paths)


if __name__ == "__main__":
    unittest.main()
