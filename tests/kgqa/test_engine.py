import unittest
import torch

from kgqa.kg.global_kg import GlobalKG
from kgqa.retrieve import engine


class _Sample:
    """最小 SampleScoreLike：单跳，两个候选尾。"""
    question = "toy question"
    topic_ids = [0]
    gold_ids = [1]
    # hop_attn argmax=0 → hop_num=1
    hop_attn = torch.tensor([0.9, 0.1])
    rel_probs = [torch.tensor([0.0, 0.8, 0.7]), torch.tensor([0.0, 0.0, 0.0])]
    ent_indices = [torch.tensor([1, 2]), torch.tensor([], dtype=torch.long)]
    ent_scores = [torch.tensor([0.6, 0.5]), torch.tensor([])]
    e_score_indices = torch.tensor([1, 2])
    e_score_values = torch.tensor([0.95, 0.4])


class TestEngine(unittest.TestCase):
    def setUp(self):
        # 边：0 --rel1--> 1, 0 --rel2--> 2
        self.kg = GlobalKG.from_triples([[0, 1, 1], [0, 2, 2]])
        self.id2ent = {0: "m.topic", 1: "m.gold", 2: "m.other"}
        self.id2rel = {1: "rel.one", 2: "rel.two"}

    def test_reconstruct_rel_dict_threshold(self):
        d = engine.reconstruct_rel_dict(torch.tensor([0.0, 0.8, 0.005]), 0.01)
        self.assertEqual(set(d), {1})
        self.assertAlmostEqual(d[1], 0.8, places=5)

    def test_retrieve_one_returns_paths_and_prediction(self):
        r = engine.retrieve_one(
            _Sample(), self.kg, self.id2ent, self.id2rel,
            method="tail_blend", beam_size=10, threshold=0.01, lambda_val=0.2,
        )
        self.assertEqual(r.question, "toy question")
        self.assertEqual(r.hop, 1)
        self.assertTrue(r.paths)
        # 首条路径应命中 gold 尾 m.gold
        tails = [p["path"][-1][2] for p in r.paths]
        self.assertIn("m.gold", tails)
        # prediction 取 e_score argmax（0.95 > 0.4）→ 只含 m.gold
        self.assertEqual(set(r.prediction), {"m.gold"})

    def test_drop_loopback_removes_self_return(self):
        paths = [([0, 1], [1], -0.1), ([0, 0], [1], -0.2)]
        kept = engine.drop_loopback_paths(paths)
        self.assertEqual(kept, [([0, 1], [1], -0.1)])


if __name__ == "__main__":
    unittest.main()
