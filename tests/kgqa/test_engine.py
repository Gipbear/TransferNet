import math
import unittest
import torch

from kgqa.retrieve.graph.global_kg import GlobalKG
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

    def test_step_score_modes_follow_declared_formulas(self):
        rel_dict = {1: 0.6, 2: 0.4}
        ent_dict = {3: 0.75, 4: 0.25}
        relation = math.log(0.6)
        entity = math.log(0.75)
        self.assertAlmostEqual(engine.compute_step_score(rel_dict, ent_dict, 1, 3), relation + entity)
        self.assertAlmostEqual(engine.compute_step_score(rel_dict, ent_dict, 1, 3, "joint"), relation + entity)
        self.assertAlmostEqual(engine.compute_step_score(rel_dict, ent_dict, 1, 3, "relation_only"), relation)
        self.assertAlmostEqual(engine.compute_step_score(rel_dict, ent_dict, 1, 3, "entity_only"), entity)

    def test_single_score_modes_keep_intersection_candidate_space(self):
        rel_dicts = [{1: 0.8, 2: 0.7}]
        ent_dicts = [{1: 0.6, 2: 0.5}]
        expected = None
        for mode in ("joint", "relation_only", "entity_only"):
            candidates = engine.search_path_candidates(
                [0], rel_dicts, ent_dicts, 1, self.kg.valid_edges_dict, 10,
                step_score_mode=mode,
            )
            paths = {(tuple(candidate.nodes), tuple(candidate.rels)) for candidate in candidates}
            if expected is None:
                expected = paths
            self.assertEqual(paths, expected)

    def test_single_score_mode_rejects_terminal_entity_fusion(self):
        with self.assertRaisesRegex(ValueError, "必须设置 eta=0"):
            engine.validate_score_scheme("relation_only", 1.0)

    def test_candidate_score_is_tail_blend_with_length_normalization(self):
        candidate = engine.PathCandidate(
            nodes=[0, 1, 2], rels=[1, 2], hop=2,
            base_score=-8.0, final_tail_score=0.25,
        )

        score = engine.compute_candidate_score(candidate, eta=2.0)

        expected = (-8.0 + 2.0 * math.log(0.25 + engine.EPS)) / 2
        self.assertAlmostEqual(score, expected)

    def test_alpha_final_keyword_is_rejected(self):
        candidate = engine.PathCandidate(
            nodes=[0, 1], rels=[1], hop=1, base_score=-3.0, final_tail_score=0.5,
        )
        with self.assertRaises(TypeError):
            engine.compute_candidate_score(candidate, alpha_final=0.7)

    def test_candidate_hops_include_every_available_step(self):
        self.assertEqual(engine.candidate_hop_numbers(3), [1, 2, 3])

    def test_retrieve_one_returns_paths_and_prediction(self):
        r = engine.retrieve_one(
            _Sample(), self.kg, self.id2ent, self.id2rel,
            beam_size=10, threshold=0.01, lambda_val=0.2,
        )
        self.assertEqual(r.question, "toy question")
        self.assertEqual(r.hop, 1)
        self.assertTrue(r.paths)
        # 首条路径应命中 gold 尾 m.gold
        tails = [p["path"][-1][2] for p in r.paths]
        self.assertIn("m.gold", tails)
        # prediction 取 e_score argmax（0.95 > 0.4）→ 只含 m.gold
        self.assertEqual(set(r.prediction), {"m.gold"})

    def test_joint_mode_is_identical_to_omitted_mode(self):
        default = engine.retrieve_one(
            _Sample(), self.kg, self.id2ent, self.id2rel,
            beam_size=10, threshold=0.01, lambda_val=0.2, eta=1.5,
        )
        explicit = engine.retrieve_one(
            _Sample(), self.kg, self.id2ent, self.id2rel,
            beam_size=10, threshold=0.01, lambda_val=0.2, eta=1.5,
            step_score_mode="joint",
        )
        self.assertEqual(default.paths, explicit.paths)

    def test_drop_loopback_removes_self_return(self):
        paths = [([0, 1], [1], -0.1), ([0, 0], [1], -0.2)]
        kept = engine.drop_loopback_paths(paths)
        self.assertEqual(kept, [([0, 1], [1], -0.1)])


if __name__ == "__main__":
    unittest.main()
