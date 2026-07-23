import unittest

import torch

from kgqa.core.contracts import SampleScore
from kgqa.retrieve.graph.global_kg import GlobalKG


class TestShortestPathBaseline(unittest.TestCase):
    def setUp(self):
        from kgqa.retrieve.shortest_path import ShortestPathParams

        self.params = ShortestPathParams(candidate_topk=2, max_paths_per_pair=20, path_budget=20)
        self.sample = SampleScore(
            question="toy question",
            topic_ids=[0],
            gold_ids=[3],
            hop_attn=torch.tensor([0.1, 0.9]),
            rel_probs=[torch.zeros(8), torch.zeros(8)],
            ent_indices=[torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)],
            ent_scores=[torch.tensor([]), torch.tensor([])],
            e_score_indices=torch.tensor([3, 4, 0]),
            e_score_values=torch.tensor([0.9, 0.5, 0.1]),
            sample_index=7,
        )
        # 两条等长最短路径通向 3；边访问顺序应由 relation_id 决定。
        self.kg = GlobalKG.from_triples([
            [0, 2, 1], [0, 1, 2], [1, 4, 3], [2, 3, 3], [0, 7, 4],
            [3, 6, 5], [5, 5, 3],  # 不应因为回路产生重复节点路径。
        ])
        self.id2ent = {0: "topic", 1: "one", 2: "two", 3: "gold", 4: "other", 5: "loop"}
        self.id2rel = {1: "r1", 2: "r2", 3: "r3", 4: "r4", 5: "r5", 6: "r6", 7: "r7"}

    def test_stably_enumerates_top_candidate_shortest_paths(self):
        from kgqa.retrieve.engine import RetrievalDiagnostics
        from kgqa.retrieve.shortest_path import retrieve_shortest_paths_one

        diagnostics = RetrievalDiagnostics()
        result = retrieve_shortest_paths_one(
            self.sample, self.kg, self.id2ent, self.id2rel, params=self.params,
            diagnostics=diagnostics,
        )

        self.assertEqual(result.sample_index, 7)
        self.assertEqual(result.prediction, {"gold": 0.9})
        self.assertEqual(
            [path["path"] for path in result.paths],
            [
                [["topic", "r1", "two"], ["two", "r3", "gold"]],
                [["topic", "r2", "one"], ["one", "r4", "gold"]],
                [["topic", "r7", "other"]],
            ],
        )
        self.assertTrue(all(path["log_score"] < 0 for path in result.paths))
        self.assertEqual(diagnostics.expanded_states, 4)
        self.assertEqual(diagnostics.candidate_paths, 3)
        self.assertEqual(diagnostics.final_paths, 3)

    def test_path_budget_candidate_boundary_and_determinism(self):
        from kgqa.retrieve.shortest_path import ShortestPathParams, retrieve_shortest_paths_one

        params = ShortestPathParams(candidate_topk=1, max_paths_per_pair=1, path_budget=1)
        first = retrieve_shortest_paths_one(
            self.sample, self.kg, self.id2ent, self.id2rel, params=params,
        )
        second = retrieve_shortest_paths_one(
            self.sample, self.kg, self.id2ent, self.id2rel, params=params,
        )

        self.assertEqual(first.paths, second.paths)
        self.assertEqual(len(first.paths), 1)
        self.assertEqual(first.paths[0]["path"][-1][-1], "gold")
        self.assertNotIn("other", [path["path"][-1][-1] for path in first.paths])

    def test_rejects_invalid_path_budget(self):
        from kgqa.retrieve.shortest_path import ShortestPathParams

        with self.assertRaisesRegex(ValueError, "candidate_topk"):
            ShortestPathParams(candidate_topk=0)
        with self.assertRaisesRegex(ValueError, "max_paths_per_pair"):
            ShortestPathParams(max_paths_per_pair=0)
        with self.assertRaisesRegex(ValueError, "path_budget"):
            ShortestPathParams(path_budget=0)


if __name__ == "__main__":
    unittest.main()
