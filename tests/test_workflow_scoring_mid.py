import unittest

from kgqa.agent.workflow.scoring import WorkflowScorer


class TestWorkflowScorerMIDAlignment(unittest.TestCase):
    def test_text_matching_without_mids(self):
        answer = {
            "answers": ["Paris"],
            "cited_indices": [1],
            "format_ok": True,
        }
        paths = [
            {"path": [["France", "capital", "Paris"]], "canonical_tail": "Paris"}
        ]
        gold = ["Paris"]
        metrics = WorkflowScorer._metrics(answer, paths, gold)
        self.assertEqual(metrics["hit_any"], 1)
        self.assertEqual(metrics["hit1"], 1)
        self.assertEqual(metrics["exact_match"], 1)

    def test_mid_alignment_when_text_differs(self):
        # 大模型生成了演员真实姓名，但标签文本只记录了俗名或别名，两者通过 MID 对齐成功
        answer = {
            "answers": ["William Roache"],
            "cited_indices": [1],
            "format_ok": True,
        }
        paths = [
            {
                "path": [["Coronation Street", "actor", "William Roache"]],
                "canonical_tail": "William Roache",
                "tail_mid": "m.01j_gs",
            }
        ]
        # 标签文本未能完全字面匹配，但提供了 golden_mids
        gold = ["m.01j_gs"]
        gold_mids = ["m.01j_gs"]
        metrics = WorkflowScorer._metrics(answer, paths, gold, gold_mids)
        self.assertEqual(metrics["hit_any"], 1)
        self.assertEqual(metrics["hit1"], 1)
        self.assertEqual(metrics["exact_match"], 1)
        self.assertEqual(metrics["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
