"""验证独立评分、配对转移和评分前的样本身份检查。"""

import unittest

from kgqa.agent.workflow.scoring import WorkflowScorer


def record(index, before, after, source="baseline"):
    answer = lambda values: {"answers": values, "is_rejection": not values, "format_ok": True,
                             "cited_indices": [], "prompt_tokens": 10, "completion_tokens": 5, "elapsed_ms": 1}
    return {"question": {"sample_index": index, "question": f"q{index}", "paths": []},
            "route": {"kind": "composition", "confidence": 1}, "initial": answer(before),
            "final": answer(after), "final_paths": [], "source": source, "pfit_calls": 1,
            "elapsed_ms": 2, "tool": None}


class TestScoring(unittest.TestCase):
    def test_rescue_and_harm_are_paired_against_actual_initial_answers(self):
        labels = {0: {"question": "q0", "golden": ["A"]}, 1: {"question": "q1", "golden": ["B"]}}
        rows = [record(0, [], ["A"], "rescue"), record(1, ["B"], ["wrong"], "comparison")]
        report = WorkflowScorer.score_records(rows, labels)
        self.assertEqual(report["paired_hit"], {"rescue": 1, "harm": 1, "net": 0})
        self.assertEqual(report["overall"]["hit_any"], 0.5)

    def test_question_mismatch_is_rejected(self):
        labels = {0: {"question": "different", "golden": ["A"]}}
        with self.assertRaises(ValueError):
            WorkflowScorer.score_records([record(0, [], ["A"])], labels)

    def test_duplicate_prediction_is_rejected(self):
        labels = {0: {"question": "q0", "golden": ["A"]}}
        with self.assertRaises(ValueError):
            WorkflowScorer.score_records([record(0, [], []), record(0, [], [])], labels)

    def test_failed_followup_tokens_are_counted(self):
        row = record(0, ["A"], ["A"])
        row["pfit_calls"] = 2
        failure = {**row["initial"], "prompt_tokens": 99, "completion_tokens": 13, "format_ok": False}
        row["call_results"] = [row["initial"], failure]
        report = WorkflowScorer.score_records([row], {0: {"question": "q0", "golden": ["A"]}})
        self.assertEqual(report["efficiency"]["prompt_tokens"], 109)
        self.assertEqual(report["efficiency"]["completion_tokens"], 18)


if __name__ == "__main__":
    unittest.main()
