"""验证多跳救援的完整路径、截断与预算状态。"""

import unittest

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.models import AnswerResult, QueryResult, RuntimeQuestion
from kgqa.agent.workflow.tools.rescue import GraphRescuer


class FakePatterns:
    def extract_patterns(self, *args):
        return [{"steps": [{"predicate": "people.person.spouse", "reverse": False}], "score": 0.8}]


class FakeKG:
    def __init__(self, status="ok"):
        self.status = status

    def adjacency(self, frontier, step):
        return QueryResult(self.status, ({"source": "m.a", "target": "m.b"},))

    def labels(self, mids):
        return QueryResult("ok", tuple({"mid": mid, "name": "B"} for mid in mids))


class TestRescue(unittest.TestCase):
    def setUp(self):
        self.question = RuntimeQuestion.from_record({"sample_index": 0, "question": "Q", "topic_mids": ["m.a"]})
        self.initial = AnswerResult((), True, True)

    def test_template_backend_is_named_and_produces_actual_mid_path(self):
        rescuer = GraphRescuer(WorkflowConfig(rescue_scorer="template"), FakePatterns(), FakeKG(), names={"m.a": "A"})
        outcome = rescuer.run(self.question, self.initial)
        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.stats["score_source"], "template")
        self.assertEqual(outcome.paths[0]["tail_mid"], "m.b")

    def test_truncated_query_returns_explicit_fallback(self):
        rescuer = GraphRescuer(WorkflowConfig(), FakePatterns(), FakeKG("truncated"))
        outcome = rescuer.run(self.question, self.initial)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "query_truncated")

    def test_failed_query_does_not_become_empty_kg_fact(self):
        rescuer = GraphRescuer(WorkflowConfig(), FakePatterns(), FakeKG("failed"))
        self.assertEqual(rescuer.run(self.question, self.initial).reason, "query_failed")

    def test_failed_labels_query_does_not_offer_mid_as_answer(self):
        class LabelFailedKG(FakeKG):
            def labels(self, mids):
                return QueryResult("failed", ())

        rescuer = GraphRescuer(WorkflowConfig(rescue_scorer="template"), FakePatterns(), LabelFailedKG())
        outcome = rescuer.run(self.question, self.initial)
        self.assertFalse(outcome.applied)
        self.assertNotIn("m.b", rescuer.names)

    def test_name_query_can_recover_after_transient_failure(self):
        class RecoveringKG(FakeKG):
            def __init__(self):
                super().__init__()
                self.label_calls = 0

            def labels(self, mids):
                self.label_calls += 1
                if self.label_calls == 1:
                    return QueryResult("failed", ())
                return super().labels(mids)

        kg = RecoveringKG()
        rescuer = GraphRescuer(WorkflowConfig(rescue_scorer="template"), FakePatterns(), kg)
        self.assertFalse(rescuer.run(self.question, self.initial).applied)
        outcome = rescuer.run(self.question, self.initial)
        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.paths[0]["path"][0][2], "B")
        self.assertEqual(kg.label_calls, 2)

    def test_known_candidate_name_recovers_without_mid_cache_poisoning(self):
        class LabelFailedKG(FakeKG):
            def labels(self, mids):
                return QueryResult("failed", ())

        rescuer = GraphRescuer(WorkflowConfig(rescue_scorer="template"), FakePatterns(), LabelFailedKG())
        rescuer.run(self.question, self.initial)
        question = RuntimeQuestion.from_record({
            "sample_index": 1, "question": "Q2", "topic_mids": ["m.a"],
            "candidates": [{"name": "B", "mids": ["m.b"]}],
        })
        outcome = rescuer.run(question, self.initial)
        self.assertEqual(outcome.paths[0]["path"][0][2], "B")

    def test_partial_search_counts_actual_queries_and_records_stop_reason(self):
        class WideKG(FakeKG):
            def adjacency(self, frontier, step):
                return QueryResult("ok", tuple({"source": "m.a", "target": f"m.b{i}"} for i in range(3)))

        config = WorkflowConfig(rescue_scorer="template", step_limit=1, allow_partial_rescue=True)
        outcome = GraphRescuer(config, FakePatterns(), WideKG()).run(self.question, self.initial)
        self.assertTrue(outcome.applied)
        self.assertTrue(outcome.stats["partial"])
        self.assertEqual(outcome.stats["adjacency_query_n"], 1)
        self.assertIn("step_limit_truncated", outcome.stats["stop_reasons"])

    def test_strict_search_rejects_internal_step_truncation(self):
        class WideKG(FakeKG):
            def adjacency(self, frontier, step):
                return QueryResult("ok", tuple({"source": "m.a", "target": f"m.b{i}"} for i in range(3)))

        config = WorkflowConfig(rescue_scorer="template", step_limit=1)
        outcome = GraphRescuer(config, FakePatterns(), WideKG()).run(self.question, self.initial)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "step_limit_truncated")

    def test_template_fallback_is_disabled_unless_configured(self):
        class EmptyScorer:
            def score(self, *args):
                return {"paths": [], "score_source": "transfernet"}

        outcome = GraphRescuer(WorkflowConfig(), FakePatterns(), FakeKG(), EmptyScorer()).run(
            self.question, self.initial
        )
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "no_scored_paths")

    def test_enabled_template_fallback_has_separate_provenance(self):
        class EmptyScorer:
            def score(self, *args):
                return {"paths": [], "score_source": "transfernet", "num_steps": 2}

        config = WorkflowConfig(allow_template_fallback=True)
        outcome = GraphRescuer(config, FakePatterns(), FakeKG(), EmptyScorer()).run(self.question, self.initial)
        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.stats["score_source"], "template_fallback")
        self.assertEqual(outcome.stats["gnn_path_n"], 0)

    def test_partial_search_keeps_valid_prefix_and_records_later_failure(self):
        class TwoPatterns(FakePatterns):
            def extract_patterns(self, *args):
                first = super().extract_patterns(*args)[0]
                return [first, {"steps": [{"predicate": "bad", "reverse": False}], "score": 0.7}]

        class MixedKG(FakeKG):
            def adjacency(self, frontier, step):
                if step["predicate"] == "bad":
                    return QueryResult("failed", ())
                return super().adjacency(frontier, step)

        config = WorkflowConfig(rescue_scorer="template", allow_partial_rescue=True)
        outcome = GraphRescuer(config, TwoPatterns(), MixedKG()).run(self.question, self.initial)
        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.stats["adjacency_query_n"], 2)
        self.assertEqual(outcome.stats["query_status_counts"], {"ok": 1, "failed": 1})
        self.assertIn("query_failed", outcome.stats["stop_reasons"])

    def test_partial_search_preserves_paths_with_explicit_query_budget_stop(self):
        class TwoPatterns(FakePatterns):
            def extract_patterns(self, *args):
                return super().extract_patterns(*args) * 2

        config = WorkflowConfig(rescue_scorer="template", max_queries=1, allow_partial_rescue=True)
        outcome = GraphRescuer(config, TwoPatterns(), FakeKG()).run(self.question, self.initial)
        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.stats["adjacency_query_n"], 1)
        self.assertIn("query_budget_exhausted", outcome.stats["stop_reasons"])

    def test_partial_triple_budget_never_keeps_an_incomplete_path(self):
        class WideKG(FakeKG):
            def adjacency(self, frontier, step):
                return QueryResult("ok", tuple({"source": "m.a", "target": f"m.b{i}"} for i in range(3)))

        config = WorkflowConfig(rescue_scorer="template", triple_budget=1, allow_partial_rescue=True)
        outcome = GraphRescuer(config, FakePatterns(), WideKG()).run(self.question, self.initial)
        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.stats["edge_n"], 1)
        self.assertEqual(len(outcome.paths), 1)
        self.assertIn("triple_budget_truncated", outcome.stats["stop_reasons"])

    def test_max_paths_per_relation_caps_redundant_paths(self):
        class MultiPattern:
            def extract_patterns(self, *args):
                return [
                    {"steps": [{"predicate": "rel_frequent", "reverse": False}], "score": 0.9},
                    {"steps": [{"predicate": "rel_rare", "reverse": False}], "score": 0.8},
                ]

        class MultiEdgeKG(FakeKG):
            def adjacency(self, frontier, step):
                if step["predicate"] == "rel_frequent":
                    return QueryResult("ok", tuple({"source": "m.a", "target": f"m.b{i}"} for i in range(5)))
                return QueryResult("ok", ({"source": "m.a", "target": "m.rare"},))

            def labels(self, mids):
                return QueryResult("ok", tuple({"mid": mid, "name": f"Name_{mid}"} for mid in mids))

        config = WorkflowConfig(rescue_scorer="template", max_paths_per_relation=2)
        outcome = GraphRescuer(config, MultiPattern(), MultiEdgeKG()).run(self.question, self.initial)
        self.assertTrue(outcome.applied)
        freq_count = sum(item["path"][0][1] == "rel_frequent" for item in outcome.paths)
        rare_count = sum(item["path"][0][1] == "rel_rare" for item in outcome.paths)
        self.assertEqual(freq_count, 2)
        self.assertEqual(rare_count, 1)


if __name__ == "__main__":
    unittest.main()
