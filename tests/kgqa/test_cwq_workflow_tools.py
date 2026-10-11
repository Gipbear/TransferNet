"""验证符号比较、数值证据标注与外部图谱查询边界。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.evidence import EvidenceFormatter
from kgqa.agent.workflow.models import AnswerResult, QueryResult, RuntimeQuestion
from kgqa.agent.workflow.qlever import QLeverClient, parse_term
from kgqa.agent.workflow.tools.comparison import NumericReranker
from kgqa.agent.workflow.tools.conjunction import AttributeTagger


class FakeLibrary:
    def classify(self, question):
        return SimpleNamespace(kind="superlative", confidence=1.0)

    def comparison_plan(self, question, kind):
        return {"chain": ["location.location.area"], "margin": 1.0, "operator": "argmax", "confidence": 1.0}

    def numeric_chains(self, question):
        return [("measurement.rate",)]


class FakeKG:
    def __init__(self, rows, status="ok", topic_rows=None, label_rows=None):
        self.rows = rows
        self.status = status
        self.topic_rows = topic_rows or []
        self.label_rows = label_rows or []

    def attributes(self, mids, chain):
        return QueryResult(self.status, tuple(self.rows), "hash")

    def intersect_topic_attribute(self, topics, chain, value):
        st = "ok" if self.topic_rows else "empty"
        return QueryResult(st, tuple(self.topic_rows), "hash_topic")


    def labels(self, mids):
        return QueryResult("ok", tuple(self.label_rows), "hash_labels")



class TestTools(unittest.TestCase):
    def setUp(self):
        self.question = RuntimeQuestion.from_record({
            "sample_index": 0, "question": "Which country has a rate of 1.8?",
            "candidates": [{"name": "A", "mids": ["m.a"]}, {"name": "B", "mids": ["m.b"]}],
            "paths": [{"path": [["topic", "r", "A"]]}, {"path": [["topic", "r", "B"]]}],
        })
        self.initial = AnswerResult(("B",), False, True)

    def test_numeric_ties_keep_all_extrema(self):
        kg = FakeKG([{"mid": "m.a", "value": "2"}, {"mid": "m.b", "value": "2.0"}])
        result = NumericReranker(WorkflowConfig(), FakeLibrary(), kg).run(self.question, self.initial)
        self.assertEqual(result.answers, ("B", "A"))

    def test_missing_comparison_values_fall_back(self):
        result = NumericReranker(WorkflowConfig(), FakeLibrary(), FakeKG([])).run(self.question, self.initial)
        self.assertFalse(result.applied)

    def test_comparative_filter_prioritizes_initial_only_when_condition_holds(self):
        values = [("A", "number", 1, {}), ("B", "number", 2, {})]
        plan = {"operator": "gt", "threshold": "0"}
        selected = NumericReranker._select(values, plan, self.initial)
        self.assertEqual([item[0] for item in selected], ["B"])
        plan["threshold"] = "1.5"
        selected = NumericReranker._select(values, plan, AnswerResult(("A",), False, True))
        self.assertEqual([item[0] for item in selected], ["B"])

    def test_comparative_cap_counts_distinct_entities(self):
        values = [("A", "number", 1, {}), ("A", "number", 2, {}), ("B", "number", 3, {})]
        initial = AnswerResult(("A", "B"), False, True)
        selected = NumericReranker._select(values, {"operator": "gt", "threshold": "0"}, initial)
        self.assertEqual([item[0] for item in selected], ["A", "B"])

    def test_top2_fallback_attribute_chain(self):
        class Top2Library(FakeLibrary):
            def comparison_plan(self, question, kind):
                return {
                    "chain": ["invalid.chain"],
                    "candidate_chains": [["invalid.chain"], ["location.location.area"]],
                    "margin": 1.0,
                    "operator": "argmax",
                    "confidence": 1.0,
                }

        class Top2KG:
            def attributes(self, mids, chain):
                if chain == ["invalid.chain"]:
                    return QueryResult("ok", (), "h1")
                return QueryResult("ok", ({"mid": "m.b", "value": "100"},), "h2")

        result = NumericReranker(WorkflowConfig(), Top2Library(), Top2KG()).run(self.question, self.initial)
        self.assertTrue(result.applied)
        self.assertEqual(result.answers, ("B",))

    def test_path_consistency_filter_removes_schema_noise(self):
        question = RuntimeQuestion.from_record({
            "sample_index": 1,
            "question": "Which university attended by X has the most students?",
            "candidates": [
                {"name": "Missouri", "mids": ["m.m"]},
                {"name": "Oxford", "mids": ["m.ox"]}
            ],
            "paths": [
                {
                    "path": [["X", "education.education.student_rev", "Missouri"]],
                    "log_score": -9.7,
                    "tail_mid": "m.m"
                },
                {
                    "path": [["schema", "rdf-schema#domain_rev", "Oxford"]],
                    "log_score": -14.5,
                    "tail_mid": "m.ox"
                }
            ]
        })
        kg = FakeKG([
            {"mid": "m.m", "value": "20000"},
            {"mid": "m.ox", "value": "30000"}
        ])
        result = NumericReranker(WorkflowConfig(), FakeLibrary(), kg).run(question, AnswerResult(("Missouri",), False, True))
        self.assertTrue(result.applied)
        # Oxford 虽然数值 30000 更大，但因纯 schema 噪声路径被踢出，Missouri 胜出
        self.assertEqual(result.answers, ("Missouri",))

    def test_numeric_label_uses_equal_value_and_keeps_original_paths(self):
        kg = FakeKG([{"mid": "m.a", "value": "1.80"}, {"mid": "m.b", "value": "1.87"}])
        result = AttributeTagger(WorkflowConfig(), FakeLibrary(), kg).run(self.question, self.initial)
        self.assertTrue(result.applied)
        self.assertIn("1.8", result.paths[0]["path"][-1][2])
        self.assertEqual(self.question.paths[0]["path"][-1][2], "A")
        self.assertEqual(result.paths[1]["path"][-1][2], "B")

    def test_more_than_three_matches_do_not_trigger(self):
        candidates = [{"name": str(i), "mids": [f"m.a{i}"]} for i in range(4)]
        question = RuntimeQuestion.from_record({
            "sample_index": 0, "question": "rate of 1.8", "candidates": candidates,
        })
        kg = FakeKG([{"mid": f"m.a{i}", "value": "1.8"} for i in range(4)])
        self.assertFalse(AttributeTagger(WorkflowConfig(), FakeLibrary(), kg).run(question, self.initial).applied)

    def test_truncated_attribute_result_allows_matching_entities(self):
        kg = FakeKG([{"mid": "m.a", "value": "1.8"}], "truncated")
        self.assertTrue(AttributeTagger(WorkflowConfig(), FakeLibrary(), kg).run(self.question, self.initial).applied)

    def test_topic_reverse_intersection_rescues_when_candidates_fail(self):
        question = RuntimeQuestion.from_record({
            "sample_index": 0, "question": "What happened to topic with rate of 1.8?",
            "topic_mids": ["m.topic"],
            "candidates": [{"name": "A", "mids": ["m.a"]}],
            "paths": [{"path": [["m.topic", "r", "A"]]}],
        })
        kg = FakeKG([], "empty", topic_rows=[{"cand": "m.c"}], label_rows=[{"mid": "m.c", "name": "C"}])
        result = AttributeTagger(WorkflowConfig(), FakeLibrary(), kg).run(question, self.initial)
        self.assertTrue(result.applied)
        self.assertEqual(result.answers, ("C",))
        self.assertIn("1.8", result.paths[-1]["path"][-1][2])

    def test_superlative_preserves_initial_when_tied(self):
        values = [("Winston Churchill", "number", 1900, {}), ("Franklin D. Roosevelt", "number", 1900, {})]
        plan = {"operator": "argmin"}
        initial = AnswerResult(("Franklin D. Roosevelt",), False, True)
        selected = NumericReranker._select(values, plan, initial)
        self.assertEqual([item[0] for item in selected], ["Franklin D. Roosevelt", "Winston Churchill"])

    def test_date_and_year_mixed_comparison(self):
        chain = ["organization.organization.date_founded"]
        v1 = EvidenceFormatter.comparable("1848", chain)
        v2 = EvidenceFormatter.comparable("1868-03-23", chain)
        self.assertIsNotNone(v1)
        self.assertIsNotNone(v2)
        self.assertEqual(v1[0], "date")
        self.assertEqual(v2[0], "date")
        self.assertLess(v1[1], v2[1])

    def test_low_margin_with_candidate_chains_still_applied(self):
        class LowMarginLibrary(FakeLibrary):
            def comparison_plan(self, question, kind):
                return {
                    "chain": ["location.location.area"],
                    "candidate_chains": [["location.location.area"], ["location.country.area"]],
                    "margin": 0.02,
                    "operator": "argmax",
                    "confidence": 1.0,
                }
        kg = FakeKG([{"mid": "m.a", "value": "100"}])
        result = NumericReranker(WorkflowConfig(), LowMarginLibrary(), kg).run(self.question, self.initial)
        self.assertTrue(result.applied)
        self.assertEqual(result.answers, ("A",))




class TestQLever(unittest.TestCase):
    def test_terms_preserve_literal_and_normalize_mid(self):
        self.assertEqual(parse_term("<http://rdf.freebase.com/ns/m.a>"), "m.a")
        self.assertEqual(parse_term('"1.80"^^<http://www.w3.org/2001/XMLSchema#decimal>'), "1.80")

    def test_qlever_unquoted_language_tag_is_not_part_of_entity_name(self):
        self.assertEqual(parse_term("Germany@en"), "Germany")

    def test_cached_tsv_is_reused_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            client = QLeverClient(WorkflowConfig(), Path(folder))
            query = "SELECT ?x WHERE {}"
            key = client.query_key(query)
            (Path(folder) / f"{key}.tsv").write_text('?x\n"value"\n')
            result = client.query(query, 10)
            self.assertTrue(result.cache_hit)
            self.assertEqual(result.rows[0]["x"], "value")

    def test_cached_typed_numeric_literal_remains_parseable(self):
        with tempfile.TemporaryDirectory() as folder:
            client = QLeverClient(WorkflowConfig(), Path(folder))
            query = "SELECT ?value WHERE {}"
            key = client.query_key(query)
            literal = '"1.80"^^<http://www.w3.org/2001/XMLSchema#decimal>'
            (Path(folder) / f"{key}.tsv").write_text(f"?value\n{literal}\n")
            self.assertEqual(client.query(query, 10).rows[0]["value"], "1.80")


if __name__ == "__main__":
    unittest.main()
