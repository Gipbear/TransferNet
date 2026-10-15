"""CWQ 比较属性证据抽取与规则判断测试。"""
from __future__ import annotations

import unittest

from kgqa.retrieve.cwq_attribute_audit import build_strict_audits
from kgqa.retrieve.cwq_attribute_evidence import (
    CandidateValues,
    build_candidate_constrained_query,
    build_sparql_query,
    evaluate_candidates,
    trace_attribute_chain,
)


class TestCwqAttributeEvidence(unittest.TestCase):
    def test_traces_two_hop_attribute_chain(self):
        tuples = [[10, 3, 20], [20, 4, 30], [10, 8, 40]]
        relation_ids = {"rel.first": 3, "rel.value": 4}

        result = trace_attribute_chain(tuples, relation_ids, 10, ("rel.first", "rel.value"))

        self.assertEqual(result.status, "found")
        self.assertEqual(result.terminal_entity_ids, (30,))

    def test_reports_missing_relation_vocabulary(self):
        result = trace_attribute_chain([[10, 3, 20]], {"rel.first": 3}, 10, ("rel.first", "rel.value"))

        self.assertEqual(result.status, "missing_relation")
        self.assertEqual(result.terminal_entity_ids, ())

    def test_builds_two_hop_freebase_query(self):
        query = build_sparql_query(
            ("government.politician.government_positions_held", "government.government_position_held.from"),
            ("m.01p369", "m.05m665"),
        )

        self.assertIn("<http://rdf.freebase.com/ns/m.01p369>", query)
        self.assertIn("government.politician.government_positions_held", query)
        self.assertIn("government.government_position_held.from", query)
        self.assertIn("?intermediate", query)

    def test_builds_candidate_constrained_official_query(self):
        sparql = (
            "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
            "SELECT ?x WHERE { FILTER (!isLiteral(?x) OR lang(?x) = '') "
            "FILTER (?to > ?start)?x ns:film.film.netflix_id ?num . } "
            "ORDER BY DESC(xsd:datetime(?num)) LIMIT 1"
        )

        query = build_candidate_constrained_query(sparql, ("m.a", "m.b"))

        self.assertIn("PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>", query)
        self.assertIn("VALUES ?x { <http://rdf.freebase.com/ns/m.a> <http://rdf.freebase.com/ns/m.b> }", query)
        self.assertIn("||", query)
        self.assertNotIn(" OR ", query)
        self.assertIn(") .\n?x ns:film.film.netflix_id", query)
        self.assertIn("xsd:dateTime", query)
        self.assertNotIn("xsd:datetime", query)

    def test_selects_comparative_candidates_with_any_qualifying_value(self):
        candidates = (
            CandidateValues("A", ("8", "12")),
            CandidateValues("B", ("7",)),
        )

        result = evaluate_candidates("gt", "10", "integer", candidates, ("A",))

        self.assertEqual(result.selected_names, ("A",))
        self.assertTrue(result.oracle_hit)

    def test_selects_superlative_candidate_by_earliest_date(self):
        candidates = (
            CandidateValues("A", ("2000-01-01",)),
            CandidateValues("B", ("1990-05-02", "1995-01-01")),
        )

        result = evaluate_candidates("argmin", None, None, candidates, ("B",))

        self.assertEqual(result.selected_names, ("B",))
        self.assertTrue(result.oracle_hit)

    def test_maps_strict_query_mid_back_to_candidate_name(self):
        samples = [{
            "sample_index": 7,
            "id": "q7",
            "question": "Which candidate is largest?",
            "compositionality_type": "superlative",
            "operator": "argmax",
            "threshold": None,
            "threshold_datatype": None,
            "attribute_chain": ["rel.value"],
            "candidates": [
                {"name": "A", "mid_entity_ids": [1], "mids": ["m.a"]},
                {"name": "B", "mid_entity_ids": [2], "mids": ["m.b"]},
            ],
        }]

        audits = build_strict_audits(samples, {7: ("B",)}, {7: ("m.b",)})

        self.assertEqual(audits[0].selected_names, ("B",))
        self.assertTrue(audits[0].oracle_hit)


if __name__ == "__main__":
    unittest.main()
