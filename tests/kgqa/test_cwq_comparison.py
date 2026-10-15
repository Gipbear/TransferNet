"""CWQ 比较约束与样本内 MID 对齐测试。"""
from __future__ import annotations

import unittest

from kgqa.retrieve.cwq_comparison import align_sample_entity_ids, extract_comparison_spec


class TestCwqComparison(unittest.TestCase):
    def test_extracts_direct_superlative_attribute(self):
        sparql = """
        SELECT DISTINCT ?x WHERE {
          ?x ns:film.film.netflix_id ?num .
        }
        ORDER BY DESC(xsd:integer(?num)) LIMIT 1
        """

        spec = extract_comparison_spec("superlative", sparql)

        self.assertEqual(spec.operator, "argmax")
        self.assertEqual(spec.attribute_chain, ("film.film.netflix_id",))
        self.assertIsNone(spec.threshold)

    def test_extracts_cvt_comparative_attribute(self):
        sparql = """
        SELECT DISTINCT ?x WHERE {
          ?x ns:education.university.number_of_undergraduates ?c .
          ?c ns:measurement_unit.dated_integer.number ?num .
          FILTER (xsd:integer(?num) > 23799) .
        }
        """

        spec = extract_comparison_spec("comparative", sparql)

        self.assertEqual(spec.operator, "gt")
        self.assertEqual(spec.attribute_chain, (
            "education.university.number_of_undergraduates",
            "measurement_unit.dated_integer.number",
        ))
        self.assertEqual(spec.threshold, "23799")
        self.assertEqual(spec.threshold_datatype, "integer")

    def test_extracts_typed_date_threshold(self):
        sparql = """
        SELECT DISTINCT ?x WHERE {
          ?x ns:government.politician.government_positions_held ?c .
          ?c ns:government.government_position_held.from ?num .
          FILTER (?num < "1998"^^xsd:dateTime) .
        }
        """

        spec = extract_comparison_spec("comparative", sparql)

        self.assertEqual(spec.operator, "lt")
        self.assertEqual(spec.threshold, "1998")
        self.assertEqual(spec.threshold_datatype, "dateTime")

    def test_aligns_ambiguous_global_entity_with_sample_candidates(self):
        rog_triples = [[0, 7, 1], [0, 7, 1]]
        mid_triples = [[10, 7, 11], [10, 7, 12]]

        aligned = align_sample_entity_ids(rog_triples, mid_triples, row_index=1)

        self.assertEqual(aligned, {0: (10,), 1: (11, 12)})


if __name__ == "__main__":
    unittest.main()
