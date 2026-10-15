"""CWQ train/dev 属性链覆盖率测试。"""
from __future__ import annotations

import unittest

from kgqa.retrieve.cwq_attribute_coverage import (
    build_coverage_audit,
    build_official_chain_inventory,
    list_unseen_chains,
    summarize_chain_coverage,
)


class TestCwqAttributeCoverage(unittest.TestCase):
    def test_builds_train_only_and_train_dev_coverage(self):
        train = [{
            "compositionality_type": "comparative",
            "sparql": "?x ns:people.person.height_meters ?num . FILTER (?num > 1.8)",
        }]
        dev = [{
            "compositionality_type": "superlative",
            "sparql": "?x ns:film.film.runtime ?num . ORDER BY DESC(?num)",
        }]
        test = [
            {
                "compositionality_type": "comparative",
                "operator": "gt",
                "attribute_chain": ["people.person.height_meters"],
            },
            {
                "compositionality_type": "superlative",
                "operator": "argmax",
                "attribute_chain": ["film.film.runtime"],
            },
        ]

        audit = build_coverage_audit(train, dev, test)

        self.assertEqual(audit.train_only.overall.covered_sample_count, 1)
        self.assertEqual(audit.train_dev.overall.covered_sample_count, 2)
        self.assertEqual(audit.unseen_train_dev, ())

    def test_builds_inventory_from_comparison_questions_only(self):
        records = [
            {
                "compositionality_type": "comparative",
                "sparql": "?x ns:people.person.height_meters ?num . FILTER (?num > 1.8)",
            },
            {
                "compositionality_type": "superlative",
                "sparql": "?x ns:film.film.runtime ?num . ORDER BY DESC(?num)",
            },
            {"compositionality_type": "composition", "sparql": "?x ns:type.object.name ?name ."},
        ]

        inventory = build_official_chain_inventory(records)

        self.assertEqual(inventory.sample_count, 3)
        self.assertEqual(inventory.comparison_sample_count, 2)
        self.assertEqual(len(inventory.chains), 2)

    def test_summarizes_exact_chain_coverage_by_question_type(self):
        known = {("people.person.height_meters",)}
        samples = [
            {
                "compositionality_type": "comparative",
                "operator": "gt",
                "attribute_chain": ["people.person.height_meters"],
            },
            {
                "compositionality_type": "comparative",
                "operator": "lt",
                "attribute_chain": ["people.person.date_of_birth"],
            },
            {
                "compositionality_type": "superlative",
                "operator": "argmax",
                "attribute_chain": ["people.person.height_meters"],
            },
        ]

        summary = summarize_chain_coverage(known, samples)

        self.assertEqual(summary.overall.covered_sample_count, 2)
        self.assertEqual(summary.overall.covered_unique_chain_count, 1)
        self.assertEqual(summary.comparative.covered_sample_count, 1)
        self.assertEqual(summary.superlative.covered_sample_count, 1)

    def test_lists_unseen_chains_by_descending_frequency(self):
        known = {("people.person.height_meters",)}
        samples = [
            {
                "compositionality_type": "comparative",
                "operator": "lt",
                "attribute_chain": ["people.person.date_of_birth"],
            },
            {
                "compositionality_type": "superlative",
                "operator": "argmin",
                "attribute_chain": ["people.person.date_of_birth"],
            },
            {
                "compositionality_type": "superlative",
                "operator": "argmax",
                "attribute_chain": ["film.film.runtime"],
            },
        ]

        unseen = list_unseen_chains(known, samples)

        self.assertEqual(unseen[0].chain, ("people.person.date_of_birth",))
        self.assertEqual(unseen[0].count, 2)
        self.assertEqual(unseen[1].count, 1)


if __name__ == "__main__":
    unittest.main()
