"""retrieve 新路径与既有公开 import 路径必须指向同一实现。"""
from __future__ import annotations

import unittest


class TestRetrieveImportCompatibility(unittest.TestCase):
    def test_legacy_domain_modules_reexport_canonical_symbols(self):
        from kgqa.eval.answer_eval import answer_summary as legacy_answer_summary
        from kgqa.kg.global_kg import GlobalKG as LegacyGlobalKG
        from kgqa.retrieve.cache.base import SampleScore
        from kgqa.retrieve.datasets.webqsp import WebQSPAdapter
        from kgqa.retrieve.eval.answer_eval import answer_summary
        from kgqa.retrieve.graph.global_kg import GlobalKG
        from kgqa.scores.base import SampleScore as LegacySampleScore
        from kgqa.datasets.webqsp import WebQSPAdapter as LegacyWebQSPAdapter

        self.assertIs(SampleScore, LegacySampleScore)
        self.assertIs(WebQSPAdapter, LegacyWebQSPAdapter)
        self.assertIs(GlobalKG, LegacyGlobalKG)
        self.assertIs(answer_summary, legacy_answer_summary)

    def test_api_and_cli_legacy_modules_reexport_canonical_symbols(self):
        from kgqa.cli.retrieve import _make_producer as legacy_make_producer
        from kgqa.retrieve.api.client import PathRetrieveClient
        from kgqa.retrieve.cli.retrieve import _make_producer
        from kgqa.server.client import PathRetrieveClient as LegacyPathRetrieveClient

        self.assertIs(_make_producer, legacy_make_producer)
        self.assertIs(PathRetrieveClient, LegacyPathRetrieveClient)

    def test_canonical_registry_keeps_planned_dataset_scope(self):
        from kgqa.retrieve.datasets.registry import get_adapter

        self.assertEqual(get_adapter("webqsp").name, "webqsp")
        self.assertEqual(get_adapter("metaqa").name, "metaqa")
        self.assertEqual(get_adapter("cwq").name, "cwq")


if __name__ == "__main__":
    unittest.main()
