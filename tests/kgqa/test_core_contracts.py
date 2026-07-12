"""core 的跨域契约与辅助函数。"""
from __future__ import annotations

import unittest


class TestCoreContracts(unittest.TestCase):
    def test_contract_types_are_available_from_core(self):
        from kgqa.core.contracts import (
            CacheMeta,
            MetricSpec,
            QASample,
            ReasonPath,
            RetrieveResult,
            SampleScore,
            ScoreBundle,
            ScoreLoader,
            ScoreProducer,
        )
        self.assertEqual(QASample.__module__, "kgqa.core.contracts")
        self.assertEqual(ReasonPath.__module__, "kgqa.core.contracts")
        self.assertEqual(RetrieveResult.__module__, "kgqa.core.contracts")
        self.assertEqual(MetricSpec.__module__, "kgqa.core.contracts")
        self.assertEqual(SampleScore.__module__, "kgqa.core.contracts")
        self.assertEqual(CacheMeta.__module__, "kgqa.core.contracts")
        self.assertEqual(ScoreBundle.__module__, "kgqa.core.contracts")
        self.assertEqual(ScoreLoader.__module__, "kgqa.core.contracts")
        self.assertEqual(ScoreProducer.__module__, "kgqa.core.contracts")

    def test_shared_helpers_are_reexported_from_agent_compatibility_paths(self):
        from kgqa.agent.common.entity_mapping import load_entity_map as legacy_load_entity_map
        from kgqa.agent.common.metrics import compute_answer_metrics as legacy_compute_answer_metrics
        from kgqa.agent.common.qa_data import parse_webqsp_qa_line as legacy_parse_qa_line
        from kgqa.core.answer_metrics import compute_answer_metrics
        from kgqa.core.entity_map import load_entity_map
        from kgqa.core.qa_formats import parse_webqsp_qa_line

        self.assertIs(load_entity_map, legacy_load_entity_map)
        self.assertIs(compute_answer_metrics, legacy_compute_answer_metrics)
        self.assertIs(parse_webqsp_qa_line, legacy_parse_qa_line)

    def test_core_qa_and_answer_helpers_preserve_behavior(self):
        from kgqa.core.answer_metrics import compute_answer_metrics
        from kgqa.core.qa_formats import parse_webqsp_qa_line

        sample = parse_webqsp_qa_line("what is this [m.topic]\tm.answer|m.answer\n")
        self.assertEqual(sample.question, "what is this")
        self.assertEqual(sample.topic_mid, "m.topic")
        self.assertEqual(sample.gold_mids, ["m.answer"])
        self.assertEqual(compute_answer_metrics(["M.Answer"], ["m.answer"])["hit1"], 1)


if __name__ == "__main__":
    unittest.main()
