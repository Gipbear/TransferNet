import os
import tempfile
import unittest

from kgqa.retrieve.datasets.registry import get_adapter
from kgqa.retrieve.datasets.webqsp import WebQSPAdapter
from kgqa.core.contracts import QASample, MetricSpec


class TestWebQSPAdapter(unittest.TestCase):
    def test_load_qa_parses_topic_and_gold(self):
        adapter = WebQSPAdapter(input_dir="data/input/WebQSP")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write("what is the language spoken in france [m.0f8l9c]\tm.04306rv|m.02bv9\n")
            path = fh.name
        try:
            samples = adapter.load_qa(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(samples), 1)
        s = samples[0]
        self.assertIsInstance(s, QASample)
        self.assertEqual(s.extra["topic_mid"], "m.0f8l9c")
        self.assertEqual(s.gold_ids, ["m.04306rv", "m.02bv9"])

    def test_metric_spec_defaults_mid(self):
        adapter = WebQSPAdapter(input_dir="data/input/WebQSP")
        spec = adapter.metric_spec()
        self.assertIsInstance(spec, MetricSpec)
        self.assertEqual(spec.gold_key, "mid")
        self.assertIsNone(spec.group_by)
        self.assertEqual(adapter.max_hop, 2)

    def test_registry_returns_webqsp(self):
        adapter = get_adapter("webqsp", input_dir="data/input/WebQSP")
        self.assertEqual(adapter.name, "webqsp")


if __name__ == "__main__":
    unittest.main()
