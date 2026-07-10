import json
import os
import tempfile
import unittest

from kgqa.datasets.metaqa import MetaQAAdapter
from kgqa.datasets.registry import get_adapter
from kgqa.types import MetricSpec, QASample


class TestMetaQAAdapter(unittest.TestCase):
    def _write_test_json(self):
        data = [
            {"question": "what does E_S appear in", "topic_entity": "Grégoire Colin",
             "answers": ["Before the Rain"], "hop": 1},
            {"question": "who directed the movies", "topic_entity": "Joe",
             "answers": ["A", "B"], "hop": 3},
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        return path

    def test_load_qa_parses_names_and_hop(self):
        adapter = MetaQAAdapter(input_dir="data/input/MetaQA_KB")
        path = self._write_test_json()
        try:
            samples = adapter.load_qa(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(samples), 2)
        self.assertIsInstance(samples[0], QASample)
        self.assertEqual(samples[0].topic_ids, ["Grégoire Colin"])
        self.assertEqual(samples[0].gold_ids, ["Before the Rain"])
        self.assertEqual(samples[0].hop, 1)
        self.assertEqual(samples[1].hop, 3)
        self.assertEqual(samples[1].gold_ids, ["A", "B"])

    def test_load_qa_limit(self):
        adapter = MetaQAAdapter(input_dir="data/input/MetaQA_KB")
        path = self._write_test_json()
        try:
            samples = adapter.load_qa(path, limit=1)
        finally:
            os.unlink(path)
        self.assertEqual(len(samples), 1)

    def test_entity_name_identity(self):
        adapter = MetaQAAdapter(input_dir="data/input/MetaQA_KB")
        self.assertEqual(adapter.entity_name("Before the Rain"), "Before the Rain")

    def test_metric_spec_name_and_hop(self):
        adapter = MetaQAAdapter(input_dir="data/input/MetaQA_KB")
        spec = adapter.metric_spec()
        self.assertIsInstance(spec, MetricSpec)
        self.assertEqual(spec.gold_key, "name")
        self.assertEqual(spec.group_by, "hop")
        self.assertEqual(adapter.max_hop, 3)

    def test_registry_returns_metaqa(self):
        adapter = get_adapter("metaqa", input_dir="data/input/MetaQA_KB")
        self.assertEqual(adapter.name, "metaqa")


if __name__ == "__main__":
    unittest.main()
