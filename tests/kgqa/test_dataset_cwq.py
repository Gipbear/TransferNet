import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from kgqa.core.contracts import MetricSpec, QASample


def _write_test_jsonl():
    rows = [
        {"id": "WebQTest-1", "question": "who is A?",
         "answers": [{"kb_id": "m.0b", "text": "B"}], "entities": [0],
         "subgraph": {"tuples": [[0, 0, 1]], "entities": [0, 1]}},
        {"id": "WebQTest-2", "question": "empty subgraph, should skip",
         "answers": [{"kb_id": "m.0c", "text": "C"}], "entities": [2],
         "subgraph": {"tuples": [], "entities": []}},
        {"id": "WebQTest-3", "question": "who is D?",
         "answers": [{"kb_id": "m.0d", "text": "D"}, {"kb_id": "m.0e", "text": "E"}],
         "entities": [3], "subgraph": {"tuples": [[3, 1, 4]], "entities": [3, 4]}},
    ]
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


class TestCWQAdapter(unittest.TestCase):
    def _adapter(self):
        from kgqa.retrieve.datasets.cwq import CWQAdapter
        return CWQAdapter(input_dir="data/input/CWQ")

    def test_load_qa_parses_and_skips_empty_subgraph(self):
        path = _write_test_jsonl()
        try:
            samples = self._adapter().load_qa(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(samples), 2)  # 空子图样本被跳过（对齐 CompWebQ DataLoader）
        self.assertIsInstance(samples[0], QASample)
        self.assertEqual(samples[0].topic_ids, [0])
        self.assertEqual(samples[0].gold_ids, ["m.0b"])
        self.assertEqual(samples[1].gold_ids, ["m.0d", "m.0e"])
        self.assertEqual(samples[1].sample_index, 1)

    def test_load_qa_limit(self):
        path = _write_test_jsonl()
        try:
            samples = self._adapter().load_qa(path, limit=1)
        finally:
            os.unlink(path)
        self.assertEqual(len(samples), 1)

    def test_kg_edge_source_builds_per_sample(self):
        adapter = self._adapter()
        kg = adapter.kg_edge_source(SimpleNamespace(triples=[[0, 0, 1], [1, 1, 2]]))
        self.assertEqual(kg.neighbors(0), [(0, 1)])
        self.assertEqual(kg.neighbors(1), [(1, 2)])

    def test_kg_edge_source_requires_sample_triples(self):
        adapter = self._adapter()
        with self.assertRaises(ValueError):
            adapter.kg_edge_source(None)
        with self.assertRaises(ValueError):
            adapter.kg_edge_source(SimpleNamespace(triples=None))

    def test_metric_spec_mid_no_group(self):
        adapter = self._adapter()
        spec = adapter.metric_spec()
        self.assertIsInstance(spec, MetricSpec)
        self.assertEqual(spec.gold_key, "mid")
        self.assertIsNone(spec.group_by)
        self.assertEqual(adapter.max_hop, 2)
        self.assertEqual(adapter.entity_name("m.0b"), "m.0b")

    def test_registry_returns_cwq(self):
        from kgqa.retrieve.datasets.registry import get_adapter
        adapter = get_adapter("cwq", input_dir="data/input/CWQ")
        self.assertEqual(adapter.name, "cwq")


if __name__ == "__main__":
    unittest.main()
