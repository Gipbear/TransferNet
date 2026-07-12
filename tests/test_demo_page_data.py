"""demo_page 题库索引与轨迹索引的单元测试。"""
import json
import tempfile
import unittest
from pathlib import Path

from kgqa.agent.common.qa_data import WebQSPQASample
from kgqa.agent.web.data import QuestionIndex, load_trace_index


def _sample(q: str) -> WebQSPQASample:
    return WebQSPQASample(question_raw=q, question=q, topic_mid="m.0x", gold_mids=["m.0g"])


class TestQuestionIndex(unittest.TestCase):
    def setUp(self):
        self.index = QuestionIndex([
            _sample("what does jamaican people speak"),
            _sample("who is the president of france"),
            _sample("what language do jamaicans use"),
        ])

    def test_search_substring_case_insensitive(self):
        hits = self.index.search("JAMAICA")
        self.assertEqual([h.sample_index for h in hits], [0, 2])
        self.assertEqual(hits[0].question, "what does jamaican people speak")

    def test_search_empty_query_returns_head(self):
        hits = self.index.search("", limit=2)
        self.assertEqual([h.sample_index for h in hits], [0, 1])

    def test_search_respects_limit(self):
        self.assertEqual(len(self.index.search("w", limit=1)), 1)

    def test_get_returns_sample(self):
        self.assertEqual(self.index.get(1).question, "who is the president of france")
        with self.assertRaises(IndexError):
            self.index.get(99)


class TestLoadTraceIndex(unittest.TestCase):
    def test_index_keyed_by_sample_index(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trace.jsonl"
            p.write_text(
                json.dumps({"sample_index": 0, "question": "a"}) + "\n\n"
                + json.dumps({"sample_index": 7, "question": "b"}) + "\n",
                encoding="utf-8",
            )
            idx = load_trace_index(str(p))
        self.assertEqual(set(idx), {0, 7})
        self.assertEqual(idx[7]["question"], "b")


if __name__ == "__main__":
    unittest.main()
