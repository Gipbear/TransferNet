"""验证旧轨迹补全只读取真实生成缓存，不修改预测答案。"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from kgqa.agent.workflow.trace import CachedTraceReader
from tests.kgqa.test_cwq_workflow_scoring import record


class FakeRenderer:
    system_prompt = "system"

    def render(self, question, paths):
        return question, 10, tuple(paths[:1])


class TestCachedTraceReader(unittest.TestCase):
    def test_trace_enrichment_preserves_answers_and_records_actual_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            row = record(0, ["A"], ["A"])
            row["question"]["paths"] = [{"path": [["a", "r", "A"]]}] * 2
            signature = json.dumps(["model", "adapter", "system", "q0", 256])
            key = hashlib.sha256(signature.encode()).hexdigest()
            (cache / (key + ".json")).write_text(json.dumps(row["initial"]))
            reader = CachedTraceReader(FakeRenderer(), "model", "adapter", 256, cache)
            result = reader.enrich(row)
            self.assertEqual(result["final"]["answers"], ["A"])
            self.assertEqual(len(result["initial_paths"]), 1)
            self.assertEqual(result["call_results"][0]["prompt_tokens"], 10)
            self.assertNotIn("initial_paths", row)

    def test_symbolic_answer_trace_uses_only_displayed_initial_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            row = record(0, ["A"], ["B"], "comparison")
            row["question"]["paths"] = [{"path": [["a", "r", "A"]]}] * 2
            row["final_paths"] = row["question"]["paths"]
            signature = json.dumps(["model", "adapter", "system", "q0", 256])
            key = hashlib.sha256(signature.encode()).hexdigest()
            (cache / (key + ".json")).write_text(json.dumps(row["initial"]))
            result = CachedTraceReader(FakeRenderer(), "model", "adapter", 256, cache).enrich(row)
            self.assertEqual(len(result["final_paths"]), 1)
            self.assertEqual(result["final"]["answers"], ["B"])

    def test_missing_generation_cache_is_not_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = CachedTraceReader(FakeRenderer(), "model", "adapter", 256, Path(directory))
            with self.assertRaises(FileNotFoundError):
                reader.enrich(record(0, [], []))
