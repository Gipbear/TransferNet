import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.llm_server.client import GenerateResponse
from oh_my_agent.path_server.client import PathRetrievalResponse
from oh_my_agent.tools import AnswerWithPathsTool, PathRetrievalTool


class FakePathClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def retrieve(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return self.response


class FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response


class SimpleAgentToolTests(unittest.TestCase):
    def test_path_retrieval_tool_returns_raw_and_named_views(self):
        response = PathRetrievalResponse(
            question="who was vice president after kennedy died",
            topics=["John F. Kennedy"],
            hop=2,
            mmr_reason_paths=[
                {"path": [["John F. Kennedy", "government.role", "Lyndon B. Johnson"]], "log_score": -1.0}
            ],
            prediction={"Lyndon B. Johnson": 0.99},
            elapsed_ms=12.5,
            raw_topics=["m.0d3k14"],
            raw_mmr_reason_paths=[
                {"path": [["m.0d3k14", "government.role", "m.0f7fy"]], "log_score": -1.0}
            ],
            raw_prediction={"m.0f7fy": 0.99},
        )
        tool = PathRetrievalTool(
            client=FakePathClient(response),
            entity_map={
                "m.0d3k14": "John F. Kennedy",
                "m.0f7fy": "Lyndon B. Johnson",
            },
        )

        result = tool("who was vice president after kennedy died", "m.0d3k14")

        self.assertEqual(result.raw_topics, ["m.0d3k14"])
        self.assertEqual(result.named_topics, ["John F. Kennedy"])
        self.assertEqual(result.raw_mmr_reason_paths[0]["path"][0][2], "m.0f7fy")
        self.assertEqual(result.named_mmr_reason_paths[0]["path"][0][2], "Lyndon B. Johnson")
        self.assertEqual(result.raw_prediction, {"m.0f7fy": 0.99})
        self.assertEqual(result.named_prediction, {"Lyndon B. Johnson": 0.99})

    def test_answer_with_paths_tool_builds_prompt_and_parses_v2_output(self):
        tool = AnswerWithPathsTool(
            client=FakeLLMClient(
                GenerateResponse(
                    text="Supporting Paths: 1\nAnswer: Lyndon B. Johnson",
                    used_adapter=True,
                    tokens_generated=7,
                    elapsed_ms=6.1,
                )
            )
        )

        result = tool(
            "who was vice president after kennedy died",
            [{"path": [["John F. Kennedy", "government.role", "Lyndon B. Johnson"]], "log_score": -1.0}],
        )

        self.assertTrue(result.format_ok)
        self.assertEqual(result.cited_path_indices, [1])
        self.assertEqual(result.answer_names, ["Lyndon B. Johnson"])
        self.assertIn("Question: who was vice president after kennedy died", result.prompt)
        self.assertIn("Reasoning Paths:", result.prompt)


if __name__ == "__main__":
    unittest.main()
