import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kgqa.agent.tools import AnswerWithPathsTool, PathRetrieveTool
from kgqa.serving.llm.client import GenerateResponse
from kgqa.retrieve.api.client import PathRetrieveResponse


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


class AgentToolTests(unittest.TestCase):
    def test_path_retrieve_tool_wraps_cached_retrieve_server(self):
        response = PathRetrieveResponse(
            question="[CLS] who was vice president after kennedy died [SEP]",
            sample_index=3,
            topics=["m.0d3k14"],
            hop=2,
            mmr_reason_paths=[
                {"path": [["m.0d3k14", "government.role", "m.0f7fy"]], "log_score": -1.0}
            ],
            prediction={"m.0f7fy": 0.99},
            elapsed_ms=12.5,
            alpha_final=1.0,
            threshold=0.01,
            beam_size=50,
            lambda_val=0.5,
            cache_path="cache.pt",
        )
        tool = PathRetrieveTool(
            client=FakePathClient(response),
            entity_map={
                "m.0d3k14": "John F. Kennedy",
                "m.0f7fy": "Lyndon B. Johnson",
            },
        )

        result = tool(
            "who was vice president after kennedy died",
            "m.0d3k14",
            sample_index=3,
        )

        self.assertEqual(tool.client.calls[0][1]["sample_index"], 3)
        self.assertEqual(tool.client.calls[0][1]["topic_entities"], ["m.0d3k14"])
        self.assertEqual(result.topic_mid, "m.0d3k14")
        self.assertEqual(result.raw_topics, ["m.0d3k14"])
        self.assertEqual(result.named_topics, ["John F. Kennedy"])
        self.assertEqual(result.raw_mmr_reason_paths[0]["path"][0][2], "m.0f7fy")
        self.assertEqual(result.named_mmr_reason_paths[0]["path"][0][2], "Lyndon B. Johnson")
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
            [
                {
                    "path": [
                        [
                            "John F. Kennedy",
                            "government.us_vice_president.to_president_reverse",
                            "Lyndon B. Johnson",
                        ]
                    ],
                    "log_score": -1.0,
                }
            ],
        )

        self.assertTrue(result.format_ok)
        self.assertEqual(result.cited_path_indices, [1])
        self.assertEqual(result.answer_names, ["Lyndon B. Johnson"])
        self.assertIn("Question: who was vice president after kennedy died", result.prompt)
        self.assertIn("Reasoning Paths:", result.prompt)
        self.assertIn(
            "1: John F. Kennedy -> government.us_vice_president.to_president_reverse -> Lyndon B. Johnson",
            result.prompt,
        )
        self.assertNotIn("<- [government.us_vice_president.to_president] -", result.prompt)


if __name__ == "__main__":
    unittest.main()
