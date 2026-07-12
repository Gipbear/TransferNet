import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kgqa.agent import CheckedBatchAgent
from kgqa.llm_server.client import GenerateResponse
from kgqa.server.client import PathRetrieveResponse
from kgqa.agent.tools import (
    AnswerWithPathsTool,
    PathRetrieveTool,
    RejectedAnswerCheckTool,
)


class FakePathClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def retrieve(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return self.response


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.responses.pop(0)


def text_response(text, tokens=2):
    return GenerateResponse(
        text=text, used_adapter=False, tokens_generated=tokens, elapsed_ms=1.0
    )


def make_response(paths, prediction=None):
    return PathRetrieveResponse(
        question="who are the members of example",
        sample_index=0,
        topics=["m.topic"],
        hop=1,
        mmr_reason_paths=paths,
        prediction=prediction or {},
        elapsed_ms=10.0,
        alpha_final=1.0,
        threshold=0.01,
        beam_size=50,
        lambda_val=0.5,
        cache_path="cache.pt",
    )


class AgentHopFilterTests(unittest.TestCase):
    def test_hop_filter_drops_answers_only_from_mismatched_hop_groups(self):
        # 检索 hop=1, m.b 仅由 2-hop 链支撑应被过滤, m.a 保留
        raw_paths = [
            {"path": [["m.topic", "rel.member", "m.a"]], "log_score": -1.0},
            {
                "path": [["m.topic", "rel.x", "m.mid"], ["m.mid", "rel.y", "m.b"]],
                "log_score": -1.5,
            },
        ]
        entity_map = {"m.topic": "Topic", "m.mid": "Mid", "m.a": "Alice", "m.b": "Bob"}
        path_client = FakePathClient(make_response(raw_paths))
        answer_client = FakeLLMClient(
            [
                text_response("Supporting Paths: 1, 2\nAnswer: Alice | Bob"),
                text_response("NONE"),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(client=path_client, entity_map=entity_map),
            answer_tool=AnswerWithPathsTool(client=answer_client),
            check_tool=RejectedAnswerCheckTool(client=answer_client),
        )

        result = agent.run(
            "who are the members of example", "m.topic", batch_size=10, hop_filter=True
        )

        self.assertEqual(result.pred_answer_names, ["Alice"])
        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a"])

    def test_hop_filter_never_empties_answers(self):
        raw_paths = [
            {
                "path": [["m.topic", "rel.x", "m.mid"], ["m.mid", "rel.y", "m.b"]],
                "log_score": -1.5,
            },
        ]
        entity_map = {"m.topic": "Topic", "m.mid": "Mid", "m.b": "Bob"}
        path_client = FakePathClient(make_response(raw_paths))
        answer_client = FakeLLMClient(
            [
                text_response("Supporting Paths: 1\nAnswer: Bob"),
                text_response("NONE"),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(client=path_client, entity_map=entity_map),
            answer_tool=AnswerWithPathsTool(client=answer_client),
            check_tool=RejectedAnswerCheckTool(client=answer_client),
        )

        result = agent.run(
            "who are the members of example", "m.topic", batch_size=10, hop_filter=True
        )

        self.assertEqual(result.pred_answer_names, ["Bob"])

    def test_hop_filter_keeps_all_answers_when_none_match_hop(self):
        # 检索 hop=1 但所有答案都只有 2-hop 支撑:hop 信号此时不可信,
        # 应放弃过滤保留全部答案,而不是只保底留第一个
        raw_paths = [
            {
                "path": [["m.topic", "rel.x", "m.mid"], ["m.mid", "rel.y", "m.b"]],
                "log_score": -1.5,
            },
            {
                "path": [["m.topic", "rel.x", "m.mid"], ["m.mid", "rel.z", "m.c"]],
                "log_score": -2.0,
            },
        ]
        entity_map = {
            "m.topic": "Topic", "m.mid": "Mid", "m.b": "Bob", "m.c": "Carol",
        }
        path_client = FakePathClient(make_response(raw_paths))
        answer_client = FakeLLMClient(
            [
                text_response("Supporting Paths: 1, 2\nAnswer: Bob | Carol"),
                text_response("NONE"),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(client=path_client, entity_map=entity_map),
            answer_tool=AnswerWithPathsTool(client=answer_client),
            check_tool=RejectedAnswerCheckTool(client=answer_client),
        )

        result = agent.run(
            "who are the members of example", "m.topic", batch_size=10, hop_filter=True
        )

        self.assertEqual(result.pred_answer_names, ["Bob", "Carol"])
        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.b", "m.c"])


if __name__ == "__main__":
    unittest.main()
