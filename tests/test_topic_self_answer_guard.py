"""末端 topic 自指守卫:剔除最终答案中 == topic 的实体。

答案=被问实体本身逻辑上不可能成立(测试集 0 gold 反例)。检索层 drop_loopback_paths
只挡住"尾==topic 的路径"这条线,但 topic 仍可能经:
  ① _answer_pair_from_paths 的 head 匹配(LLM 把 topic 名当答案 → 取到路径头=topic)
  ② large_answer_expansion(关系组全 KG 尾里含 topic)
进入答案。本守卫在答案装配末端统一收口,覆盖所有入口。
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kgqa.agent import CheckedBatchAgent
from kgqa.serving.llm.client import GenerateResponse
from kgqa.retrieve.api.client import PathRetrieveResponse
from kgqa.agent.tools import (
    AnswerWithPathsTool,
    PathRetrieveTool,
    RejectedAnswerCheckTool,
)


class FakePathClient:
    def __init__(self, response):
        self.response = response

    def retrieve(self, question, **kwargs):
        return self.response


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, prompt, **kwargs):
        return self.responses.pop(0)


def text_response(text):
    return GenerateResponse(
        text=text, used_adapter=False, tokens_generated=2, elapsed_ms=1.0
    )


def make_response(paths, prediction=None, group_tails=None):
    return PathRetrieveResponse(
        question="what is example about",
        sample_index=0,
        topics=["m.topic"],
        hop=1,
        mmr_reason_paths=paths,
        prediction=prediction or {},
        elapsed_ms=10.0,
        eta=1.0,
        threshold=0.01,
        beam_size=50,
        lambda_val=0.2,
        cache_path="cache.pt",
        group_tails=group_tails or {},
    )


class TopicSelfAnswerGuardTests(unittest.TestCase):
    def test_drops_topic_introduced_by_head_match(self):
        # LLM 把 topic 名 "Topic" 当答案 → head 匹配把 m.topic 取成答案;Bob 是真答案。
        # 守卫应剔除 m.topic,保留 m.b。
        raw_paths = [
            {"path": [["m.topic", "rel.x", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.y", "m.b"]], "log_score": -1.1},
        ]
        entity_map = {"m.topic": "Topic", "m.a": "Alice", "m.b": "Bob"}
        answer_client = FakeLLMClient(
            [
                text_response("Supporting Paths: 1, 2\nAnswer: Topic | Bob"),
                text_response("NONE"),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths)), entity_map=entity_map
            ),
            answer_tool=AnswerWithPathsTool(client=answer_client),
            check_tool=RejectedAnswerCheckTool(client=answer_client),
        )

        result = agent.run("what is example about", "m.topic", batch_size=10)

        self.assertNotIn("m.topic", result.pred_answer_disambiguated_mids)
        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.b"])
        self.assertEqual(result.pred_answer_names, ["Bob"])

    def test_drops_topic_introduced_by_large_answer_expansion(self):
        # 关系组全尾里含 topic(m.topic),expansion 会把它补进答案;守卫应剔除,
        # 且 large_answer_expanded_mids 里也不留 topic。
        raw_paths = [
            {"path": [["m.topic", "rel.c", f"m.{i}"]], "log_score": -1.0}
            for i in "abcdefgh"
        ]
        entity_map = {"m.topic": "Topic", **{f"m.{c}": c.upper() for c in "abcdefgh"}}
        prediction = {f"m.{c}": 0.95 for c in "abcdefgh"}
        prediction["m.topic"] = 0.95  # topic 自身 e_score 高,满足 expansion 门
        group_tails = {"m.topic|rel.c": [f"m.{c}" for c in "abcdefgh"] + ["m.topic"]}
        answer_lines = "Supporting Paths: " + ", ".join(str(i) for i in range(1, 9))
        answer = "Answer: " + " | ".join(c.upper() for c in "abcdefgh")
        answer_client = FakeLLMClient(
            [text_response(f"{answer_lines}\n{answer}"), text_response("NONE")]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(
                    make_response(raw_paths, prediction, group_tails)
                ),
                entity_map=entity_map,
            ),
            answer_tool=AnswerWithPathsTool(client=answer_client),
            check_tool=RejectedAnswerCheckTool(client=answer_client),
        )

        result = agent.run(
            "what is example about",
            "m.topic",
            batch_size=10,
            large_answer_expansion=True,
            expansion_min_answers=2,
        )

        self.assertNotIn("m.topic", result.pred_answer_disambiguated_mids)
        self.assertNotIn("m.topic", result.large_answer_expanded_mids)

    def test_keeps_all_when_topic_is_only_answer(self):
        # LLM 只答了 topic 自己(答错成主体):守卫剔除后答案为空,不应保底留 topic。
        raw_paths = [{"path": [["m.topic", "rel.x", "m.a"]], "log_score": -1.0}]
        entity_map = {"m.topic": "Topic", "m.a": "Alice"}
        answer_client = FakeLLMClient(
            [
                text_response("Supporting Paths: 1\nAnswer: Topic"),
                text_response("NONE"),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths)), entity_map=entity_map
            ),
            answer_tool=AnswerWithPathsTool(client=answer_client),
            check_tool=RejectedAnswerCheckTool(client=answer_client),
        )

        result = agent.run("what is example about", "m.topic", batch_size=10)

        self.assertNotIn("m.topic", result.pred_answer_disambiguated_mids)
        self.assertEqual(result.pred_answer_disambiguated_mids, [])

    def test_guard_can_be_disabled_for_ablation(self):
        # drop_topic_self=False 关闭守卫:topic 应保留(供消融对照)
        raw_paths = [
            {"path": [["m.topic", "rel.x", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.y", "m.b"]], "log_score": -1.1},
        ]
        entity_map = {"m.topic": "Topic", "m.a": "Alice", "m.b": "Bob"}
        answer_client = FakeLLMClient(
            [
                text_response("Supporting Paths: 1, 2\nAnswer: Topic | Bob"),
                text_response("NONE"),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths)), entity_map=entity_map
            ),
            answer_tool=AnswerWithPathsTool(client=answer_client),
            check_tool=RejectedAnswerCheckTool(client=answer_client),
        )

        result = agent.run(
            "what is example about", "m.topic", batch_size=10, drop_topic_self=False
        )

        self.assertIn("m.topic", result.pred_answer_disambiguated_mids)

    def test_non_topic_answers_untouched(self):
        raw_paths = [
            {"path": [["m.topic", "rel.x", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.y", "m.b"]], "log_score": -1.1},
        ]
        entity_map = {"m.topic": "Topic", "m.a": "Alice", "m.b": "Bob"}
        answer_client = FakeLLMClient(
            [
                text_response("Supporting Paths: 1, 2\nAnswer: Alice | Bob"),
                text_response("NONE"),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths)), entity_map=entity_map
            ),
            answer_tool=AnswerWithPathsTool(client=answer_client),
            check_tool=RejectedAnswerCheckTool(client=answer_client),
        )

        result = agent.run("what is example about", "m.topic", batch_size=10)

        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a", "m.b"])
        self.assertEqual(result.pred_answer_names, ["Alice", "Bob"])


if __name__ == "__main__":
    unittest.main()
