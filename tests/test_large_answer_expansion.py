import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.agent import CheckedBatchWebQAgent
from oh_my_agent.cli.eval_checked_batch_agent import build_parser
from oh_my_agent.llm_server.client import GenerateResponse
from oh_my_agent.path_retrieve_server.client import PathRetrieveResponse
from oh_my_agent.tools import (
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


def text_response(text, tokens=2):
    return GenerateResponse(
        text=text, used_adapter=False, tokens_generated=tokens, elapsed_ms=1.0
    )


def make_response(question, paths, prediction=None, group_tails=None):
    return PathRetrieveResponse(
        question=question,
        sample_index=0,
        topics=["m.topic"],
        hop=1,
        mmr_reason_paths=paths,
        prediction=prediction or {},
        elapsed_ms=10.0,
        method="tail_blend",
        alpha_final=1.0,
        threshold=0.01,
        beam_size=50,
        lambda_val=0.5,
        cache_path="cache.pt",
        group_tails=group_tails or {},
    )


ENTITY_MAP = {
    "m.topic": "Topic",
    "m.a": "Alpha",
    "m.b": "Beta",
    "m.c": "Gamma",
    "m.d": "Delta",
}

RAW_PATHS = [
    {"path": [["m.topic", "rel.contains", "m.a"]], "log_score": -1.0},
    {"path": [["m.topic", "rel.contains", "m.b"]], "log_score": -1.1},
]

KG_GROUP_TAILS = {"m.topic|rel.contains": ["m.a", "m.b", "m.c", "m.d"]}

TWO_GROUP_PATHS = [
    {"path": [["m.topic", "rel.contains", "m.a"]], "log_score": -1.0},
    {"path": [["m.topic", "rel.contains", "m.b"]], "log_score": -1.1},
    {"path": [["m.topic", "rel.member", "m.d"]], "log_score": -1.2},
]

TWO_GROUP_KG = {
    "m.topic|rel.contains": ["m.a", "m.b", "m.c"],
    "m.topic|rel.member": ["m.d", "m.e"],
}


def build_agent(question, prediction):
    path_client = FakePathClient(make_response(question, RAW_PATHS, prediction))
    answer_client = FakeLLMClient(
        [
            text_response("Supporting Paths: 1, 2\nAnswer: Alpha | Beta"),
            text_response("NONE"),
        ]
    )
    return CheckedBatchWebQAgent(
        path_tool=PathRetrieveTool(client=path_client, entity_map=ENTITY_MAP),
        answer_tool=AnswerWithPathsTool(client=answer_client),
        check_tool=RejectedAnswerCheckTool(client=answer_client),
    )


class LargeAnswerExpansionTests(unittest.TestCase):
    def test_expands_winning_group_with_prediction_gated_kg_tails(self):
        # m.c 在 prediction 内被补进; m.d 不在 prediction 内被挡掉
        prediction = {"m.a": 1.0, "m.b": 0.9, "m.c": 0.8}
        agent = build_agent("what countries are in example region", prediction)

        result = agent.run(
            "what countries are in example region",
            "m.topic",
            batch_size=10,
            large_answer_expansion=True,
            kg_group_tails=KG_GROUP_TAILS,
            expansion_min_answers=2,
        )

        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a", "m.b", "m.c"])
        self.assertEqual(result.pred_answer_names, ["Alpha", "Beta", "Gamma"])
        self.assertEqual(result.large_answer_expanded_mids, ["m.c"])

    def test_expands_multiple_groups_up_to_top_groups(self):
        # rel.contains(2 答案) 与 rel.member(1 答案) 都支撑最终答案:
        # top_groups=2 时两组都展开, m.c 与 m.e 均被补进
        prediction = {"m.a": 1.0, "m.b": 0.9, "m.c": 0.8, "m.d": 0.7, "m.e": 0.6}
        path_client = FakePathClient(
            make_response("what countries are in example region", TWO_GROUP_PATHS, prediction)
        )
        entity_map = dict(ENTITY_MAP, **{"m.e": "Epsilon"})
        answer_client = FakeLLMClient(
            [
                text_response("Supporting Paths: 1, 2, 3\nAnswer: Alpha | Beta | Delta"),
                text_response("NONE"),
            ]
        )
        agent = CheckedBatchWebQAgent(
            path_tool=PathRetrieveTool(client=path_client, entity_map=entity_map),
            answer_tool=AnswerWithPathsTool(client=answer_client),
            check_tool=RejectedAnswerCheckTool(client=answer_client),
        )

        result = agent.run(
            "what countries are in example region",
            "m.topic",
            batch_size=10,
            large_answer_expansion=True,
            kg_group_tails=TWO_GROUP_KG,
            expansion_min_answers=2,
            expansion_top_groups=2,
        )

        self.assertEqual(
            result.pred_answer_disambiguated_mids,
            ["m.a", "m.b", "m.d", "m.c", "m.e"],
        )
        self.assertEqual(sorted(result.large_answer_expanded_mids), ["m.c", "m.e"])

    def test_skips_questions_with_selective_constraint_words(self):
        prediction = {"m.a": 1.0, "m.b": 0.9, "m.c": 0.8}
        agent = build_agent("what was the first country in example", prediction)

        result = agent.run(
            "what was the first country in example",
            "m.topic",
            batch_size=10,
            large_answer_expansion=True,
            kg_group_tails=KG_GROUP_TAILS,
            expansion_min_answers=2,
        )

        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a", "m.b"])
        self.assertEqual(result.large_answer_expanded_mids, [])

    def test_skips_when_answer_count_below_threshold(self):
        prediction = {"m.a": 1.0, "m.b": 0.9, "m.c": 0.8}
        agent = build_agent("what countries are in example region", prediction)

        result = agent.run(
            "what countries are in example region",
            "m.topic",
            batch_size=10,
            large_answer_expansion=True,
            kg_group_tails=KG_GROUP_TAILS,
            expansion_min_answers=3,
        )

        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a", "m.b"])
        self.assertEqual(result.large_answer_expanded_mids, [])

    def test_noop_without_kg_entry_for_winning_group(self):
        prediction = {"m.a": 1.0, "m.b": 0.9, "m.c": 0.8}
        agent = build_agent("what countries are in example region", prediction)

        result = agent.run(
            "what countries are in example region",
            "m.topic",
            batch_size=10,
            large_answer_expansion=True,
            kg_group_tails={"m.other|rel.x": ["m.z"]},
            expansion_min_answers=2,
        )

        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a", "m.b"])
        self.assertEqual(result.large_answer_expanded_mids, [])

    def test_expansion_uses_online_group_tails_from_retrieval(self):
        # retrieval 自带在线 group_tails(已 prediction 过滤);agent 不传文件 kg_group_tails,
        # expansion 应直接用在线结果补出 m.c
        prediction = {"m.a": 1.0, "m.b": 0.9, "m.c": 0.8}
        response = make_response(
            "what countries are in example region", RAW_PATHS, prediction,
            group_tails={"m.topic|rel.contains": ["m.a", "m.b", "m.c"]},
        )
        answer_client = FakeLLMClient(
            [
                text_response("Supporting Paths: 1, 2\nAnswer: Alpha | Beta"),
                text_response("NONE"),
            ]
        )
        agent = CheckedBatchWebQAgent(
            path_tool=PathRetrieveTool(client=FakePathClient(response), entity_map=ENTITY_MAP),
            answer_tool=AnswerWithPathsTool(client=answer_client),
            check_tool=RejectedAnswerCheckTool(client=answer_client),
        )

        result = agent.run(
            "what countries are in example region",
            "m.topic",
            batch_size=10,
            large_answer_expansion=True,
            expansion_min_answers=2,
        )

        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a", "m.b", "m.c"])
        self.assertEqual(result.large_answer_expanded_mids, ["m.c"])

    def test_online_group_tails_takes_precedence_over_file(self):
        # 同时存在在线 group_tails(含 m.c)与文件 kg_group_tails(含 m.d):应优先在线 → 补 m.c
        prediction = {"m.a": 1.0, "m.b": 0.9, "m.c": 0.8, "m.d": 0.7}
        response = make_response(
            "what countries are in example region", RAW_PATHS, prediction,
            group_tails={"m.topic|rel.contains": ["m.a", "m.b", "m.c"]},
        )
        answer_client = FakeLLMClient(
            [
                text_response("Supporting Paths: 1, 2\nAnswer: Alpha | Beta"),
                text_response("NONE"),
            ]
        )
        agent = CheckedBatchWebQAgent(
            path_tool=PathRetrieveTool(client=FakePathClient(response), entity_map=ENTITY_MAP),
            answer_tool=AnswerWithPathsTool(client=answer_client),
            check_tool=RejectedAnswerCheckTool(client=answer_client),
        )

        result = agent.run(
            "what countries are in example region",
            "m.topic",
            batch_size=10,
            large_answer_expansion=True,
            kg_group_tails={"m.topic|rel.contains": ["m.a", "m.b", "m.d"]},
            expansion_min_answers=2,
        )

        self.assertEqual(result.large_answer_expanded_mids, ["m.c"])

    def test_disabled_by_default(self):
        prediction = {"m.a": 1.0, "m.b": 0.9, "m.c": 0.8}
        agent = build_agent("what countries are in example region", prediction)

        result = agent.run(
            "what countries are in example region",
            "m.topic",
            batch_size=10,
            kg_group_tails=KG_GROUP_TAILS,
            expansion_min_answers=2,
        )

        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a", "m.b"])
        self.assertEqual(result.large_answer_expanded_mids, [])


class CliFlagTests(unittest.TestCase):
    def test_parser_accepts_large_answer_expansion_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--large_answer_expansion",
                "--kg_group_tails_file",
                "tails.json",
                "--expansion_min_answers",
                "5",
                "--expansion_top_groups",
                "2",
            ]
        )
        self.assertTrue(args.large_answer_expansion)
        self.assertEqual(args.kg_group_tails_file, "tails.json")
        self.assertEqual(args.expansion_min_answers, 5)
        self.assertEqual(args.expansion_top_groups, 2)

    def test_parser_accepts_check_use_adapter(self):
        parser = build_parser()
        self.assertFalse(parser.parse_args([]).check_use_adapter)
        self.assertTrue(parser.parse_args(["--check_use_adapter"]).check_use_adapter)

    def test_kg_group_tails_file_optional_with_online_source(self):
        # 在线 group_tails 可用后,expansion 不再强制要 sidecar 文件:无文件返回 None(不报错)
        from oh_my_agent.cli.eval_checked_batch_agent import load_file_kg_group_tails

        self.assertIsNone(load_file_kg_group_tails(True, ""))
        self.assertIsNone(load_file_kg_group_tails(False, "ignored.json"))

    def test_kg_group_tails_file_loaded_when_provided(self):
        import json
        import os
        import tempfile

        from oh_my_agent.cli.eval_checked_batch_agent import load_file_kg_group_tails

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"m.t|rel.x": ["m.a"]}, handle)
            path = handle.name
        try:
            self.assertEqual(load_file_kg_group_tails(True, path), {"m.t|rel.x": ["m.a"]})
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
