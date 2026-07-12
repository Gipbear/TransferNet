import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kgqa.agent import CheckedBatchAgent
from kgqa.agent.cli import eval_checked_batch as eval_checked_batch_agent
from kgqa.serving.llm.client import GenerateResponse, LLMClient, SiliconFlowLLMClient
from kgqa.retrieve.api.client import PathRetrieveClient, PathRetrieveResponse
from kgqa.agent.tools import AnswerWithPathsTool, PathRetrieveTool, RejectedAnswerCheckTool
from tests.agent_fixtures import FakeLLMClient, FakePathClient


def make_response(paths, prediction=None):
    return PathRetrieveResponse(
        question="where is example from",
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


class CheckedBatchAgentTests(unittest.TestCase):
    def test_batches_continue_on_all_correct_and_accumulate_answers(self):
        raw_paths = [
            {"path": [["m.topic", "rel.a", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.b", "m.b"]], "log_score": -2.0},
            {"path": [["m.topic", "rel.c", "m.c"]], "log_score": -3.0},
            {"path": [["m.topic", "rel.c", "m.c"]], "log_score": -4.0},
        ]
        entity_map = {
            "m.topic": "Topic",
            "m.a": "Answer A",
            "m.b": "Answer B",
            "m.c": "Answer C",
        }
        path_client = FakePathClient(make_response(raw_paths, prediction={"m.c": 0.9}))
        llm_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 1, 2\nAnswer: Answer A | Answer B",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="NONE", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
                GenerateResponse(
                    text="Supporting Paths: 1, 2\nAnswer: Answer A | Answer C",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="NONE", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(client=path_client, entity_map=entity_map),
            answer_tool=AnswerWithPathsTool(client=llm_client),
            check_tool=RejectedAnswerCheckTool(client=llm_client),
        )

        result = agent.run("where is example from", "m.topic", batch_size=2)

        retrieve_kwargs = path_client.calls[0][1]
        self.assertNotIn("method", retrieve_kwargs)
        self.assertEqual(retrieve_kwargs["alpha_final"], 1.0)
        self.assertEqual(retrieve_kwargs["beam_size"], 50)
        self.assertEqual(retrieve_kwargs["lambda_val"], 0.2)
        self.assertEqual(result.stop_reason, "path_exhausted")
        self.assertEqual([item.batch_status for item in result.iterations], ["all_correct", "all_correct"])
        self.assertEqual(result.iterations[1].global_cited_path_indices, [3, 4])
        self.assertEqual(result.final_accepted_path_indices, [1, 2, 3, 4])
        self.assertEqual(result.relation_expanded_path_indices, [])
        self.assertEqual(result.pred_answer_names, ["Answer A", "Answer B", "Answer C"])
        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a", "m.b", "m.c"])
        self.assertEqual(result.checked_paths_count, 4)
        self.assertEqual(result.accepted_paths_count, 4)

    def test_secondary_check_tool_is_used_after_first_batch(self):
        raw_paths = [
            {"path": [["m.topic", "rel.a", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.b", "m.b"]], "log_score": -2.0},
            {"path": [["m.topic", "rel.c", "m.c"]], "log_score": -3.0},
        ]
        entity_map = {
            "m.topic": "Topic",
            "m.a": "Answer A",
            "m.b": "Answer B",
            "m.c": "Answer C",
        }
        answer_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 1\nAnswer: Answer A",
                    used_adapter=True,
                    tokens_generated=4,
                    elapsed_ms=2.0,
                ),
                GenerateResponse(
                    text="Supporting Paths: 1\nAnswer: Answer B",
                    used_adapter=True,
                    tokens_generated=4,
                    elapsed_ms=2.0,
                ),
            ]
        )
        first_check_client = FakeLLMClient(
            [GenerateResponse(text="NONE", used_adapter=False, tokens_generated=1, elapsed_ms=1.0)]
        )
        later_check_client = FakeLLMClient(
            [GenerateResponse(text="1", used_adapter=False, tokens_generated=1, elapsed_ms=1.0)]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths)),
                entity_map=entity_map,
            ),
            answer_tool=AnswerWithPathsTool(client=answer_client),
            check_tool=RejectedAnswerCheckTool(client=first_check_client),
            check_tool_after_first=RejectedAnswerCheckTool(client=later_check_client, reject_policy="strict"),
        )

        result = agent.run("where is example from", "m.topic", batch_size=1)

        self.assertEqual(len(answer_client.calls), 2)
        self.assertEqual(len(first_check_client.calls), 1)
        self.assertEqual(len(later_check_client.calls), 1)
        self.assertEqual(len(result.iterations), 2)
        self.assertEqual(result.iterations[0].path_check["path_evaluations"][0]["raw_output"], "NONE")
        self.assertEqual(result.iterations[1].path_check["path_evaluations"][0]["raw_output"], "1")
        self.assertEqual(result.stop_reason, "all_wrong_after_answer")
        self.assertEqual(result.final_accepted_path_indices, [1])
        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a"])

    def test_can_disable_all_wrong_after_answer_stop_for_full_trace_collection(self):
        raw_paths = [
            {"path": [["m.topic", "rel.a", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.b", "m.b"]], "log_score": -2.0},
            {"path": [["m.topic", "rel.c", "m.c"]], "log_score": -3.0},
        ]
        entity_map = {
            "m.topic": "Topic",
            "m.a": "Answer A",
            "m.b": "Answer B",
            "m.c": "Answer C",
        }
        answer_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 1\nAnswer: Answer A",
                    used_adapter=True,
                    tokens_generated=4,
                    elapsed_ms=2.0,
                ),
                GenerateResponse(
                    text="Supporting Paths: 1\nAnswer: Answer B",
                    used_adapter=True,
                    tokens_generated=4,
                    elapsed_ms=2.0,
                ),
                GenerateResponse(
                    text="Supporting Paths: 1\nAnswer: Answer C",
                    used_adapter=True,
                    tokens_generated=4,
                    elapsed_ms=2.0,
                ),
            ]
        )
        first_check_client = FakeLLMClient(
            [GenerateResponse(text="NONE", used_adapter=False, tokens_generated=1, elapsed_ms=1.0)]
        )
        later_check_client = FakeLLMClient(
            [
                GenerateResponse(text="1", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
                GenerateResponse(text="NONE", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths)),
                entity_map=entity_map,
            ),
            answer_tool=AnswerWithPathsTool(client=answer_client),
            check_tool=RejectedAnswerCheckTool(client=first_check_client),
            check_tool_after_first=RejectedAnswerCheckTool(
                client=later_check_client, reject_policy="strict"
            ),
        )

        result = agent.run(
            "where is example from",
            "m.topic",
            batch_size=1,
            no_all_wrong_after_answer_stop=True,
        )

        self.assertEqual(len(result.iterations), 3)
        self.assertEqual(result.stop_reason, "path_exhausted")
        self.assertEqual(result.final_accepted_path_indices, [1, 3])
        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a", "m.c"])

    def test_score_margin_drops_answers_far_below_top_score(self):
        raw_paths = [
            {"path": [["m.topic", "rel.a", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.b", "m.b"]], "log_score": -2.0},
            {"path": [["m.topic", "rel.c", "m.c"]], "log_score": -6.0},
        ]
        entity_map = {
            "m.topic": "Topic",
            "m.a": "Answer A",
            "m.b": "Answer B",
            "m.c": "Answer C",
        }
        path_client = FakePathClient(
            make_response(raw_paths, prediction={"m.a": 0.9, "m.b": 0.8, "m.c": 0.7})
        )
        llm_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 1, 2, 3\nAnswer: Answer A | Answer B | Answer C",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="NONE", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(client=path_client, entity_map=entity_map),
            answer_tool=AnswerWithPathsTool(client=llm_client),
            check_tool=RejectedAnswerCheckTool(client=llm_client),
        )

        result = agent.run(
            "where is example from", "m.topic", batch_size=3, score_margin=4.0
        )

        # 三条路径均被接受，但 m.c 的 log_score 距顶超过 margin 被滤掉。
        self.assertEqual(result.final_accepted_path_indices, [1, 2, 3])
        self.assertEqual(result.pred_answer_names, ["Answer A", "Answer B"])
        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a", "m.b"])

    def test_mixed_stops_when_accepted_entity_count_does_not_exceed_one_third(self):
        raw_paths = [
            {"path": [["m.topic", "rel.a", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.b", "m.b"]], "log_score": -2.0},
            {"path": [["m.topic", "rel.c", "m.c"]], "log_score": -3.0},
        ]
        llm_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 1, 2, 3\nAnswer: Answer A | Answer B | Answer C",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="2,3", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths)),
                entity_map={
                    "m.topic": "Topic",
                    "m.a": "Answer A",
                    "m.b": "Answer B",
                    "m.c": "Answer C",
                },
            ),
            answer_tool=AnswerWithPathsTool(client=llm_client),
            check_tool=RejectedAnswerCheckTool(client=llm_client),
        )

        result = agent.run("where is example from", "m.topic", batch_size=3)

        self.assertEqual(result.iterations[0].batch_status, "mixed")
        self.assertEqual(result.stop_reason, "mixed")
        self.assertEqual(result.final_accepted_path_indices, [1])

    def test_no_early_stop_continues_past_low_accept_mixed_batch(self):
        raw_paths = [
            {"path": [["m.topic", "rel.a", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.b", "m.b"]], "log_score": -2.0},
            {"path": [["m.topic", "rel.c", "m.c"]], "log_score": -3.0},
            {"path": [["m.topic", "rel.d", "m.d"]], "log_score": -4.0},
        ]
        llm_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 1, 2, 3\nAnswer: Answer A | Answer B | Answer C",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="2,3", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
                GenerateResponse(
                    text="Supporting Paths: 1\nAnswer: Answer D",
                    used_adapter=True,
                    tokens_generated=4,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="NONE", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths)),
                entity_map={
                    "m.topic": "Topic",
                    "m.a": "Answer A",
                    "m.b": "Answer B",
                    "m.c": "Answer C",
                    "m.d": "Answer D",
                },
            ),
            answer_tool=AnswerWithPathsTool(client=llm_client),
            check_tool=RejectedAnswerCheckTool(client=llm_client),
        )

        result = agent.run(
            "where is example from", "m.topic", batch_size=3, no_early_stop=True
        )

        self.assertEqual(len(result.iterations), 2)
        self.assertEqual(result.stop_reason, "path_exhausted")
        self.assertEqual(result.final_accepted_path_indices, [1, 4])

    def test_mixed_continues_when_accepted_entity_count_exceeds_one_third(self):
        raw_paths = [
            {"path": [["m.topic", "rel.a", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.b", "m.b"]], "log_score": -2.0},
            {"path": [["m.topic", "rel.c", "m.c"]], "log_score": -3.0},
            {"path": [["m.topic", "rel.d", "m.d"]], "log_score": -4.0},
            {"path": [["m.topic", "rel.e", "m.e"]], "log_score": -5.0},
            {"path": [["m.topic", "rel.f", "m.f"]], "log_score": -6.0},
        ]
        llm_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 1, 2, 3\nAnswer: Answer A | Answer B | Answer C",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="3", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
                GenerateResponse(
                    text="Supporting Paths: 1, 2, 3\nAnswer: Answer D | Answer E | Answer F",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="1,2,3", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths)),
                entity_map={
                    "m.topic": "Topic",
                    "m.a": "Answer A",
                    "m.b": "Answer B",
                    "m.c": "Answer C",
                    "m.d": "Answer D",
                    "m.e": "Answer E",
                    "m.f": "Answer F",
                },
            ),
            answer_tool=AnswerWithPathsTool(client=llm_client),
            check_tool=RejectedAnswerCheckTool(client=llm_client),
        )

        result = agent.run("where is example from", "m.topic", batch_size=3)

        self.assertEqual([item.batch_status for item in result.iterations], ["mixed", "all_wrong"])
        self.assertEqual(result.stop_reason, "path_exhausted")
        self.assertEqual(result.final_accepted_path_indices, [1, 2])

    def test_mixed_threshold_counts_entities_not_paths(self):
        raw_paths = [
            {"path": [["m.topic", "rel.a1", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.a2", "m.a"]], "log_score": -2.0},
            {"path": [["m.topic", "rel.b", "m.b"]], "log_score": -3.0},
            {"path": [["m.topic", "rel.c", "m.c"]], "log_score": -4.0},
            {"path": [["m.topic", "rel.d", "m.d"]], "log_score": -5.0},
        ]
        llm_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 1, 2, 3, 4\nAnswer: Answer A | Answer B | Answer C",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="2,3", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths)),
                entity_map={
                    "m.topic": "Topic",
                    "m.a": "Answer A",
                    "m.b": "Answer B",
                    "m.c": "Answer C",
                    "m.d": "Answer D",
                },
            ),
            answer_tool=AnswerWithPathsTool(client=llm_client),
            check_tool=RejectedAnswerCheckTool(client=llm_client),
        )

        result = agent.run("where is example from", "m.topic", batch_size=4)

        self.assertEqual(len(result.iterations), 1)
        self.assertEqual(result.iterations[0].batch_status, "mixed")
        self.assertEqual(result.stop_reason, "mixed")
        self.assertEqual(result.final_accepted_path_indices, [1, 2])
        self.assertEqual(result.pred_answer_names, ["Answer A"])

    def test_dedupe_tail_paths_keeps_first_tail_before_batching(self):
        raw_paths = [
            {"path": [["m.topic", "rel.a1", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.a2", "m.a"]], "log_score": -2.0},
            {"path": [["m.topic", "rel.b", "m.b"]], "log_score": -3.0},
        ]
        entity_map = {
            "m.topic": "Topic",
            "m.a": "Answer A",
            "m.b": "Answer B",
        }
        llm_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 1, 2\nAnswer: Answer A | Answer B",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="NONE", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths)),
                entity_map=entity_map,
            ),
            answer_tool=AnswerWithPathsTool(client=llm_client),
            check_tool=RejectedAnswerCheckTool(client=llm_client),
        )

        result = agent.run(
            "where is example from",
            "m.topic",
            batch_size=3,
            dedupe_tail_paths=True,
        )

        self.assertEqual(result.raw_mmr_reason_paths, [raw_paths[0], raw_paths[2]])
        self.assertEqual(
            [path["path"] for path in result.named_mmr_reason_paths],
            [
                [["Topic", "rel.a1", "Answer A"]],
                [["Topic", "rel.b", "Answer B"]],
            ],
        )
        self.assertEqual(result.iterations[0].batch_size, 2)
        self.assertEqual(result.iterations[0].global_cited_path_indices, [1, 2])
        self.assertEqual(result.final_accepted_path_indices, [1, 2])
        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a", "m.b"])

    def test_relation_expansion_only_uses_checked_cited_paths(self):
        raw_paths = [
            {"path": [["m.topic", "rel.location", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.location", "m.b"]], "log_score": -2.0},
            {"path": [["m.topic", "rel.location", "m.c"]], "log_score": -3.0},
        ]
        entity_map = {
            "m.topic": "Topic",
            "m.a": "Answer A",
            "m.b": "Answer B",
            "m.c": "Answer C",
        }
        llm_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 1, 2\nAnswer: Answer A",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="2", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths, prediction={"m.b": 0.9})),
                entity_map=entity_map,
            ),
            answer_tool=AnswerWithPathsTool(client=llm_client),
            check_tool=RejectedAnswerCheckTool(client=llm_client),
        )

        result = agent.run("where is example from", "m.topic", batch_size=3)

        self.assertEqual(result.cited_path_indices, [1, 2])
        self.assertEqual(result.final_accepted_path_indices, [1])
        self.assertEqual(result.relation_expanded_path_indices, [2])
        self.assertEqual(result.pred_answer_names, ["Answer A", "Answer B"])
        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a", "m.b"])

    def test_disable_relation_expansion_keeps_rejected_candidates_out(self):
        raw_paths = [
            {"path": [["m.topic", "rel.location", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.location", "m.b"]], "log_score": -2.0},
        ]
        entity_map = {"m.topic": "Topic", "m.a": "Answer A", "m.b": "Answer B"}
        llm_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 1, 2\nAnswer: Answer A",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="2", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths, prediction={"m.b": 0.9})),
                entity_map=entity_map,
            ),
            answer_tool=AnswerWithPathsTool(client=llm_client),
            check_tool=RejectedAnswerCheckTool(client=llm_client),
        )

        result = agent.run(
            "where is example from",
            "m.topic",
            batch_size=2,
            enable_relation_expansion=False,
        )

        self.assertEqual(result.final_accepted_path_indices, [1])
        self.assertEqual(result.relation_expanded_path_indices, [])
        self.assertEqual(result.pred_answer_names, ["Answer A"])
        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a"])

    def test_relation_expansion_requires_raw_prediction_tail(self):
        raw_paths = [
            {"path": [["m.topic", "rel.location", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.location", "m.b"]], "log_score": -2.0},
            {"path": [["m.topic", "rel.location", "m.c"]], "log_score": -3.0},
        ]
        entity_map = {
            "m.topic": "Topic",
            "m.a": "Answer A",
            "m.b": "Answer B",
            "m.c": "Answer C",
        }
        llm_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 1, 2, 3\nAnswer: Answer A",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="2,3", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths, prediction={"m.c": 0.9})),
                entity_map=entity_map,
            ),
            answer_tool=AnswerWithPathsTool(client=llm_client),
            check_tool=RejectedAnswerCheckTool(client=llm_client),
        )

        result = agent.run("where is example from", "m.topic", batch_size=3)

        self.assertEqual(result.cited_path_indices, [1, 2, 3])
        self.assertEqual(result.final_accepted_path_indices, [1])
        self.assertEqual(result.relation_expanded_path_indices, [3])
        self.assertEqual(result.pred_answer_names, ["Answer A", "Answer C"])
        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a", "m.c"])

    def test_answer_name_can_select_middle_entity_without_gold(self):
        raw_paths = [
            {
                "path": [
                    ["m.topic", "rel.forward", "m.middle"],
                    ["m.middle", "rel.inverse_reverse", "m.tail"],
                ],
                "log_score": -1.0,
            },
        ]
        entity_map = {
            "m.topic": "Topic",
            "m.middle": "Middle Answer",
            "m.tail": "Tail Entity",
        }
        llm_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 1\nAnswer: Middle Answer",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="NONE", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths)),
                entity_map=entity_map,
            ),
            answer_tool=AnswerWithPathsTool(client=llm_client),
            check_tool=RejectedAnswerCheckTool(client=llm_client),
        )

        result = agent.run("where is example from", "m.topic", batch_size=1)

        self.assertEqual(result.final_accepted_path_indices, [1])
        self.assertEqual(result.pred_answer_names, ["Middle Answer"])
        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.middle"])

    def test_all_wrong_and_invalid_citations_continue_until_exhausted(self):
        raw_paths = [
            {"path": [["m.topic", "rel.a", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.b", "m.b"]], "log_score": -2.0},
        ]
        llm_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 99\nAnswer: Missing",
                    used_adapter=True,
                    tokens_generated=4,
                    elapsed_ms=2.0,
                ),
                GenerateResponse(
                    text="Supporting Paths: (none)\nAnswer: (none)",
                    used_adapter=True,
                    tokens_generated=4,
                    elapsed_ms=2.0,
                ),
            ]
        )
        agent = CheckedBatchAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths)),
                entity_map={"m.topic": "Topic", "m.a": "Answer A", "m.b": "Answer B"},
            ),
            answer_tool=AnswerWithPathsTool(client=llm_client),
            check_tool=RejectedAnswerCheckTool(client=llm_client),
        )

        result = agent.run("where is example from", "m.topic", batch_size=1)

        self.assertEqual(result.stop_reason, "path_exhausted")
        self.assertEqual([item.batch_status for item in result.iterations], ["all_wrong", "all_wrong"])
        self.assertEqual(result.final_accepted_path_indices, [])
        self.assertEqual(result.pred_answer_disambiguated_mids, [])
        self.assertEqual(result.checked_paths_count, 0)

    def test_eval_cli_writes_summary_with_batch_diagnostics(self):
        path_response = make_response(
            [
                {"path": [["m.topic", "rel.a", "m.a"]], "log_score": -1.0},
                {"path": [["m.topic", "rel.b", "m.b"]], "log_score": -2.0},
                {"path": [["m.topic", "rel.c", "m.c"]], "log_score": -3.0},
            ]
        )
        responses = [
            GenerateResponse(
                text="Supporting Paths: 1, 2, 3\nAnswer: Answer A | Answer B | Answer C",
                used_adapter=True,
                tokens_generated=8,
                elapsed_ms=5.0,
            ),
            GenerateResponse(text="2,3", used_adapter=False, tokens_generated=3, elapsed_ms=1.0),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            qa_path = tmp_path / "qa.txt"
            qa_path.write_text("where is example from [m.topic]\tm.a\n", encoding="utf-8")
            entity_map_path = tmp_path / "mapped_entities.txt"
            entity_map_path.write_text(
                "m.topic\tTopic\nm.a\tAnswer A\nm.b\tAnswer B\nm.c\tAnswer C\n",
                encoding="utf-8",
            )
            output_dir = tmp_path / "checked"

            with patch.object(PathRetrieveClient, "retrieve", return_value=path_response), patch.object(
                LLMClient, "health", return_value={"status": "ok"}
            ), patch.object(
                PathRetrieveClient, "health", return_value={"status": "ok"}
            ), patch.object(LLMClient, "generate", side_effect=responses):
                exit_code = eval_checked_batch_agent.main(
                    [
                        "--input",
                        str(qa_path),
                        "--output",
                        str(output_dir),
                        "--entity_map",
                        str(entity_map_path),
                    ]
                )

            output_path = output_dir / "checked_batch_eval.jsonl"
            summary_path = output_dir / "checked_batch_eval_summary.json"
            retrieval_path = output_dir / "initial_retrieval.jsonl"
            answer_path = output_dir / "initial_answer.jsonl"
            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
            retrieval_records = [
                json.loads(line) for line in retrieval_path.read_text(encoding="utf-8").splitlines()
            ]
            answer_records = [
                json.loads(line) for line in answer_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(exit_code, 0)
        record_keys = list(records[0])
        self.assertLess(record_keys.index("hit1"), record_keys.index("raw_topics"))
        self.assertLess(record_keys.index("citation_accuracy"), record_keys.index("raw_topics"))
        self.assertEqual(records[0]["final_accepted_path_indices"], [1])
        self.assertEqual(records[0]["iterations"][0]["global_cited_path_indices"], [1, 2, 3])
        self.assertEqual(summary["n"], 1)
        self.assertEqual(summary["hit1"], 1.0)
        self.assertEqual(summary["avg_batches_used"], 1.0)
        self.assertEqual(summary["avg_checked_paths"], 3.0)
        self.assertEqual(summary["avg_accepted_paths"], 1.0)
        self.assertEqual(summary["stop_reason_counts"], {"mixed": 1})
        self.assertEqual(summary["output_dir"], str(output_dir))
        self.assertEqual(summary["initial_retrieval_path"], str(retrieval_path))
        self.assertEqual(summary["initial_answer_path"], str(answer_path))
        self.assertEqual(retrieval_records[0]["mmr_reason_paths"], path_response.mmr_reason_paths)
        self.assertEqual(retrieval_records[0]["golden"], ["m.a"])
        self.assertEqual(answer_records[0]["llm_raw_output"], responses[0].text)
        self.assertEqual(answer_records[0]["llm_pred"], ["Answer A", "Answer B", "Answer C"])
        self.assertEqual(answer_records[0]["cited_indices"], [1, 2, 3])

    def test_eval_cli_resumes_completed_samples_without_calling_services(self):
        path_response = make_response(
            [{"path": [["m.topic", "rel.a", "m.a"]], "log_score": -1.0}]
        )
        responses = [
            GenerateResponse(
                text="Supporting Paths: 1\nAnswer: Answer A",
                used_adapter=True,
                tokens_generated=4,
                elapsed_ms=2.0,
            ),
            GenerateResponse(text="NONE", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            qa_path = tmp_path / "qa.txt"
            qa_path.write_text("where is example from [m.topic]\tm.a\n", encoding="utf-8")
            entity_map_path = tmp_path / "mapped_entities.txt"
            entity_map_path.write_text(
                "m.topic\tTopic\nm.a\tAnswer A\n", encoding="utf-8"
            )
            output_dir = tmp_path / "checked"
            argv = [
                "--input", str(qa_path), "--output", str(output_dir),
                "--entity_map", str(entity_map_path),
            ]

            with patch.object(PathRetrieveClient, "retrieve", return_value=path_response), patch.object(
                LLMClient, "health", return_value={"status": "ok"}
            ), patch.object(
                PathRetrieveClient, "health", return_value={"status": "ok"}
            ), patch.object(LLMClient, "generate", side_effect=responses):
                self.assertEqual(eval_checked_batch_agent.main(argv), 0)

            output_path = output_dir / "checked_batch_eval.jsonl"
            first_output = output_path.read_text(encoding="utf-8")
            with patch.object(PathRetrieveClient, "retrieve", side_effect=AssertionError), patch.object(
                LLMClient, "health", return_value={"status": "ok"}
            ), patch.object(
                PathRetrieveClient, "health", return_value={"status": "ok"}
            ), patch.object(LLMClient, "generate", side_effect=AssertionError):
                self.assertEqual(eval_checked_batch_agent.main(argv), 0)

            summary = json.loads(
                (output_dir / "checked_batch_eval_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(output_path.read_text(encoding="utf-8"), first_output)

        self.assertEqual(summary["n"], 1)

    def test_eval_cli_can_use_siliconflow_for_checks_only(self):
        path_response = make_response(
            [
                {"path": [["m.topic", "rel.a", "m.a"]], "log_score": -1.0},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            qa_path = tmp_path / "qa.txt"
            qa_path.write_text("where is example from [m.topic]\tm.a\n", encoding="utf-8")
            entity_map_path = tmp_path / "mapped_entities.txt"
            entity_map_path.write_text(
                "m.topic\tTopic\nm.a\tAnswer A\n",
                encoding="utf-8",
            )
            output_dir = tmp_path / "checked"

            with patch.dict("os.environ", {"SILICONFLOW_API_KEY": "sf-token"}), patch.object(
                PathRetrieveClient, "retrieve", return_value=path_response
            ), patch.object(
                LLMClient,
                "health",
                return_value={"status": "ok"},
            ), patch.object(
                PathRetrieveClient,
                "health",
                return_value={"status": "ok"},
            ), patch.object(
                LLMClient,
                "generate",
                return_value=GenerateResponse(
                    text="Supporting Paths: 1\nAnswer: Answer A",
                    used_adapter=True,
                    tokens_generated=4,
                    elapsed_ms=2.0,
                ),
            ) as answer_generate, patch.object(
                SiliconFlowLLMClient,
                "generate",
                return_value=GenerateResponse(
                    text="NONE",
                    used_adapter=False,
                    tokens_generated=1,
                    elapsed_ms=1.0,
                ),
            ) as check_generate:
                exit_code = eval_checked_batch_agent.main(
                    [
                        "--input",
                        str(qa_path),
                        "--output",
                        str(output_dir),
                        "--entity_map",
                        str(entity_map_path),
                        "--check_backend",
                        "siliconflow",
                        "--check_siliconflow_model",
                        "Qwen/Qwen3.6-35B-A3B",
                        "--skip_server_check",
                    ]
                )

            summary = json.loads(
                (output_dir / "checked_batch_eval_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(answer_generate.call_count, 1)
        self.assertEqual(check_generate.call_count, 1)
        self.assertEqual(summary["check_backend"], "siliconflow")
        self.assertEqual(summary["check_siliconflow_model"], "Qwen/Qwen3.6-35B-A3B")

    def test_eval_cli_runs_single_sample_index(self):
        path_response = make_response(
            [
                {"path": [["m.topic2", "rel.a", "m.a"]], "log_score": -1.0},
            ]
        )
        responses = [
            GenerateResponse(
                text="Supporting Paths: 1\nAnswer: Answer A",
                used_adapter=True,
                tokens_generated=4,
                elapsed_ms=2.0,
            ),
            GenerateResponse(text="NONE", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            qa_path = tmp_path / "qa.txt"
            qa_path.write_text(
                "first question [m.topic1]\tm.x\n"
                "second question [m.topic2]\tm.a\n",
                encoding="utf-8",
            )
            entity_map_path = tmp_path / "mapped_entities.txt"
            entity_map_path.write_text(
                "m.topic2\tTopic 2\nm.a\tAnswer A\n",
                encoding="utf-8",
            )
            output_dir = tmp_path / "checked"

            with patch.object(PathRetrieveClient, "retrieve", return_value=path_response) as retrieve_mock, patch.object(
                LLMClient, "health", return_value={"status": "ok"}
            ), patch.object(
                PathRetrieveClient, "health", return_value={"status": "ok"}
            ), patch.object(LLMClient, "generate", side_effect=responses):
                exit_code = eval_checked_batch_agent.main(
                    [
                        "--input",
                        str(qa_path),
                        "--output",
                        str(output_dir),
                        "--entity_map",
                        str(entity_map_path),
                        "--sample_index",
                        "1",
                    ]
                )

            output_path = output_dir / "checked_batch_eval.jsonl"
            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            summary = json.loads(
                (output_dir / "checked_batch_eval_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sample_index"], 1)
        self.assertEqual(records[0]["question"], "second question")
        retrieve_kwargs = retrieve_mock.call_args.kwargs
        self.assertEqual(retrieve_kwargs["sample_index"], 1)
        self.assertEqual(retrieve_kwargs["topic_entities"], ["m.topic2"])
        self.assertEqual(summary["n"], 1)
        self.assertEqual(summary["sample_index"], 1)

    def test_eval_cli_runs_sample_indices_list_in_order(self):
        path_response = make_response(
            [
                {"path": [["m.topic", "rel.a", "m.a"]], "log_score": -1.0},
            ]
        )
        responses = [
            GenerateResponse(
                text="Supporting Paths: 1\nAnswer: Answer A",
                used_adapter=True,
                tokens_generated=4,
                elapsed_ms=2.0,
            ),
            GenerateResponse(text="NONE", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            GenerateResponse(
                text="Supporting Paths: 1\nAnswer: Answer A",
                used_adapter=True,
                tokens_generated=4,
                elapsed_ms=2.0,
            ),
            GenerateResponse(text="NONE", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            qa_path = tmp_path / "qa.txt"
            qa_path.write_text(
                "first question [m.topic]\tm.a\n"
                "second question [m.topic]\tm.a\n"
                "third question [m.topic]\tm.a\n",
                encoding="utf-8",
            )
            entity_map_path = tmp_path / "mapped_entities.txt"
            entity_map_path.write_text(
                "m.topic\tTopic\nm.a\tAnswer A\n",
                encoding="utf-8",
            )
            output_dir = tmp_path / "checked"

            with patch.object(PathRetrieveClient, "retrieve", return_value=path_response) as retrieve_mock, patch.object(
                LLMClient, "health", return_value={"status": "ok"}
            ), patch.object(
                PathRetrieveClient, "health", return_value={"status": "ok"}
            ), patch.object(LLMClient, "generate", side_effect=responses):
                exit_code = eval_checked_batch_agent.main(
                    [
                        "--input",
                        str(qa_path),
                        "--output",
                        str(output_dir),
                        "--entity_map",
                        str(entity_map_path),
                        "--sample_indices",
                        "2,0",
                    ]
                )

            output_path = output_dir / "checked_batch_eval.jsonl"
            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            summary = json.loads(
                (output_dir / "checked_batch_eval_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual([record["sample_index"] for record in records], [2, 0])
        self.assertEqual([record["question"] for record in records], ["third question", "first question"])
        retrieve_sample_indices = [
            call.kwargs["sample_index"] for call in retrieve_mock.call_args_list
        ]
        self.assertEqual(retrieve_sample_indices, [2, 0])
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["sample_indices"], [2, 0])


if __name__ == "__main__":
    unittest.main()
