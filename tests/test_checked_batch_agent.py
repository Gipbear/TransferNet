import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.agent import CheckedBatchWebQAgent
from oh_my_agent.cli import eval_checked_batch_agent
from oh_my_agent.llm_server.client import GenerateResponse, LLMClient
from oh_my_agent.path_retrieve_server.client import PathRetrieveClient, PathRetrieveResponse
from oh_my_agent.tools import AnswerWithPathsTool, CitedPathCheckTool, PathRetrieveTool


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


def make_response(paths):
    return PathRetrieveResponse(
        question="where is example from",
        sample_index=0,
        topics=["m.topic"],
        hop=1,
        mmr_reason_paths=paths,
        prediction={},
        elapsed_ms=10.0,
        method="tail_blend",
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
        path_client = FakePathClient(make_response(raw_paths))
        llm_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 1, 2\nAnswer: Answer A | Answer B",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="Y", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
                GenerateResponse(text="Y", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
                GenerateResponse(
                    text="Supporting Paths: 1, 2\nAnswer: Answer A | Answer C",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="Y", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
                GenerateResponse(text="N", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            ]
        )
        agent = CheckedBatchWebQAgent(
            path_tool=PathRetrieveTool(client=path_client, entity_map=entity_map),
            answer_tool=AnswerWithPathsTool(client=llm_client),
            check_tool=CitedPathCheckTool(client=llm_client),
        )

        result = agent.run("where is example from", "m.topic", batch_size=2)

        retrieve_kwargs = path_client.calls[0][1]
        self.assertEqual(retrieve_kwargs["method"], "tail_blend")
        self.assertEqual(retrieve_kwargs["alpha_final"], 1.0)
        self.assertEqual(retrieve_kwargs["beam_size"], 50)
        self.assertEqual(retrieve_kwargs["lambda_val"], 0.5)
        self.assertEqual(result.stop_reason, "mixed")
        self.assertEqual([item.batch_status for item in result.iterations], ["all_correct", "mixed"])
        self.assertEqual(result.iterations[1].global_cited_path_indices, [3, 4])
        self.assertEqual(result.final_accepted_path_indices, [1, 2, 3])
        self.assertEqual(result.pred_answer_names, ["Answer A", "Answer B", "Answer C"])
        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.a", "m.b", "m.c"])
        self.assertEqual(result.checked_paths_count, 4)
        self.assertEqual(result.accepted_paths_count, 3)

    def test_all_correct_compares_against_full_reasoning_batch(self):
        raw_paths = [
            {"path": [["m.topic", "rel.a", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.b", "m.b"]], "log_score": -2.0},
            {"path": [["m.topic", "rel.c", "m.c"]], "log_score": -3.0},
        ]
        llm_client = FakeLLMClient(
            [
                GenerateResponse(
                    text="Supporting Paths: 1, 2\nAnswer: Answer A | Answer B",
                    used_adapter=True,
                    tokens_generated=8,
                    elapsed_ms=5.0,
                ),
                GenerateResponse(text="Y", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
                GenerateResponse(text="Y", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            ]
        )
        agent = CheckedBatchWebQAgent(
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
            check_tool=CitedPathCheckTool(client=llm_client),
        )

        result = agent.run("where is example from", "m.topic", batch_size=3)

        self.assertEqual(result.iterations[0].batch_status, "mixed")
        self.assertEqual(result.stop_reason, "mixed")
        self.assertEqual(result.final_accepted_path_indices, [1, 2])

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
        agent = CheckedBatchWebQAgent(
            path_tool=PathRetrieveTool(
                client=FakePathClient(make_response(raw_paths)),
                entity_map={"m.topic": "Topic", "m.a": "Answer A", "m.b": "Answer B"},
            ),
            answer_tool=AnswerWithPathsTool(client=llm_client),
            check_tool=CitedPathCheckTool(client=llm_client),
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
            ]
        )
        responses = [
            GenerateResponse(
                text="Supporting Paths: 1, 2\nAnswer: Answer A | Answer B",
                used_adapter=True,
                tokens_generated=8,
                elapsed_ms=5.0,
            ),
            GenerateResponse(text="Y", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            GenerateResponse(text="N", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            qa_path = tmp_path / "qa.txt"
            qa_path.write_text("where is example from [m.topic]\tm.a\n", encoding="utf-8")
            entity_map_path = tmp_path / "mapped_entities.txt"
            entity_map_path.write_text(
                "m.topic\tTopic\nm.a\tAnswer A\nm.b\tAnswer B\n",
                encoding="utf-8",
            )
            output_path = tmp_path / "checked.jsonl"

            with patch.object(PathRetrieveClient, "retrieve", return_value=path_response), patch.object(
                LLMClient, "health", return_value={"status": "ok"}
            ), patch.object(LLMClient, "generate", side_effect=responses):
                exit_code = eval_checked_batch_agent.main(
                    [
                        "--input",
                        str(qa_path),
                        "--output",
                        str(output_path),
                        "--entity_map",
                        str(entity_map_path),
                        "--no_archive",
                    ]
                )

            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            summary = json.loads(
                output_path.with_name("checked_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(records[0]["final_accepted_path_indices"], [1])
        self.assertEqual(records[0]["iterations"][0]["global_cited_path_indices"], [1, 2])
        self.assertEqual(summary["n"], 1)
        self.assertEqual(summary["hit1"], 1.0)
        self.assertEqual(summary["avg_batches_used"], 1.0)
        self.assertEqual(summary["avg_checked_paths"], 2.0)
        self.assertEqual(summary["avg_accepted_paths"], 1.0)
        self.assertEqual(summary["stop_reason_counts"], {"mixed": 1})


if __name__ == "__main__":
    unittest.main()
