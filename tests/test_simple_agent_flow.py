import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.agent import SimpleWebQAgent
from oh_my_agent.cli import eval_webqsp, run_simple_agent
from oh_my_agent.llm_server.client import GenerateResponse, LLMClient
from oh_my_agent.path_server.client import PathRetrievalClient, PathRetrievalResponse
from oh_my_agent.tools import AnswerWithPathsTool, PathRetrievalTool


class FakePathClient:
    def __init__(self, response):
        self.response = response

    def retrieve(self, question, **kwargs):
        return self.response


class FakeLLMClient:
    def __init__(self, response):
        self.response = response

    def generate(self, prompt, **kwargs):
        return self.response


class SimpleAgentFlowTests(unittest.TestCase):
    def test_simple_agent_runs_fixed_two_step_flow(self):
        path_tool = PathRetrievalTool(
            client=FakePathClient(
                PathRetrievalResponse(
                    question="where is jamarcus russell from",
                    topics=["JaMarcus Russell"],
                    hop=1,
                    mmr_reason_paths=[
                        {"path": [["JaMarcus Russell", "people.person.place_of_birth", "Mobile"]], "log_score": -1.0}
                    ],
                    prediction={"Mobile": 0.99},
                    elapsed_ms=10.0,
                    raw_topics=["m.0cjcdj"],
                    raw_mmr_reason_paths=[
                        {"path": [["m.0cjcdj", "people.person.place_of_birth", "m.058cm"]], "log_score": -1.0}
                    ],
                    raw_prediction={"m.058cm": 0.99},
                )
            ),
            entity_map={
                "m.0cjcdj": "JaMarcus Russell",
                "m.058cm": "Mobile",
                "m.other": "Mobile",
            },
        )
        answer_tool = AnswerWithPathsTool(
            client=FakeLLMClient(
                GenerateResponse(
                    text="Supporting Paths: 1\nAnswer: Mobile",
                    used_adapter=True,
                    tokens_generated=5,
                    elapsed_ms=4.0,
                )
            )
        )
        agent = SimpleWebQAgent(path_tool=path_tool, answer_tool=answer_tool)

        result = agent.run("where is jamarcus russell from", "m.0cjcdj")

        self.assertEqual(result.pred_answer_names, ["Mobile"])
        self.assertEqual(result.pred_answer_expanded_mids, ["m.058cm", "m.other"])
        self.assertEqual(result.pred_answer_disambiguated_mids, ["m.058cm"])
        self.assertEqual(result.cited_path_indices, [1])

    def test_eval_cli_writes_jsonl_and_summary(self):
        path_response = PathRetrievalResponse(
            question="where is jamarcus russell from",
            topics=["JaMarcus Russell"],
            hop=1,
            mmr_reason_paths=[
                {"path": [["JaMarcus Russell", "people.person.place_of_birth", "Mobile"]], "log_score": -1.0}
            ],
            prediction={"Mobile": 0.99},
            elapsed_ms=10.0,
            raw_topics=["m.0cjcdj"],
            raw_mmr_reason_paths=[
                {"path": [["m.0cjcdj", "people.person.place_of_birth", "m.058cm"]], "log_score": -1.0}
            ],
            raw_prediction={"m.058cm": 0.99},
        )
        llm_response = GenerateResponse(
            text="Supporting Paths: 1\nAnswer: Mobile",
            used_adapter=True,
            tokens_generated=5,
            elapsed_ms=4.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            qa_path = tmp_path / "qa.txt"
            qa_path.write_text(
                "where is jamarcus russell from [m.0cjcdj]\tm.058cm\n",
                encoding="utf-8",
            )
            entity_map_path = tmp_path / "mapped_entities.txt"
            entity_map_path.write_text(
                "m.0cjcdj\tJaMarcus Russell\nm.058cm\tMobile\n",
                encoding="utf-8",
            )
            output_path = tmp_path / "results.jsonl"

            with patch.object(PathRetrievalClient, "retrieve", return_value=path_response), patch.object(
                LLMClient, "generate", return_value=llm_response
            ):
                exit_code = eval_webqsp.main(
                    [
                        "--input",
                        str(qa_path),
                        "--output",
                        str(output_path),
                        "--entity_map",
                        str(entity_map_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            summary = json.loads(
                output_path.with_name("results_summary.json").read_text(encoding="utf-8")
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["pred_answer_disambiguated_mids"], ["m.058cm"])
            self.assertEqual(records[0]["cited_path_indices"], [1])
            self.assertEqual(summary["n"], 1)
            self.assertEqual(summary["hit1"], 1.0)
            self.assertEqual(summary["format_compliance"], 1.0)

    def test_run_simple_agent_cli_prints_json_result(self):
        path_response = PathRetrievalResponse(
            question="where is jamarcus russell from",
            topics=["JaMarcus Russell"],
            hop=1,
            mmr_reason_paths=[
                {"path": [["JaMarcus Russell", "people.person.place_of_birth", "Mobile"]], "log_score": -1.0}
            ],
            prediction={"Mobile": 0.99},
            elapsed_ms=10.0,
            raw_topics=["m.0cjcdj"],
            raw_mmr_reason_paths=[
                {"path": [["m.0cjcdj", "people.person.place_of_birth", "m.058cm"]], "log_score": -1.0}
            ],
            raw_prediction={"m.058cm": 0.99},
        )
        llm_response = GenerateResponse(
            text="Supporting Paths: 1\nAnswer: Mobile",
            used_adapter=True,
            tokens_generated=5,
            elapsed_ms=4.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            entity_map_path = Path(tmp_dir) / "mapped_entities.txt"
            entity_map_path.write_text(
                "m.0cjcdj\tJaMarcus Russell\nm.058cm\tMobile\n",
                encoding="utf-8",
            )
            buffer = io.StringIO()
            with patch.object(PathRetrievalClient, "retrieve", return_value=path_response), patch.object(
                LLMClient, "generate", return_value=llm_response
            ), contextlib.redirect_stdout(buffer):
                exit_code = run_simple_agent.main(
                    [
                        "--question",
                        "where is jamarcus russell from",
                        "--topic_mid",
                        "m.0cjcdj",
                        "--entity_map",
                        str(entity_map_path),
                    ]
                )

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["pred_answer_disambiguated_mids"], ["m.058cm"])


if __name__ == "__main__":
    unittest.main()
