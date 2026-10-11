from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from experiments.ch5.run import _agent_command
from kgqa.agent.cli import eval_checked_batch
from kgqa.agent.comparison_answer import (
    ComparisonAnswerDecision,
    CWQComparisonAnswerTool,
    apply_comparison_decision,
)
from kgqa.retrieve.api.client import PathRetrieveClient, PathRetrieveResponse
from kgqa.serving.llm.client import GenerateResponse, LLMClient


class CWQComparisonAnswerToolTests(unittest.TestCase):
    def test_rerank_merge_prioritizes_selected_existing_answers_and_keeps_remainder(self):
        @dataclass(frozen=True)
        class Result:
            pred_answer_names: list[str]
            pred_answer_expanded_mids: list[str]
            pred_answer_disambiguated_mids: list[str]
            final_answer_count: int

        result = Result(["Wrong", "Selected B", "Tail"], ["old"], ["old"], 3)
        decision = ComparisonAnswerDecision(
            0, "q", ["Selected A", "selected b"], "attribute_rule", "rerank_merge", True, "selected",
        )

        merged = apply_comparison_decision(result, decision)

        self.assertEqual(merged.pred_answer_names, ["Selected B", "Selected A", "Wrong", "Tail"])
        self.assertEqual(merged.pred_answer_expanded_mids, merged.pred_answer_names)
        self.assertEqual(merged.pred_answer_disambiguated_mids, merged.pred_answer_names)
        self.assertEqual(merged.final_answer_count, 4)

    def test_resume_rejects_disabling_a_previous_comparison_run(self):
        records = {
            0: {
                "comparison_tool": {
                    "enabled": True,
                    "artifact_path": "/tmp/comparison.jsonl",
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "different comparison answers"):
            eval_checked_batch._validate_comparison_resume(records, None)

    def test_ch5_command_forwards_comparison_artifact(self):
        command = _agent_command(
            "cwq",
            {
                "beam_size": 50,
                "lambda_val": 0.2,
                "batch_size": 20,
                "check_mode": "hybrid-reject-list",
            },
            Path("output"),
            {
                "qa_file": "qa.jsonl",
                "cwq_comparison_answers": "comparison.jsonl",
            },
        )

        option_index = command.index("--cwq_comparison_answers")
        self.assertEqual(command[option_index + 1], "comparison.jsonl")

    def test_selects_only_matching_attribute_rule_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact = Path(tmp_dir) / "answers.jsonl"
            rows = [
                {
                    "sample_index": 0,
                    "question": "Which country has the smallest code?",
                    "answers": ["Hungary"],
                    "source": "attribute_rule",
                },
                {
                    "sample_index": 1,
                    "question": "Who leads France?",
                    "answers": ["France"],
                    "source": "ch4",
                },
            ]
            artifact.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            tool = CWQComparisonAnswerTool(str(artifact))

            selected = tool(0, rows[0]["question"])
            upstream = tool(1, rows[1]["question"])
            mismatch = tool(0, "Different question")

        self.assertTrue(selected.applied)
        self.assertEqual(selected.answers, ["Hungary"])
        self.assertEqual(selected.reason, "selected")
        self.assertFalse(upstream.applied)
        self.assertEqual(upstream.reason, "upstream_answer")
        self.assertFalse(mismatch.applied)
        self.assertEqual(mismatch.reason, "question_mismatch")

    def test_eval_cli_overrides_answer_and_records_provenance(self):
        question = "Which country has the smallest code?"
        path_response = PathRetrieveResponse(
            question=question,
            sample_index=0,
            topics=["Europe"],
            hop=1,
            mmr_reason_paths=[
                {"path": [["Europe", "location.code", "France"]], "log_score": -1.0}
            ],
            prediction={"France": 0.9},
            elapsed_ms=10.0,
            eta=1.0,
            threshold=0.01,
            beam_size=50,
            lambda_val=0.2,
            cache_path="cache.pt",
        )
        responses = [
            GenerateResponse(
                text="Supporting Paths: 1\nAnswer: France",
                used_adapter=True,
                tokens_generated=4,
                elapsed_ms=2.0,
            ),
            GenerateResponse(
                text="NONE",
                used_adapter=False,
                tokens_generated=1,
                elapsed_ms=1.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            qa_path = tmp_path / "qa.jsonl"
            qa_path.write_text(
                json.dumps(
                    {
                        "question": question,
                        "topics": ["Europe"],
                        "golden": ["Hungary"],
                        "sample_index": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            artifact = tmp_path / "answers.jsonl"
            artifact.write_text(
                json.dumps(
                    {
                        "sample_index": 0,
                        "question": question,
                        "answers": ["Hungary"],
                        "source": "attribute_rule",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = tmp_path / "output"

            with patch.object(
                PathRetrieveClient,
                "retrieve",
                return_value=path_response,
            ), patch.object(LLMClient, "generate", side_effect=responses):
                exit_code = eval_checked_batch.main(
                    [
                        "--dataset",
                        "cwq",
                        "--input",
                        str(qa_path),
                        "--output",
                        str(output_dir),
                        "--cwq_comparison_answers",
                        str(artifact),
                        "--skip_server_check",
                    ]
                )

            record = json.loads(
                (output_dir / "checked_batch_eval.jsonl").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (output_dir / "checked_batch_eval_summary.json").read_text(encoding="utf-8")
            )
            initial_answer = json.loads(
                (output_dir / "initial_answer.jsonl").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(record["pred_answer_names"], ["Hungary"])
        self.assertEqual(record["pred_answer_disambiguated_mids"], ["Hungary"])
        self.assertEqual(record["answer_source"], "cwq_comparison_tool")
        self.assertEqual(record["comparison_tool"]["original_answers"], ["France"])
        self.assertEqual(record["hit1"], 1.0)
        self.assertEqual(record["path_faithfulness_scope"], "checked_batch_answer_before_override")
        self.assertEqual(initial_answer["llm_pred"], ["France"])
        self.assertEqual(summary["comparison_override_count"], 1)
        self.assertEqual(summary["comparison_fallback_count"], 0)


if __name__ == "__main__":
    unittest.main()
