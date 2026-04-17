import argparse
import json
import tempfile
import unittest
import logging
import os
import subprocess
import sys
import importlib
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathfinder_agent.agent import PathfinderAgent
from pathfinder_agent.tools.question_utils import normalize_question
from pathfinder_agent.tools.answer_aggregator import aggregate_answers
from pathfinder_agent.tools.query_rewriter import rewrite_question
import run_agent_eval
from run_agent_eval import resolve_scored_answers


class PathfinderAgentFlowTest(unittest.TestCase):
    def _agent(self):
        agent = PathfinderAgent.__new__(PathfinderAgent)
        agent.model = object()
        agent.tokenizer = object()
        agent.transfernet_wrapper = object()
        return agent

    def test_normalize_question_removes_special_tokens_and_wordpieces(self):
        self.assertEqual(
            normalize_question("[CLS] what did george or ##well write [SEP]"),
            "what did george orwell write",
        )

    def test_run_requires_transfernet_wrapper(self):
        agent = self._agent()
        agent.transfernet_wrapper = None

        with self.assertRaisesRegex(RuntimeError, "transfernet_wrapper"):
            agent.run("question", "m.topic")

    def test_agent_import_does_not_configure_root_logger(self):
        code = """
import logging
import pathfinder_agent.agent as agent_module

root_handlers_after_import = len(logging.getLogger().handlers)
agent_logger = logging.getLogger("pathfinder_agent")
file_handlers_before_setup = sum(isinstance(h, logging.FileHandler) for h in agent_logger.handlers)
agent_module._setup_agent_logging()
agent_module._setup_agent_logging()
file_handlers_after_setup = sum(isinstance(h, logging.FileHandler) for h in agent_logger.handlers)

print(root_handlers_after_import)
print(file_handlers_before_setup)
print(file_handlers_after_setup)
"""
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(result.stdout.splitlines(), ["0", "0", "1"])

    def test_run_agent_eval_suppresses_transformers_warnings(self):
        self.assertEqual(logging.getLogger("transformers").level, logging.ERROR)

    def test_run_agent_eval_can_resuppress_after_transformers_import(self):
        import transformers.modeling_attn_mask_utils

        logging.getLogger("transformers").setLevel(logging.WARNING)
        run_agent_eval._suppress_transformers_warning_noise()

        self.assertEqual(logging.getLogger("transformers").level, logging.ERROR)

    def test_fallback_answer_is_reverified_before_aggregation(self):
        agent = self._agent()

        with (
            patch("pathfinder_agent.agent.rewrite_question", return_value=["raw question"]) as rewrite,
            patch(
                "pathfinder_agent.agent.retrieve_paths",
                side_effect=[["primary_path"], ["fallback_path"]],
            ) as retrieve,
            patch(
                "pathfinder_agent.agent.reason_with_paths",
                side_effect=[(["bad"], {1}), (["good"], {1})],
            ) as reason,
            patch(
                "pathfinder_agent.agent.verify_answer",
                side_effect=[(False, "slot mismatch"), (True, "OK")],
            ) as verify,
            patch(
                "pathfinder_agent.agent.aggregate_answers",
                return_value=["good"],
            ) as aggregate,
        ):
            result = agent.run("[CLS] raw question [SEP]", "m.topic")

        self.assertEqual(result, ["good"])
        rewrite.assert_called_once_with(agent.model, agent.tokenizer, "raw question", "m.topic")
        self.assertEqual(retrieve.call_args_list[0].kwargs["fallback"], False)
        self.assertEqual(retrieve.call_args_list[1].kwargs["fallback"], True)
        self.assertEqual(reason.call_args_list[1].args[3], ["fallback_path"])
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(agent.last_evidence_paths, ["fallback_path"])
        aggregate.assert_called_once_with(
            agent.model,
            agent.tokenizer,
            [["good"]],
            question="raw question",
        )

    def test_invalid_fallback_answer_is_not_aggregated(self):
        agent = self._agent()

        with (
            patch("pathfinder_agent.agent.rewrite_question", return_value=["raw question"]),
            patch(
                "pathfinder_agent.agent.retrieve_paths",
                side_effect=[["primary_path"], ["fallback_path"]],
            ),
            patch(
                "pathfinder_agent.agent.reason_with_paths",
                side_effect=[(["bad"], {1}), (["still_bad"], {1})],
            ),
            patch(
                "pathfinder_agent.agent.verify_answer",
                side_effect=[(False, "slot mismatch"), (False, "hallucination")],
            ),
            patch(
                "pathfinder_agent.agent.aggregate_answers",
                return_value=[],
            ) as aggregate,
        ):
            result = agent.run("raw question", "m.topic")

        self.assertEqual(result, [])
        self.assertEqual(agent.last_evidence_paths, [])
        aggregate.assert_called_once_with(
            agent.model,
            agent.tokenizer,
            [],
            question="raw question",
        )

    def test_run_executes_online_pipeline(self):
        agent = self._agent()
        with (
            patch("pathfinder_agent.agent.rewrite_question", return_value=["raw question"]) as rewrite,
            patch(
                "pathfinder_agent.agent.retrieve_paths",
                return_value=[{"path": [["m.topic", "rel", "m.online"]], "log_score": 1.0}],
            ) as retrieve,
            patch(
                "pathfinder_agent.agent.reason_with_paths",
                return_value=(["m.online"], {1}),
            ) as reason,
            patch(
                "pathfinder_agent.agent.verify_answer",
                return_value=(True, "OK"),
            ) as verify,
            patch(
                "pathfinder_agent.agent.aggregate_answers",
                return_value=["m.online"],
            ) as aggregate,
        ):
            result = agent.run("raw question", "m.topic")

        self.assertEqual(result, ["m.online"])
        rewrite.assert_called_once()
        retrieve.assert_called_once()
        reason.assert_called_once()
        verify.assert_called_once()
        aggregate.assert_called_once()
        self.assertEqual(
            agent.last_evidence_paths,
            [{"path": [["m.topic", "rel", "m.online"]], "log_score": 1.0}],
        )
        self.assertEqual(agent.last_run_metadata["agent_mode"], "online_pipeline")
        self.assertEqual(agent.last_run_metadata["selected_source"], "online_pipeline")
        self.assertFalse(agent.last_run_metadata["fallback_used"])
        self.assertEqual(agent.last_run_metadata["final_evidence_source"], "online_primary")
        self.assertNotIn("selector_confidence", agent.last_run_metadata)
        self.assertNotIn("risk_flags", agent.last_run_metadata)
        self.assertNotIn("rerank_used", agent.last_run_metadata)


class PathfinderEvalMetricTest(unittest.TestCase):
    def test_main_calls_agent_without_record_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "output.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "question": "who is the governor",
                        "golden": ["m.answer"],
                        "topics": ["m.topic"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            args = argparse.Namespace(
                input=str(input_path),
                output=str(output_path),
                ckpt="dummy.ckpt",
                input_dir="dummy_input",
                model="dummy-model",
                adapter=None,
                entity_map=None,
                limit=0,
                device="cpu",
                server_url=None,
                prediction_mode="online",
            )
            agent = unittest.mock.Mock()
            agent.run.return_value = ["m.answer"]
            agent.last_evidence_paths = []
            agent.last_run_metadata = {
                "agent_mode": "online_pipeline",
                "selected_source": "online_pipeline",
                "fallback_used": False,
                "final_evidence_source": "online_primary",
            }

            with (
                patch("run_agent_eval.parse_args", return_value=args),
                patch("run_agent_eval.setup_logger", return_value=unittest.mock.Mock()),
                patch("run_agent_eval.load_eval_entity_map", return_value=(None, None)),
                patch("run_agent_eval.PathfinderAgent", return_value=agent),
                patch("run_agent_eval.TransferNetWrapper"),
            ):
                run_agent_eval.main()

        agent.run.assert_called_once_with("who is the governor", "m.topic")

    def test_remote_eval_payload_does_not_include_record_context(self):
        captured_payload = {}

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "pred_answer": ["m.answer"],
                        "evidence_paths": [],
                        "agent_mode": "online_pipeline",
                        "selected_source": "online_pipeline",
                        "fallback_used": False,
                        "final_evidence_source": "online_primary",
                    }
                ).encode()

        def _fake_urlopen(req, timeout=0):
            captured_payload.update(json.loads(req.data.decode()))
            return _FakeResponse()

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            run_agent_eval._run_sample_remote(
                "http://localhost:8787",
                "who is the governor",
                "m.topic",
            )

        self.assertEqual(
            captured_payload,
            {
                "question": "who is the governor",
                "topic_entity": "m.topic",
            },
        )

    def test_name_predictions_are_path_constrained_before_metrics(self):
        pred_answers, expanded_answers, metrics = resolve_scored_answers(
            pred_answers=["Salisbury"],
            golden=["m.0jgvy"],
            mmr_paths=[
                {"path": [["m.topic", "rel", "m.0jgvy"]], "log_score": 0.0},
            ],
            rev_entity_map={"salisbury": {"m.010hl7", "m.0jgvy"}},
        )

        self.assertEqual(expanded_answers, ["m.010hl7", "m.0jgvy"])
        self.assertEqual(pred_answers, ["m.0jgvy"])
        self.assertEqual(metrics["hit1"], 1)
        self.assertEqual(metrics["f1"], 1.0)

    def test_mid_predictions_keep_existing_metric_behavior_without_entity_map(self):
        pred_answers, expanded_answers, metrics = resolve_scored_answers(
            pred_answers=["m.answer"],
            golden=["m.answer"],
            mmr_paths=[],
            rev_entity_map=None,
        )

        self.assertIsNone(expanded_answers)
        self.assertEqual(pred_answers, ["m.answer"])
        self.assertEqual(metrics["hit1"], 1)


class PathfinderAggregatorTest(unittest.TestCase):
    def test_aggregate_answers_preserves_observed_candidates_without_llm_cleanup(self):
        result = aggregate_answers(
            object(),
            object(),
            [["m.correct"], ["m.other"], ["m.correct"]],
            question="who is the governor",
        )

        self.assertEqual(result, ["m.correct", "m.other"])


class PathfinderConfigTest(unittest.TestCase):
    def test_default_adapter_points_to_groupAname_v2(self):
        config = importlib.import_module("pathfinder_agent.config")

        self.assertTrue(config.LORA_ADAPTER_PATH.endswith("models/webqsp/ablation/groupAname_v2"))


class PathfinderQueryRewriterTest(unittest.TestCase):
    def test_rewrite_question_filters_explanatory_lines_and_keeps_question_variants(self):
        model = type(
            "DummyModel",
            (),
            {"device": "cpu", "generate": lambda self, **_: torch.tensor([[0, 1]])},
        )()
        tokenizer = type(
            "DummyTokenizer",
            (),
            {
                "eos_token_id": 0,
                "decode": lambda self, *_args, **_kwargs: "\n".join(
                    [
                        "[m.topic] is a person, not an electorate. The correct question should be about the electorate represented by [m.topic].",
                        "What electorate does [m.topic] represent?",
                        "[m.topic] wrote extensively on philosophical topics.",
                    ]
                ),
            },
        )()

        with patch(
            "pathfinder_agent.tools.query_rewriter.apply_template_and_pad",
            return_value={"input_ids": torch.tensor([[0]])},
        ):
            rewrites = rewrite_question(
                model,
                tokenizer,
                "what electorate does anna bligh represent",
                "m.topic",
            )

        self.assertEqual(
            rewrites,
            [
                "what electorate does anna bligh represent",
                "What electorate does [m.topic] represent?",
            ],
        )


class PathfinderReasonerVerifierTest(unittest.TestCase):
    def test_reasoner_filters_raw_mid_and_missing_label_answers(self):
        reasoner = importlib.import_module("pathfinder_agent.tools.llm_reasoner")
        model = type(
            "DummyModel",
            (),
            {"device": "cpu", "generate": lambda self, **_: torch.tensor([[0, 1]])},
        )()
        tokenizer = type(
            "DummyTokenizer",
            (),
            {
                "eos_token_id": 0,
                "decode": lambda self, *_args, **_kwargs: (
                    "Supporting Paths: 2\n"
                    "Answer: m.0123 | No English Label | LA Galaxy"
                ),
            },
        )()
        paths = [
            {"path": [["David Beckham", "sports.pro_athlete.teams", "Paris Saint-Germain FC"]]},
            {"path": [["David Beckham", "sports.pro_athlete.teams", "LA Galaxy"]]},
        ]

        with patch(
            "pathfinder_agent.tools.llm_reasoner.apply_template_and_pad",
            return_value={"input_ids": torch.tensor([[0]])},
        ):
            answers, indices = reasoner.reason_with_paths(
                model,
                tokenizer,
                "what team did david beckham play for in 2011",
                paths,
            )

        self.assertEqual(indices, {2})
        self.assertEqual(answers, ["LA Galaxy"])

    def test_reasoner_collapses_single_answer_questions_to_best_supported_candidate(self):
        reasoner = importlib.import_module("pathfinder_agent.tools.llm_reasoner")
        model = type(
            "DummyModel",
            (),
            {"device": "cpu", "generate": lambda self, **_: torch.tensor([[0, 1]])},
        )()
        tokenizer = type(
            "DummyTokenizer",
            (),
            {
                "eos_token_id": 0,
                "decode": lambda self, *_args, **_kwargs: (
                    "Supporting Paths: 1, 2\n"
                    "Answer: Miami | New York City | Brooklyn"
                ),
            },
        )()
        paths = [
            {
                "path": [
                    ["Jamarcus Russell", "people.person.place_of_birth", "Miami"],
                    ["Miami", "location.location.containedby", "Florida"],
                ]
            },
            {
                "path": [
                    ["Jamarcus Russell", "people.person.place_of_birth", "Brooklyn"],
                    ["Brooklyn", "location.location.containedby", "New York City"],
                ]
            },
        ]

        with patch(
            "pathfinder_agent.tools.llm_reasoner.apply_template_and_pad",
            return_value={"input_ids": torch.tensor([[0]])},
        ):
            answers, indices = reasoner.reason_with_paths(
                model,
                tokenizer,
                "where is jamarcus russell from",
                paths,
            )

        self.assertEqual(indices, {1, 2})
        self.assertEqual(answers, ["Miami"])

    def test_verifier_rejects_raw_mid_without_consulting_llm(self):
        verifier = importlib.import_module("pathfinder_agent.tools.answer_verifier")
        model = type("DummyModel", (), {"device": "cpu"})()
        tokenizer = object()

        is_valid, feedback = verifier.verify_answer(
            model,
            tokenizer,
            "what was jesse james killed with",
            ["m.034qg"],
            {1},
            [{"path": [["Jesse James", "people.deceased_person.cause_of_death", "handgun"]]}],
        )

        self.assertFalse(is_valid)
        self.assertIn("raw MID", feedback)

    def test_verifier_rejects_candidate_not_supported_by_cited_paths(self):
        verifier = importlib.import_module("pathfinder_agent.tools.answer_verifier")
        model = type("DummyModel", (), {"device": "cpu"})()
        tokenizer = object()

        is_valid, feedback = verifier.verify_answer(
            model,
            tokenizer,
            "who did the voice of darth vader in episode 3",
            ["Hayden Christensen"],
            {1},
            [{"path": [["Anakin Skywalker", "film.performance.actor", "James Earl Jones"]]}],
        )

        self.assertFalse(is_valid)
        self.assertIn("supported", feedback)


class PathfinderRetrieverPolicyTest(unittest.TestCase):
    def test_temporal_fallback_policy_relaxes_tail_dedup(self):
        retriever = importlib.import_module("pathfinder_agent.tools.dynamic_retriever")

        primary = retriever.get_retrieval_policy(
            "what team did david beckham play for in 2011",
            fallback=False,
        )
        fallback = retriever.get_retrieval_policy(
            "what team did david beckham play for in 2011",
            fallback=True,
        )

        self.assertEqual(primary["beam_size"], retriever.BEAM_SIZE_PRIMARY)
        self.assertEqual(fallback["beam_size"], retriever.BEAM_SIZE_FALLBACK)
        self.assertGreater(fallback["max_duplicate_tail_nodes"], primary["max_duplicate_tail_nodes"])


if __name__ == "__main__":
    unittest.main()
