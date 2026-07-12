import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kgqa.agent.tools import RejectedAnswerCheckTool
from kgqa.agent.tools.cited_path_check import (
    STRICT_REJECTED_ANSWER_CHECK_SYSTEM,
    parse_rejected_answer_indices,
)
from kgqa.llm_server.client import GenerateResponse


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.responses.pop(0)


class RejectedAnswerCheckToolTests(unittest.TestCase):
    def test_rejected_answer_check_keeps_complement_by_candidate_answer(self):
        client = FakeLLMClient(
            [
                GenerateResponse(text="2", used_adapter=False, tokens_generated=3, elapsed_ms=4.0),
            ]
        )
        tool = RejectedAnswerCheckTool(client=client)
        named_paths = [
            {"path": [["Jamaica", "location.country.languages_spoken", "Jamaican English"]]},
            {"path": [["Jamaica", "location.country.languages_spoken", "Jamaican Patois"]]},
            {"path": [["Jamaica", "language.human_language.main_country_reverse", "Jamaican English"]]},
        ]
        raw_paths = [
            {"path": [["m.03_r3", "location.country.languages_spoken", "m.01428y"]]},
            {"path": [["m.03_r3", "location.country.languages_spoken", "m.04ygk0"]]},
            {"path": [["m.03_r3", "language.human_language.main_country_reverse", "m.01428y"]]},
        ]

        result = tool(
            "what does jamaican people speak",
            named_paths,
            raw_paths=raw_paths,
            cited_path_indices=[1, 2, 3],
        )

        self.assertEqual(result.check_mode, "reject-answer-list:loose")
        self.assertEqual(result.rejected_answer_indices, [2])
        self.assertEqual(result.accepted_path_indices, [1, 3])
        self.assertEqual(result.predicted_answer_names, ["Jamaican English"])
        self.assertEqual(result.predicted_mids, ["m.01428y"])
        self.assertEqual(result.total_tokens_generated, 3)
        self.assertEqual(result.total_elapsed_ms, 4.0)
        self.assertEqual(len(client.calls), 1)
        self.assertIn("Candidate Answers:", client.calls[0][0])
        self.assertIn("[1] Jamaican English (supported by paths: 1, 3)", client.calls[0][0])
        self.assertIn("[2] Jamaican Patois (supported by paths: 2)", client.calls[0][0])
        self.assertEqual(client.calls[0][1]["max_new_tokens"], 48)

    def test_strict_rejected_answer_check_uses_strict_policy_prompt(self):
        client = FakeLLMClient(
            [
                GenerateResponse(text="1", used_adapter=False, tokens_generated=3, elapsed_ms=4.0),
            ]
        )
        tool = RejectedAnswerCheckTool(client=client, reject_policy="strict")
        named_paths = [
            {"path": [["Topic", "rel.associated_place", "Related Place"]]},
        ]
        raw_paths = [
            {"path": [["m.topic", "rel.associated_place", "m.place"]]},
        ]

        result = tool(
            "who directed the topic",
            named_paths,
            raw_paths=raw_paths,
            cited_path_indices=[1],
        )

        self.assertEqual(result.check_mode, "reject-answer-list:strict")
        self.assertEqual(result.rejected_answer_indices, [1])
        self.assertEqual(result.accepted_path_indices, [])
        self.assertIn("indirect or two-hop", client.calls[0][0])
        self.assertEqual(client.calls[0][1]["system_prompt"], STRICT_REJECTED_ANSWER_CHECK_SYSTEM)

    def test_from_record_uses_record_fields(self):
        client = FakeLLMClient(
            [
                GenerateResponse(text="NONE", used_adapter=False, tokens_generated=2, elapsed_ms=2.0),
            ]
        )
        tool = RejectedAnswerCheckTool(client=client)
        record = {
            "question": "what timezone is sweden",
            "cited_path_indices": [1, 2],
            "named_mmr_reason_paths": [
                {"path": [["Sweden", "location.location.time_zones", "Central European Time"]]},
                {"path": [["Sweden", "time.time_zone.locations_in_this_time_zone", "Central European Time"]]},
            ],
            "raw_mmr_reason_paths": [
                {"path": [["m.0d0vqn", "location.location.time_zones", "m.02llzg"]]},
                {"path": [["m.0d0vqn", "time.time_zone.locations_in_this_time_zone", "m.02llzg"]]},
            ],
        }

        result = tool.from_record(record)

        self.assertEqual(result.accepted_path_indices, [1, 2])
        self.assertEqual(result.predicted_answer_names, ["Central European Time"])
        self.assertEqual(result.predicted_mids, ["m.02llzg"])
        self.assertEqual(result.path_evaluations[0].tail_mid, "m.02llzg")

    def test_no_candidates_returns_empty_result_without_llm_call(self):
        client = FakeLLMClient([])
        tool = RejectedAnswerCheckTool(client=client)

        result = tool(
            "any question",
            [{"path": [["m.topic", "rel.a", "m.a"]]}],
            cited_path_indices=[99],
        )

        self.assertEqual(result.accepted_path_indices, [])
        self.assertEqual(result.candidate_answers, [])
        self.assertEqual(len(client.calls), 0)

    def test_parse_rejected_answer_indices_ignores_none_and_out_of_range(self):
        self.assertEqual(parse_rejected_answer_indices("NONE", 3), [])
        self.assertEqual(parse_rejected_answer_indices("1,3,99", 3), [1, 3])


if __name__ == "__main__":
    unittest.main()
