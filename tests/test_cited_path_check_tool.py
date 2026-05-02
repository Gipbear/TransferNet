import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.llm_server.client import GenerateResponse
from oh_my_agent.tools import CitedPathCheckTool
from oh_my_agent.tools.cited_path_check import (
    CITED_PATH_CHECK_SYSTEM,
    build_cited_path_prompt,
    parse_cited_path_output,
)


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.responses.pop(0)


class CitedPathCheckToolTests(unittest.TestCase):
    def test_system_prompt_uses_loose_standard(self):
        self.assertIn("Use a loose standard", CITED_PATH_CHECK_SYSTEM)
        self.assertIn("Answer ONLY 'Y'", CITED_PATH_CHECK_SYSTEM)

    def test_build_prompt_formats_one_path(self):
        prompt = build_cited_path_prompt(
            "What is Obama's father's name?",
            [["Barack Obama", "people.person.parents", "Barack Obama Sr."]],
        )
        self.assertIn("Q: What is Obama's father's name?", prompt)
        self.assertIn("Barack Obama - [people.person.parents] -> Barack Obama Sr.", prompt)
        self.assertTrue(prompt.endswith("Output:"))

    def test_parse_cited_path_output_requires_leading_y(self):
        self.assertTrue(parse_cited_path_output("Y"))
        self.assertTrue(parse_cited_path_output(" yes"))
        self.assertFalse(parse_cited_path_output("N"))
        self.assertFalse(parse_cited_path_output(""))

    def test_checks_only_valid_cited_indices_and_collects_tails(self):
        client = FakeLLMClient(
            [
                GenerateResponse(text="Y", used_adapter=False, tokens_generated=1, elapsed_ms=2.0),
                GenerateResponse(text="N", used_adapter=False, tokens_generated=1, elapsed_ms=3.0),
            ]
        )
        tool = CitedPathCheckTool(client=client)
        named_paths = [
            {"path": [["John F. Kennedy", "government.us_president.vice_president", "Lyndon B. Johnson"]]},
            {"path": [["John F. Kennedy", "people.person.place_of_birth", "Brookline"]]},
        ]
        raw_paths = [
            {"path": [["m.0d3k14", "government.us_president.vice_president", "m.0f7fy"]]},
            {"path": [["m.0d3k14", "people.person.place_of_birth", "m.0vzm"]]},
        ]

        result = tool(
            "who was vice president after kennedy died",
            named_paths,
            raw_paths=raw_paths,
            cited_path_indices=[0, 2, 1, 99],
        )

        self.assertEqual(result.cited_path_indices, [1, 2])
        self.assertEqual(result.accepted_path_indices, [1])
        self.assertEqual(result.predicted_answer_names, ["Lyndon B. Johnson"])
        self.assertEqual(result.predicted_mids, ["m.0f7fy"])
        self.assertTrue(result.any_accepted_path)
        self.assertEqual(result.total_tokens_generated, 2)
        self.assertEqual(result.total_elapsed_ms, 5.0)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0][1]["max_new_tokens"], 2)
        self.assertFalse(client.calls[0][1]["use_adapter"])

    def test_from_record_uses_record_fields_and_deduplicates_answers(self):
        client = FakeLLMClient(
            [
                GenerateResponse(text="Y", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
                GenerateResponse(text="Y", used_adapter=False, tokens_generated=1, elapsed_ms=1.0),
            ]
        )
        tool = CitedPathCheckTool(client=client)
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


if __name__ == "__main__":
    unittest.main()
