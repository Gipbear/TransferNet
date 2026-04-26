import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.llm_server.client import GenerateResponse
from oh_my_agent.tools.answer_check import (
    AnswerCheckTool,
    apply_verify_guardrails,
    parse_verify_output,
)


class FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response


class VerifyPromptTests(unittest.TestCase):
    def test_verify_prompt_has_no_step_headers(self):
        from oh_my_agent.tools.answer_check import VERIFY_ANSWER_CHECK_SYSTEM

        self.assertNotIn("STEP 1", VERIFY_ANSWER_CHECK_SYSTEM)
        self.assertNotIn("STEP 2", VERIFY_ANSWER_CHECK_SYSTEM)

    def test_verify_prompt_has_few_shot_examples(self):
        from oh_my_agent.tools.answer_check import VERIFY_ANSWER_CHECK_SYSTEM

        self.assertIn("Examples:", VERIFY_ANSWER_CHECK_SYSTEM)
        self.assertIn("Verdict: CORRECT", VERIFY_ANSWER_CHECK_SYSTEM)
        self.assertIn("Verdict: INCORRECT", VERIFY_ANSWER_CHECK_SYSTEM)

    def test_verify_mode_default_max_tokens_is_256(self):
        tool = AnswerCheckTool(mode="verify")
        self.assertEqual(tool.default_max_new_tokens, 256)

    def test_non_verify_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only 'verify' is supported"):
            AnswerCheckTool(mode="strict")

    def test_verify_prompt_covers_spouse_wife_near_miss(self):
        # spouse/spouse_s 应被明确列为 wife/husband 的近似关系
        from oh_my_agent.tools.answer_check import VERIFY_ANSWER_CHECK_SYSTEM
        self.assertIn("spouse", VERIFY_ANSWER_CHECK_SYSTEM.lower())

    def test_verify_prompt_covers_cast_performance_near_miss(self):
        # regular_cast / performance.actor 应被列为 'who plays' 的近似关系
        from oh_my_agent.tools.answer_check import VERIFY_ANSWER_CHECK_SYSTEM
        prompt_lower = VERIFY_ANSWER_CHECK_SYSTEM.lower()
        self.assertTrue(
            "regular_cast" in prompt_lower
            or "performance.actor" in prompt_lower
            or "voice_cast" in prompt_lower,
        )

    def test_verify_prompt_covers_leadership_control_near_miss(self):
        # government_positions_held / leader_of 应被列为 'control/govern' 的近似关系
        from oh_my_agent.tools.answer_check import VERIFY_ANSWER_CHECK_SYSTEM
        prompt_lower = VERIFY_ANSWER_CHECK_SYSTEM.lower()
        self.assertTrue(
            "government_positions_held" in prompt_lower
            or "leader_of" in prompt_lower
            or "president_of" in prompt_lower,
        )


class VerifyModeTests(unittest.TestCase):
    def test_parse_verify_output_valid_path_correct(self):
        parsed = parse_verify_output(
            "P1: VALID — pro_athlete.teams matches 'what team'\n"
            "Match: yes — 'Newcastle Jets FC' ≈ tail of P1\n"
            "Verdict: CORRECT"
        )
        self.assertEqual(parsed["path_verdicts"], {"P1": "VALID"})
        self.assertTrue(parsed["any_valid_path"])
        self.assertEqual(parsed["match"], "yes")
        self.assertEqual(parsed["verdict"], "CORRECT")

    def test_parse_verify_output_all_invalid(self):
        parsed = parse_verify_output(
            "P1: INVALID — starring is about cast, not director\n"
            "Match: no — no valid path to check\n"
            "Verdict: INCORRECT"
        )
        self.assertEqual(parsed["path_verdicts"], {"P1": "INVALID"})
        self.assertFalse(parsed["any_valid_path"])
        self.assertEqual(parsed["match"], "no")
        self.assertEqual(parsed["verdict"], "INCORRECT")

    def test_parse_verify_output_mixed_paths_no_match(self):
        parsed = parse_verify_output(
            "P1: INVALID — starring is about actors\n"
            "P2: VALID — directed_by matches 'who directed'\n"
            "Match: no — 'Spielberg' ≠ 'Nolan'\n"
            "Verdict: INCORRECT"
        )
        self.assertEqual(parsed["path_verdicts"]["P1"], "INVALID")
        self.assertEqual(parsed["path_verdicts"]["P2"], "VALID")
        self.assertTrue(parsed["any_valid_path"])
        self.assertEqual(parsed["match"], "no")
        self.assertEqual(parsed["verdict"], "INCORRECT")

    def test_parse_verify_output_forces_incorrect_when_no_valid_path(self):
        parsed = parse_verify_output(
            "P1: INVALID — wrong relation\n"
            "Match: yes — some match\n"
            "Verdict: CORRECT"
        )
        self.assertFalse(parsed["any_valid_path"])
        self.assertEqual(parsed["verdict"], "INCORRECT")

    def test_parse_verify_output_forces_incorrect_when_match_no(self):
        parsed = parse_verify_output(
            "P1: VALID — relation ok\n"
            "Match: no — answer not found\n"
            "Verdict: CORRECT"
        )
        self.assertEqual(parsed["verdict"], "INCORRECT")

    def test_verify_mode_tool_returns_path_verdicts(self):
        client = FakeLLMClient(
            GenerateResponse(
                text="P1: VALID — teams matches question\nMatch: yes — answer ≈ P1 tail\nVerdict: CORRECT",
                used_adapter=False,
                tokens_generated=15,
                elapsed_ms=5.0,
            )
        )
        tool = AnswerCheckTool(client=client)
        result = tool("what team", ["Newcastle Jets FC"], "  P1: Heskey - [teams] -> Newcastle Jets FC")
        self.assertEqual(result.mode, "verify")
        self.assertEqual(result.path_verdicts["P1"], "VALID")
        self.assertTrue(result.any_valid_path)
        self.assertEqual(result.match, "yes")
        self.assertEqual(result.verdict, "CORRECT")

    def test_verify_mode_prompt_has_paths_before_answers(self):
        client = FakeLLMClient(
            GenerateResponse(
                text="P1: VALID — ok\nMatch: yes — ok\nVerdict: CORRECT",
                used_adapter=False,
                tokens_generated=10,
                elapsed_ms=3.0,
            )
        )
        tool = AnswerCheckTool(client=client)
        tool("what team", ["Newcastle Jets FC"], "  P1: Heskey - [teams] -> Newcastle Jets FC")
        prompt, _ = client.calls[0]
        paths_pos = prompt.index("Paths:")
        answers_pos = prompt.index("Predicted Answers:")
        self.assertLess(paths_pos, answers_pos)

    def test_verify_mode_dynamic_max_tokens_for_many_paths(self):
        client = FakeLLMClient(
            GenerateResponse(
                text="P1: VALID — ok\nMatch: yes — ok\nVerdict: CORRECT",
                used_adapter=False,
                tokens_generated=10,
                elapsed_ms=3.0,
            )
        )
        paths_text = "\n".join(f"  P{i}: Entity - [rel] -> Tail" for i in range(1, 16))
        tool = AnswerCheckTool(client=client)
        tool("what team", ["answer"], paths_text)
        _, kwargs = client.calls[0]
        self.assertGreater(kwargs["max_new_tokens"], 256)

    def test_verify_mode_dynamic_max_tokens_19_paths(self):
        # 19条路径（残余parse error场景）系数×20时给460不够，×30应给650
        client = FakeLLMClient(
            GenerateResponse(
                text="P1: VALID — ok\nMatch: yes — ok\nVerdict: CORRECT",
                used_adapter=False,
                tokens_generated=10,
                elapsed_ms=3.0,
            )
        )
        paths_text = "\n".join(f"  P{i}: Entity - [rel] -> Tail" for i in range(1, 20))
        tool = AnswerCheckTool(client=client)
        tool("what", ["answer"], paths_text)
        _, kwargs = client.calls[0]
        self.assertGreaterEqual(kwargs["max_new_tokens"], 19 * 30 + 80)

    def test_verify_guardrails_allow_mixed_list_with_real_answer(self):
        parsed = {
            "path_verdicts": {"P1": "VALID"},
            "path_reasons": {"P1": "ok"},
            "any_valid_path": True,
            "match": "yes",
            "match_detail": "Gregg Davis matches",
            "verdict": "CORRECT",
        }
        result = apply_verify_guardrails(
            "who did kim richards marry",
            ["Gregg Davis", "m.0bjbnly"],
            parsed,
        )
        self.assertEqual(result["verdict"], "CORRECT")

    def test_verify_guardrails_reject_all_placeholder_answers(self):
        parsed = {
            "path_verdicts": {"P1": "VALID"},
            "path_reasons": {"P1": "ok"},
            "any_valid_path": True,
            "match": "yes",
            "match_detail": "",
            "verdict": "CORRECT",
        }
        result = apply_verify_guardrails(
            "some question",
            ["m.0abc123", "m.0xyz456"],
            parsed,
        )
        self.assertEqual(result["verdict"], "INCORRECT")
        self.assertEqual(result["match"], "no")

    def test_verify_mode_single_path_keeps_default_tokens(self):
        client = FakeLLMClient(
            GenerateResponse(
                text="P1: VALID — ok\nMatch: yes — ok\nVerdict: CORRECT",
                used_adapter=False,
                tokens_generated=10,
                elapsed_ms=3.0,
            )
        )
        tool = AnswerCheckTool(client=client)
        tool("what team", ["answer"], "  P1: Entity - [rel] -> Tail")
        _, kwargs = client.calls[0]
        self.assertEqual(kwargs["max_new_tokens"], 256)


if __name__ == "__main__":
    unittest.main()
