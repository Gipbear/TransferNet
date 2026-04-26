"""LLM answer-check tool for validating KGQA predictions against cited paths."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from oh_my_agent.common.prompting import _format_schema_chain
from oh_my_agent.llm_server import LLMClient


LEGACY_VERIFY_ANSWER_CHECK_SYSTEM = (
    "Verify a KGQA answer by checking path relations then answer match.\n\n"
    "For each cited path, label it VALID or INVALID:\n"
    "  VALID: the relation type directly answers the question's slot.\n"
    "  INVALID: the relation targets a different concept.\n"
    "  Loose standard: broader and near-miss relations count as VALID.\n"
    "    e.g. 'parents' covers 'father/mother'; 'spouse/spouse_s' covers 'wife/husband';\n"
    "    'starring/regular_cast' covers 'who plays/acts/voices';\n"
    "    'government_positions_held' covers 'control/rule/govern'.\n"
    "  For multi-hop paths, every hop must be appropriate.\n\n"
    "For VALID paths only: check if any predicted answer matches the tail entity of the path's final edge.\n"
    "  Forward A - [rel] -> B: B is the tail.\n"
    "  Reverse A <- [rel] - B: B is the tail.\n"
    "  Accept aliases and name variants as matches.\n"
    "  If no path is VALID, Match must be 'no' and Verdict must be INCORRECT.\n\n"
    "Output ONLY the following lines — no headers, no step labels, no extra text:\n"
    "P1: <VALID|INVALID> — <reason>\n"
    "P2: <VALID|INVALID> — <reason>\n"
    "Match: <yes|no> — <answer ≈ tail of Px, or none>\n"
    "Verdict: <CORRECT|INCORRECT>\n\n"
    "Examples:\n\n"
    "Q: What team does Heskey play for?\n"
    "Paths:\n"
    "  P1: Emile Heskey - [sports.pro_athlete.teams] -> Newcastle Jets FC\n"
    "  P2: Emile Heskey - [sports.pro_athlete.position] -> Forward\n"
    "Predicted Answers:\n"
    "  A1: Newcastle Jets FC\n"
    "P1: VALID — pro_athlete.teams matches 'what team'\n"
    "P2: INVALID — position is not the asked slot\n"
    "Match: yes — 'Newcastle Jets FC' ≈ tail of P1\n"
    "Verdict: CORRECT\n\n"
    "Q: Who directed Inception?\n"
    "Paths:\n"
    "  P1: Inception - [film.film.starring] -> Leonardo DiCaprio\n"
    "Predicted Answers:\n"
    "  A1: Steven Spielberg\n"
    "P1: INVALID — starring is about cast, not director\n"
    "Match: no — none\n"
    "Verdict: INCORRECT\n\n"
    "Q: What is Obama's father's name?\n"
    "Paths:\n"
    "  P1: Barack Obama - [people.person.parents] -> Barack Obama Sr.\n"
    "  P2: Barack Obama - [people.person.sibling_s] -> Auma Obama\n"
    "Predicted Answers:\n"
    "  A1: Barack Obama Sr.\n"
    "P1: VALID — parents is broad enough to cover 'father'\n"
    "P2: INVALID — sibling is not the asked slot\n"
    "Match: yes — 'Barack Obama Sr.' ≈ tail of P1\n"
    "Verdict: CORRECT\n\n"
    "Q: Who is Niall Ferguson's wife?\n"
    "Paths:\n"
    "  P1: Niall Ferguson - [people.person.spouse_s] -> Ayaan Hirsi Ali\n"
    "  P2: Niall Ferguson - [people.person.children] -> Thomas Ferguson\n"
    "Predicted Answers:\n"
    "  A1: Ayaan Hirsi Ali\n"
    "P1: VALID — spouse_s covers 'wife' (family role near-miss)\n"
    "P2: INVALID — children is not the asked slot\n"
    "Match: yes — 'Ayaan Hirsi Ali' ≈ tail of P1\n"
    "Verdict: CORRECT\n\n"
    "Q: Who plays the voice of KITT in Knight Rider?\n"
    "Paths:\n"
    "  P1: Knight Rider - [tv.tv_program.regular_cast] -> William Daniels\n"
    "  P2: Knight Rider - [tv.tv_program.country_of_origin] -> United States\n"
    "Predicted Answers:\n"
    "  A1: William Daniels\n"
    "P1: VALID — regular_cast covers 'who plays the voice of' (performance near-miss)\n"
    "P2: INVALID — country_of_origin is not the asked slot\n"
    "Match: yes — 'William Daniels' ≈ tail of P1\n"
    "Verdict: CORRECT"
)

VERIFY_ANSWER_CHECK_SYSTEM = (
    "Verify a KGQA answer in two stages: relation fit, then answer match.\n\n"
    "Path rule:\n"
    "  VALID: relation answers the asked slot.\n"
    "  INVALID: relation points to a different slot or concept.\n"
    "  Loose near-miss rules still count as VALID:\n"
    "    parents -> father/mother\n"
    "    spouse/spouse_s -> wife/husband\n"
    "    starring/regular_cast -> plays/voices\n"
    "    government_positions_held -> control/govern\n"
    "  For multi-hop paths, every hop must fit.\n\n"
    "Match rule:\n"
    "  Only check VALID paths.\n"
    "  Compare predicted answers to the tail entity of the final edge.\n"
    "  Forward A - [rel] -> B: tail is B.\n"
    "  Reverse A <- [rel] - B: tail is B.\n"
    "  Accept aliases and name variants.\n"
    "  If no path is VALID, Match must be no and Verdict must be INCORRECT.\n\n"
    "Output only these lines, with short reasons and no extra text:\n"
    "P1: <VALID|INVALID> - <short reason>\n"
    "P2: <VALID|INVALID> - <short reason>\n"
    "Match: <yes|no> - <P ids or none>\n"
    "Verdict: <CORRECT|INCORRECT>\n\n"
    "Examples:\n\n"
    "Q: What is Obama's father's name?\n"
    "Paths:\n"
    "  P1: Barack Obama - [people.person.parents] -> Barack Obama Sr.\n"
    "  P2: Barack Obama - [people.person.sibling_s] -> Auma Obama\n"
    "Predicted Answers:\n"
    "  A1: Barack Obama Sr.\n"
    "P1: VALID - parents covers father\n"
    "P2: INVALID - sibling wrong slot\n"
    "Match: yes - P1\n"
    "Verdict: CORRECT\n\n"
    "Q: Who directed Inception?\n"
    "Paths:\n"
    "  P1: Inception - [film.film.starring] -> Leonardo DiCaprio\n"
    "Predicted Answers:\n"
    "  A1: Steven Spielberg\n"
    "P1: INVALID - starring not director\n"
    "Match: no - none\n"
    "Verdict: INCORRECT"
)

PLACEHOLDER_RE = re.compile(r"^(?:m|g)\.[A-Za-z0-9_]+$")
_PATH_LINE_RE = re.compile(r"^P(\d+):\s*(VALID|INVALID)\s*[—–-]\s*(.*)", re.I)
_MATCH_LINE_RE = re.compile(r"^Match:\s*(yes|no)\s*(?:[—–-]\s*(.*))?$", re.I)


@dataclass(frozen=True)
class AnswerCheckToolResult:
    """Structured output from the answer-check tool."""

    mode: str
    question: str
    pred_answers: list[str]
    prompt: str
    raw_output: str
    verdict: str
    tokens_generated: int
    elapsed_ms: float
    path_verdicts: dict[str, str] = field(default_factory=dict)
    path_reasons: dict[str, str] = field(default_factory=dict)
    any_valid_path: bool = False
    match: str = ""
    match_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_paths_text(record: dict[str, Any]) -> str:
    named_paths = record.get("named_mmr_reason_paths", [])
    # cited_path_indices is 1-based; 0 is filtered out
    cited_indices = set(record.get("cited_path_indices", []))
    cited_paths = [
        named_paths[i - 1]["path"]
        for i in sorted(cited_indices)
        if 0 < i <= len(named_paths)
    ]
    if not cited_paths:
        return "(no cited paths)"
    return "\n".join(
        f"  P{idx + 1}: {_format_schema_chain(path)}"
        for idx, path in enumerate(cited_paths)
    )


def build_verify_prompt(question: str, paths_text: str, pred_answers: list[str]) -> str:
    """Paths-first layout so the LLM evaluates relations before seeing predicted answers."""
    if pred_answers:
        answer_text = "\n".join(
            f"  A{idx + 1}: {answer}"
            for idx, answer in enumerate(pred_answers)
        )
    else:
        answer_text = "  (none)"
    return f"Q: {question}\nPaths:\n{paths_text}\nPredicted Answers:\n{answer_text}"


def parse_verify_output(text: str) -> dict[str, Any]:
    """Parse the structured output from the verify-mode LLM."""
    path_verdicts: dict[str, str] = {}
    path_reasons: dict[str, str] = {}
    match_val = ""
    match_detail = ""
    verdict = "PARSE_ERROR"

    for line in text.strip().splitlines():
        line = line.strip()
        if m := _PATH_LINE_RE.match(line):
            pid = f"P{m.group(1)}"
            path_verdicts[pid] = m.group(2).upper()
            path_reasons[pid] = m.group(3).strip()
        elif m := _MATCH_LINE_RE.match(line):
            match_val = m.group(1).lower()
            match_detail = (m.group(2) or "").strip()
        elif m := re.match(r"Verdict:\s*(CORRECT|INCORRECT)", line, re.I):
            verdict = m.group(1).upper()

    any_valid = any(v == "VALID" for v in path_verdicts.values())

    # Consistency enforcement: no valid path -> always INCORRECT regardless of LLM output.
    if not any_valid:
        verdict = "INCORRECT"
        match_val = match_val or "no"

    # Match=no overrides a CORRECT verdict; Match=yes with a valid path upgrades INCORRECT.
    if match_val == "no" and verdict == "CORRECT":
        verdict = "INCORRECT"
    if match_val == "yes" and any_valid and verdict == "INCORRECT":
        verdict = "CORRECT"

    return {
        "path_verdicts": path_verdicts,
        "path_reasons": path_reasons,
        "any_valid_path": any_valid,
        "match": match_val,
        "match_detail": match_detail,
        "verdict": verdict,
    }


def is_placeholder_answer(answer: str) -> bool:
    answer = answer.strip()
    return (
        not answer
        or "XMLSchema#" in answer
        or PLACEHOLDER_RE.match(answer) is not None
    )


def apply_verify_guardrails(question: str, pred_answers: list[str], parsed: dict[str, Any]) -> dict[str, Any]:
    """Reject verify results only when every predicted answer is a placeholder."""
    if pred_answers and all(is_placeholder_answer(a) for a in pred_answers):
        parsed["match"] = "no"
        parsed["match_detail"] = "placeholder answers rejected"
        parsed["verdict"] = "INCORRECT"
    return parsed


class AnswerCheckTool:
    """Tool wrapper around the local LLM server for answer verification."""

    def __init__(
        self,
        *,
        client: LLMClient | None = None,
        base_url: str = "http://localhost:8788",
        mode: str = "verify",
        default_use_adapter: bool = False,
        default_max_new_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> None:
        if mode != "verify":
            raise ValueError(f"Unsupported answer-check mode: {mode}; only 'verify' is supported")
        self.client = client or LLMClient(base_url)
        self.mode = mode
        self.default_use_adapter = default_use_adapter
        self.default_max_new_tokens = 256 if default_max_new_tokens is None else default_max_new_tokens
        self.system_prompt = VERIFY_ANSWER_CHECK_SYSTEM if system_prompt is None else system_prompt

    def __call__(
        self,
        question: str,
        pred_answers: list[str],
        paths_text: str,
        *,
        use_adapter: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> AnswerCheckToolResult:
        use_adapter = self.default_use_adapter if use_adapter is None else use_adapter
        max_new_tokens = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens

        n_paths = len(re.findall(r"^\s*P\d+:", paths_text, re.MULTILINE))
        if n_paths > 1:
            max_new_tokens = max(max_new_tokens, n_paths * 30 + 80)

        prompt = build_verify_prompt(question, paths_text, pred_answers)
        response = self.client.generate(
            prompt,
            use_adapter=use_adapter,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            system_prompt=self.system_prompt,
        )
        parsed = parse_verify_output(response.text)
        parsed = apply_verify_guardrails(question, pred_answers, parsed)
        return AnswerCheckToolResult(
            mode=self.mode,
            question=question,
            pred_answers=pred_answers,
            prompt=prompt,
            raw_output=response.text,
            verdict=parsed["verdict"],
            tokens_generated=response.tokens_generated,
            elapsed_ms=response.elapsed_ms,
            path_verdicts=parsed["path_verdicts"],
            path_reasons=parsed["path_reasons"],
            any_valid_path=parsed["any_valid_path"],
            match=parsed["match"],
            match_detail=parsed["match_detail"],
        )
