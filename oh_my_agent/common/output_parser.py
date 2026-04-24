"""Parsers for LLM answer formats."""

from __future__ import annotations

from dataclasses import dataclass
import re


REJECTION_SENTINEL = "(none)"

_ANSWER_RE = re.compile(r"Answer\s*[:：]\s*(.+)", re.IGNORECASE)
_CITE_RE = re.compile(r"Supporting\s*Paths?\s*[:：]\s*([\d,\s]+)", re.IGNORECASE)
_REJECT_CITE_RE = re.compile(r"Supporting\s*Paths?\s*[:：]\s*\(none\)", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"^entity\d*$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedV2Output:
    """Parsed fields from the V2 text output format."""

    answers: list[str]
    cited_indices: list[int]
    format_ok: bool


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _parse_answers(raw_answers: str) -> list[str]:
    answers = []
    for raw_value in raw_answers.split("|"):
        answer = raw_value.strip().strip("\"'[]")
        if answer and not _PLACEHOLDER_RE.match(answer):
            answers.append(answer)
    return _dedupe_preserve_order(answers)


def parse_v2_output(raw_text: str) -> ParsedV2Output:
    """Parse the V2 text answer format used by the simple agent."""
    raw_text = (raw_text or "").strip()
    cite_match = _CITE_RE.search(raw_text)
    reject_cite = bool(_REJECT_CITE_RE.search(raw_text))
    answer_match = _ANSWER_RE.search(raw_text)
    format_ok = bool((cite_match or reject_cite) and answer_match)

    cited_indices: list[int] = []
    if cite_match and not reject_cite:
        seen = set()
        for token in re.split(r"[,\s]+", cite_match.group(1)):
            token = token.strip()
            if token.isdigit():
                index = int(token)
                if index not in seen:
                    cited_indices.append(index)
                    seen.add(index)

    if reject_cite and answer_match and REJECTION_SENTINEL in answer_match.group(1).lower():
        answers = [REJECTION_SENTINEL]
    else:
        answers = (
            _parse_answers(answer_match.group(1).strip().splitlines()[0])
            if answer_match
            else []
        )

    return ParsedV2Output(
        answers=answers,
        cited_indices=cited_indices,
        format_ok=format_ok,
    )
