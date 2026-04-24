"""WebQSP QA text parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
import re


_QUESTION_BOUNDARY_TOKEN_RE = re.compile(r"\s*(?:\[CLS\]|\[SEP\])\s*")
_WORDPIECE_MARKER_RE = re.compile(r"\s*##\s*")


def clean_question_text(question: str) -> str:
    """Remove WebQSP BERT boundary tokens and wordpiece markers."""
    question = _QUESTION_BOUNDARY_TOKEN_RE.sub(" ", question or "")
    question = _WORDPIECE_MARKER_RE.sub("", question)
    return " ".join(question.split()).strip()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


@dataclass(frozen=True)
class WebQSPQASample:
    """A single QA sample from qa_test_webqsp_fixed.txt."""

    question_raw: str
    question: str
    topic_mid: str
    gold_mids: list[str]


def parse_webqsp_qa_line(line: str) -> WebQSPQASample:
    """Parse one WebQSP QA line into structured fields."""
    raw_line = line.strip()
    if not raw_line:
        raise ValueError("QA line is empty")

    parts = raw_line.split("\t")
    raw_question = parts[0].strip()
    answers = parts[1].split("|") if len(parts) > 1 and parts[1].strip() else []

    topic_mid = ""
    if " [" in raw_question and raw_question.endswith("]"):
        raw_question, topic_mid = raw_question.rsplit(" [", 1)
        topic_mid = topic_mid[:-1]

    return WebQSPQASample(
        question_raw=raw_question,
        question=clean_question_text(raw_question),
        topic_mid=topic_mid,
        gold_mids=_dedupe_preserve_order([answer.strip() for answer in answers]),
    )


def load_webqsp_qa_samples(path: str, limit: int = 0) -> list[WebQSPQASample]:
    """Load WebQSP QA samples from a text file."""
    samples: list[WebQSPQASample] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            samples.append(parse_webqsp_qa_line(line))
            if limit > 0 and len(samples) >= limit:
                break
    return samples
