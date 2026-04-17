# pathfinder_agent/tools/llm_reasoner.py
"""
LLM Reasoning module: feeds question + retrieved paths to the fine-tuned LLaMA-3.1 adapter
and extracts the (answers, cited_path_indices) tuple.
Output format aligns with V2 of the eval_faithfulness pipeline:
  Supporting Paths: 1, 3
  Answer: entity_name
"""
import re
import torch
from dataclasses import dataclass

from pathfinder_agent.tools.chat_tokenization import apply_template_and_pad

# V2 parsing regexes (consistent with eval_faithfulness.py)
_ANSWER_RE      = re.compile(r"Answer\s*[:]\s*(.+)", re.IGNORECASE)
_CITE_RE        = re.compile(r"Supporting\s*Paths?\s*[:]\s*([\d,\s]+)", re.IGNORECASE)
_REJECT_CITE_RE = re.compile(r"Supporting\s*Paths?\s*[:]\s*\(none\)", re.IGNORECASE)
_MID_RE         = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class _QuestionProfile:
    intent: str
    list_expected: bool


def _question_profile(question: str) -> _QuestionProfile:
    q = f" {question.lower()} "
    list_expected = any(
        token in q
        for token in (
            " who are ",
            " what are ",
            " which ",
            " what countries ",
            " what books ",
            " what songs ",
            " languages ",
            " representatives ",
            " what all ",
            " zip code ",
        )
    )
    if any(token in q for token in (" father", " dad", " mother", " parent")):
        return _QuestionProfile("parent", False)
    if any(token in q for token in (" wife", " husband", " spouse", " married")):
        return _QuestionProfile("spouse", False)
    if any(token in q for token in (" team ", " play for", " played for")):
        return _QuestionProfile("sports_team", list_expected)
    if any(token in q for token in (" president", " governor", " representative", " electorate", " vp ", " vice president")):
        return _QuestionProfile("government", list_expected)
    if any(token in q for token in (" language", " speak", " writing system")):
        return _QuestionProfile("language", list_expected)
    if any(token in q for token in (" where ", " country ", " state ", " county ", " city ", " come from ", " located ", " born ")):
        return _QuestionProfile("location", list_expected)
    if any(token in q for token in (" voice", " voiced", " play in episode", " plays ")):
        return _QuestionProfile("actor_voice", False)
    return _QuestionProfile("unknown", list_expected)


def _relation_family(relation: str) -> str:
    rel = relation.lower()
    if any(token in rel for token in ("parents", "children_reverse", "father", "mother")):
        return "parent"
    if "spouse" in rel or "marriage" in rel:
        return "spouse"
    if any(token in rel for token in ("sports.pro_athlete.teams", "sports_team_roster.team", "sports_team.roster_reverse", "team", "roster")):
        return "sports_team"
    if any(token in rel for token in ("government_position", "office_holder", "vice_president", "representative", "political_district")):
        return "government"
    if any(token in rel for token in ("language", "languages_spoken", "writing_system")):
        return "language"
    if any(token in rel for token in ("place_of_birth", "place_of_death", "containedby", "contains", "location.", "administrative", "capital")):
        return "location"
    if any(token in rel for token in ("actor", "voice", "performance", "starring_roles")):
        return "actor_voice"
    return "other"

SYSTEM_PROMPT = """\
You are a knowledge graph question answering assistant.
You are given a question and a set of evidence paths from a knowledge graph.
Each path is numbered and shows chains of entities and relations.
Your task is to:
1. Select the paths that support the answer.
2. Output the answer entity (or entities, separated by |).

Format your response EXACTLY as:
Supporting Paths: <comma-separated path numbers, or (none) if none apply>
Answer: <answer entity | another entity>"""


def _format_paths(paths: list) -> str:
    """Format path list to numbered arrow-chain strings for the LLM prompt."""
    lines = []
    for i, p in enumerate(paths, start=1):
        edges = p.get("path", [])
        chain = " -> ".join(
            f"{e[0]} -[{e[1]}]-> {e[2]}" for e in edges
        )
        lines.append(f"Path {i}: {chain}")
    return "\n".join(lines)


def _apply_template_and_pad(tokenizer, messages, device):
    """
    Apply chat template and pad to a tensor.
    Kept as a local compatibility wrapper for existing imports/tests.
    """
    return apply_template_and_pad(tokenizer, messages, device)


def _normalize_value(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _supported_entity_strings(paths: list, cited_indices: set[int]) -> set[str]:
    show_indices = cited_indices if cited_indices else set(range(1, len(paths) + 1))
    supported = set()
    for idx, path in enumerate(paths, start=1):
        if idx not in show_indices:
            continue
        for head, _rel, tail in path.get("path", []):
            supported.add(_normalize_value(str(head)))
            supported.add(_normalize_value(str(tail)))
    return supported


def _sanitize_answers(raw_answers: list[str], paths: list, cited_indices: set[int]) -> list[str]:
    supported = _supported_entity_strings(paths, cited_indices)
    cleaned = []
    seen = set()
    for answer in raw_answers:
        candidate = answer.strip().strip("\"'[]")
        normalized = _normalize_value(candidate)
        if not normalized or normalized in seen:
            continue
        if candidate == "No English Label" or _MID_RE.match(candidate):
            continue
        if supported and normalized not in supported:
            continue
        seen.add(normalized)
        cleaned.append(candidate)
    return cleaned


_SINGLE_ANSWER_INTENTS = {"parent", "spouse", "sports_team", "government", "location", "actor_voice"}


def _collapse_single_answer_candidates(
    question: str,
    answers: list[str],
    paths: list,
    cited_indices: set[int],
) -> list[str]:
    profile = _question_profile(question)
    if profile.list_expected or profile.intent not in _SINGLE_ANSWER_INTENTS or len(answers) <= 1:
        return answers

    scoped_paths = [
        path for idx, path in enumerate(paths, start=1)
        if not cited_indices or idx in cited_indices
    ]
    if not scoped_paths:
        return answers

    answer_order = {_normalize_value(answer): idx for idx, answer in enumerate(answers)}
    scored = []
    for answer in answers:
        norm = _normalize_value(answer)
        mention_count = 0
        family_match = 0
        min_tail_hop = 10**6
        for path in scoped_paths:
            path_edges = path.get("path", [])
            matched_this_path = False
            for hop_idx, edge in enumerate(path_edges):
                head, rel, tail = edge
                rel_family = _relation_family(rel)
                if _normalize_value(str(tail)) == norm:
                    matched_this_path = True
                    min_tail_hop = min(min_tail_hop, hop_idx)
                    if rel_family == profile.intent:
                        family_match = 1
                elif _normalize_value(str(head)) == norm and rel_family == profile.intent:
                    matched_this_path = True
            if matched_this_path:
                mention_count += 1

        scored.append(
            (
                -family_match,
                min_tail_hop,
                -mention_count,
                answer_order.get(norm, 10**6),
                answer,
            )
        )

    scored.sort()
    return [scored[0][-1]]


def reason_with_paths(model, tokenizer, question: str, paths: list):
    """
    Invoke the fine-tuned LLaMA-3.1 adapter to generate answer from retrieved paths.
    Returns:
        candidate_answers (list[str])
        used_path_indices  (set[int])  -- 1-based
    """
    if not paths:
        return [], set()

    path_text = _format_paths(paths)
    user_content = (
        f"Question: {question}\n\n"
        f"Evidence Paths:\n{path_text}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    inputs = _apply_template_and_pad(tokenizer, messages, model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    raw = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()

    # Parse V2 format
    cite_m      = _CITE_RE.search(raw)
    reject_cite = bool(_REJECT_CITE_RE.search(raw))
    answer_m    = _ANSWER_RE.search(raw)

    cited_indices = set()
    if cite_m and not reject_cite:
        for tok in re.split(r"[,\s]+", cite_m.group(1)):
            if tok.strip().isdigit():
                cited_indices.add(int(tok.strip()))

    if answer_m:
        ans_raw = answer_m.group(1).strip().splitlines()[0]
        raw_answers = [a.strip().strip("\"'[]") for a in re.split(r"[|]", ans_raw) if a.strip()]
        answers = _sanitize_answers(raw_answers, paths, cited_indices)
        answers = _collapse_single_answer_candidates(question, answers, paths, cited_indices)
    else:
        answers = []

    return answers, cited_indices
