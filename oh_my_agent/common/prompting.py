"""Prompt builders for the simple QA agent."""

from __future__ import annotations


SYSTEM_PROMPT_V2_NAME = (
    "You are a KGQA assistant. "
    "Given reasoning paths from a knowledge graph and a question, "
    "identify which paths support the answer, then extract the answer "
    "from the tail entities of those supporting paths.\n"
    "Rules:\n"
    "- Only output entity names that appear in the provided paths.\n"
    "- Do not generate or fabricate new entity names.\n"
    "Output format:\n"
    "Supporting Paths: <path numbers>\n"
    "Answer: <entity_name> | <entity_name>"
)


def format_chain(path_edges: list) -> str:
    """Serialize path edges as a plain chain."""
    if not path_edges:
        return ""
    parts = [path_edges[0][0]]
    for _, rel, tail in path_edges:
        parts.extend([rel, tail])
    return " -> ".join(parts)


def build_reasoning_prompt(question: str, named_paths: list[dict]) -> str:
    """Render the user prompt with named reasoning paths (plain chain format)."""
    lines = [f"Question: {question}", "", "Reasoning Paths:"]
    for index, path_dict in enumerate(named_paths, start=1):
        chain = format_chain(path_dict.get("path", []))
        lines.append(f"{index}: {chain}")
    return "\n".join(lines)
