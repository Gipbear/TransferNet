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


def _format_schema_chain(path_edges: list) -> str:
    """Serialize path edges as a schema-aware chain.

    Forward edge: E - [rel] -> E
    Reverse edge (*_reverse): E <- [base_rel] - E
    Multi-hop: E0 - [r1] -> E1 <- [r2] - E2
    """
    if not path_edges:
        return ""
    parts = [path_edges[0][0]]
    for head, rel, tail in path_edges:
        if parts[-1] != head:
            parts.append(head)
        if rel.endswith("_reverse"):
            base_rel = rel[: -len("_reverse")]
            arrow = f"<- [{base_rel}] -"
        else:
            arrow = f"- [{rel}] ->"
        parts.extend([arrow, tail])
    return " ".join(parts)


def build_reasoning_prompt(question: str, named_paths: list[dict]) -> str:
    """Render the user prompt with named reasoning paths (schema format)."""
    lines = [f"Question: {question}", "", "Reasoning Paths:"]
    for index, path_dict in enumerate(named_paths, start=1):
        chain = _format_schema_chain(path_dict.get("path", []))
        lines.append(f"{index}: {chain}")
    return "\n".join(lines)
