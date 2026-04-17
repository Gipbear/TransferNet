# pathfinder_agent/tools/answer_aggregator.py
"""Answer Aggregator: deterministically merge candidate answers."""


def aggregate_answers(model, tokenizer, answer_lists: list, question: str = ""):
    """
    Merge answers from multiple retrieval/reasoning rounds.

    A stable, deterministic union is applied. We intentionally avoid a
    second generative cleanup pass here because logs showed it could replace
    validated candidates with unsupported answers.

    Args:
        answer_lists: list of list[str], one per rewritten query
        question:     original question string (used in LLM cleanup prompt)
    Returns:
        final_answers: list[str]
    """
    # -- Step 1: flat union (case-insensitive dedup, preserve order) --
    seen = set()
    merged = []
    for al in answer_lists:
        for ans in al:
            key = ans.strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(ans.strip())

    if not merged:
        return []

    return merged

