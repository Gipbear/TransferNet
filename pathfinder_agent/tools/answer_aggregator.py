# pathfinder_agent/tools/answer_aggregator.py
"""
Answer Aggregator: merge (union) candidate answers from multiple query variants.
Uses a simple union strategy; optionally calls the LLM for deduplication / ranking
when the merged set is large.
"""
import re
import torch

AGGREGATOR_SYSTEM_PROMPT = """\
You are an answer post-processing assistant for a question answering system.
You will receive a question and a list of candidate answer entities gathered from
multiple retrieval rounds.
Your job is to:
1. Remove exact duplicates (case-insensitive).
2. Remove entities that are clearly irrelevant to the question's target slot
   (e.g., if the question asks for a country, remove city-level answers only if
   a country-level answer is present).
3. Output the final deduplicated and ranked answer list, one entity per line.
Output ONLY the final entities, one per line, with no extra explanation."""


def aggregate_answers(model, tokenizer, answer_lists: list, question: str = ""):
    """
    Merge answers from multiple retrieval/reasoning rounds.

    Simple union is applied first. If the merged set has more than 1 entry,
    we optionally call the LLM for a light cleanup pass.

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

    # -- Step 2: if only one variant or one answer, skip LLM cleanup --
    if len(answer_lists) <= 1 or len(merged) == 1:
        return merged

    # -- Step 3: LLM cleanup for multi-answer scenarios --
    if model is None or tokenizer is None:
        return merged

    candidates_text = "\n".join(f"- {a}" for a in merged)
    user_content = (
        f"Question: {question}\n\n"
        f"Candidate Answers:\n{candidates_text}"
    )
    messages = [
        {"role": "system", "content": AGGREGATOR_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    # Use tokenizer.pad() pattern to avoid BatchEncoding-vs-tensor issues
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    inputs = tokenizer.pad(
        [{"input_ids": token_ids}],
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    raw = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()

    final = [
        line.lstrip("-•* ").strip()
        for line in raw.splitlines()
        if line.strip()
    ]
    return final if final else merged

