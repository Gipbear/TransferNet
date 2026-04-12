# pathfinder_agent/tools/answer_verifier.py
"""
Answer Verifier: uses the LLaMA-3.1 model to verify whether the candidate answer is
semantically aligned with the question's target slot.
Returns (is_valid: bool, feedback: str).

Detects:
  - Slot mismatch (e.g. question asks for "father" but answer is a "spouse")
  - Entity surface-form splits (e.g. a comma-separated name split into two answers)
"""
import re
import torch

VERIFIER_SYSTEM_PROMPT = """\
You are a quality-checking agent for a knowledge-graph question answering system.
You will be given:
  - A question
  - A list of evidence paths used to generate the answer
  - A candidate answer

Your job is to decide whether the candidate answer is VALID or INVALID.
An answer is INVALID if:
  1. SLOT MISMATCH: The answer entity belongs to a different category than what the question asks for.
     Example: question asks "who is X's father" but the answer is a spouse or a sibling.
  2. SURFACE SPLIT: A single entity name was incorrectly split into multiple items
     (e.g. "Queen Elizabeth, The Queen Mother" split into ["Queen Elizabeth", "The Queen Mother"]).
  3. HALLUCINATION: The answer entity does not appear in any of the provided evidence paths.

If the answer is VALID, respond with exactly: VALID
If the answer is INVALID, respond with: INVALID: <brief one-line reason>"""


def _format_paths_for_verifier(paths, used_indices):
    """Show only the cited paths (1-based), or all if none cited."""
    show_indices = used_indices if used_indices else set(range(1, len(paths) + 1))
    lines = []
    for i, p in enumerate(paths, start=1):
        if i in show_indices:
            edges = p.get("path", [])
            chain = " -> ".join(f"{e[0]} -[{e[1]}]-> {e[2]}" for e in edges)
            lines.append(f"Path {i}: {chain}")
    return "\n".join(lines) if lines else "(no paths available)"


def verify_answer(model, tokenizer, question: str, candidate_answers: list,
                  used_path_indices: set, all_paths: list):
    """
    Verify the candidate answers.
    Returns (is_valid: bool, feedback: str)
    """
    if not candidate_answers:
        return False, "No answer was generated."

    path_text   = _format_paths_for_verifier(all_paths, used_path_indices)
    answer_text = " | ".join(candidate_answers)

    user_content = (
        f"Question: {question}\n\n"
        f"Evidence Paths Used:\n{path_text}\n\n"
        f"Candidate Answer: {answer_text}"
    )
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    # Use tokenizer.pad() pattern (same as eval_faithfulness.py) to avoid
    # BatchEncoding-vs-tensor issues with unsloth's generate()
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
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    verdict = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()

    if verdict.upper().startswith("VALID") and "INVALID" not in verdict.upper():
        return True, "OK"
    else:
        feedback = verdict if verdict else "Verifier returned no output"
        return False, feedback
