# pathfinder_agent/tools/query_rewriter.py
import re
import torch

from pathfinder_agent.tools.chat_tokenization import apply_template_and_pad

REWRITER_SYSTEM_PROMPT = """\
You are a question-rewriting assistant for a knowledge graph QA system.
Your task is to rewrite the given question into 2 semantically similar but differently phrased variants
to improve retrieval recall.
Constraints:
1. Every variant MUST include the topic entity wrapped in square brackets exactly as shown.
2. Do NOT replace or omit the original entity name.
Output only the rewritten sentences, one per line, with no extra explanation."""

QUESTION_PREFIXES = (
    "what", "who", "where", "when", "why", "which", "how",
    "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "will", "would", "should", "has", "have", "had",
)

META_PHRASES = (
    "correct question",
    "rewritten variant",
    "rewritten variants",
    "here are",
    "the question should",
    "should be about",
)

_MID_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")


def _constraint_markers(question: str) -> list[str]:
    lowered = f" {question.lower()} "
    markers = []
    for token in (" first ", " second ", " third ", " when ", " before ", " after ", " during "):
        if token in lowered:
            markers.append(token.strip())

    for piece in lowered.split():
        if piece.isdigit() and len(piece) == 4:
            markers.append(piece)

    if " second term " in lowered:
        markers.append("second term")
    if " first term " in lowered:
        markers.append("first term")
    return markers


def _preserves_constraints(question: str, candidate: str) -> bool:
    required = _constraint_markers(question)
    if not required:
        return True

    lowered = candidate.lower()
    return all(marker in lowered for marker in required)


def _looks_like_question_variant(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if any(phrase in lowered for phrase in META_PHRASES):
        return False
    first_token = re.sub(r'^[\[\("]+', '', stripped).split(maxsplit=1)[0].lower()
    return stripped.endswith("?") or first_token in QUESTION_PREFIXES


def _variant_mentions_topic_entity(candidate: str, topic_entity: str) -> bool:
    if _MID_RE.match((topic_entity or "").strip()):
        return True
    return topic_entity.lower() in candidate.lower()


def _build_user_content(original_question: str, topic_entity: str) -> str:
    if _MID_RE.match((topic_entity or "").strip()):
        return (
            f"Original question: {original_question}\n"
            "Preserve the main entity mention from the original question in every rewrite."
        )
    return (
        f"Original question: {original_question}\n"
        f"Topic entity: [{topic_entity}]"
    )


def rewrite_question(model, tokenizer, original_question, topic_entity):
    """
    Rewrite the original question into 2 semantically similar variants.
    Constraint: all variants must preserve the topic entity enclosed in square brackets,
    e.g. [David Beckham], so the TransferNet DataLoader can parse it correctly.
    """
    user_content = _build_user_content(original_question, topic_entity)
    messages = [
        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    inputs = apply_template_and_pad(tokenizer, messages, model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()

    rewritten_queries = [original_question]
    for line in response.split('\n'):
        line = line.strip()
        cleaned_line = re.sub(r'^(\d+\.|-)\s*', '', line)
        if (
            cleaned_line
            and _looks_like_question_variant(cleaned_line)
            and _variant_mentions_topic_entity(cleaned_line, topic_entity)
            and _preserves_constraints(original_question, cleaned_line)
            and cleaned_line not in rewritten_queries
        ):
            rewritten_queries.append(cleaned_line)

    # Cap at 3 variants total (original + 2 rewrites)
    return rewritten_queries[:3]
