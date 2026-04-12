# pathfinder_agent/tools/query_rewriter.py
import re
import torch

REWRITER_SYSTEM_PROMPT = """\
You are a question-rewriting assistant for a knowledge graph QA system.
Your task is to rewrite the given question into 2 semantically similar but differently phrased variants
to improve retrieval recall.
Constraints:
1. Every variant MUST include the topic entity wrapped in square brackets exactly as shown.
2. Do NOT replace or omit the original entity name.
Output only the rewritten sentences, one per line, with no extra explanation."""


def rewrite_question(model, tokenizer, original_question, topic_entity):
    """
    Rewrite the original question into 2 semantically similar variants.
    Constraint: all variants must preserve the topic entity enclosed in square brackets,
    e.g. [David Beckham], so the TransferNet DataLoader can parse it correctly.
    """
    user_content = (
        f"Original question: {original_question}\n"
        f"Topic entity: [{topic_entity}]"
    )
    messages = [
        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
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
        # Only keep lines that still contain the topic entity (in any bracket form)
        if line and topic_entity.lower() in line.lower():
            cleaned_line = re.sub(r'^(\d+\.|-)\s*', '', line)
            if cleaned_line and cleaned_line not in rewritten_queries:
                rewritten_queries.append(cleaned_line)

    # Cap at 3 variants total (original + 2 rewrites)
    return rewritten_queries[:3]
