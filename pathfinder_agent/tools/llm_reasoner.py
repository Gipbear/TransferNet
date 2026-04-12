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

# V2 parsing regexes (consistent with eval_faithfulness.py)
_ANSWER_RE      = re.compile(r"Answer\s*[:]\s*(.+)", re.IGNORECASE)
_CITE_RE        = re.compile(r"Supporting\s*Paths?\s*[:]\s*([\d,\s]+)", re.IGNORECASE)
_REJECT_CITE_RE = re.compile(r"Supporting\s*Paths?\s*[:]\s*\(none\)", re.IGNORECASE)

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
    Mirrors eval_faithfulness.py pattern:
      - apply_chat_template without return_tensors -> list of token IDs
      - tokenizer.pad() -> BatchEncoding with 'input_ids' tensor
    This avoids the BatchEncoding-vs-tensor confusion with unsloth's generate().
    """
    # Returns a list of ints (not tensor) when return_tensors is omitted
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    # Wrap in batch dim and pad
    inputs = tokenizer.pad(
        [{"input_ids": token_ids}],
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(device)
    return inputs  # dict with 'input_ids' and 'attention_mask'


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
        answers = [a.strip().strip("\"'[]") for a in re.split(r"[|]", ans_raw) if a.strip()]
    else:
        answers = []

    return answers, cited_indices
