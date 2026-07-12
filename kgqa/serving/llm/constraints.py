"""Constrained decoding for reject-list outputs.

The reject-list grammar accepts exactly:
- "NONE", or
- comma-separated distinct indices in [1, max_option_index], no leading zeros.

`build_reject_list_prefix_fn` returns a `prefix_allowed_tokens_fn` for
transformers `generate`, restricting output tokens so any decoded text stays a
valid prefix of the grammar.
"""

from __future__ import annotations

from typing import Callable


def _split_segments(text: str) -> list[str] | None:
    """Split a digits/commas body into segments; None if malformed."""
    if not text:
        return []
    if any(ch not in "0123456789," for ch in text):
        return None
    segments = text.split(",")
    # 中间出现空段（如 "1,,2" 或开头是逗号）非法；结尾空段表示刚输出逗号，合法
    for segment in segments[:-1]:
        if not segment:
            return None
    return segments


def _valid_number(segment: str, max_index: int) -> bool:
    if not segment or segment[0] == "0":
        return False
    return int(segment) <= max_index


def valid_reject_list_prefix(text: str, max_index: int) -> bool:
    """Whether text can still be extended into a valid reject-list output."""
    if not text:
        return True
    if text[0] == "N":
        return "NONE".startswith(text)

    segments = _split_segments(text)
    if segments is None:
        return False
    if segments[-1] == "":
        complete_segments = segments[:-1]
        partial = None
    else:
        complete_segments = segments[:-1]
        partial = segments[-1]

    seen: set[int] = set()
    for segment in complete_segments:
        if not _valid_number(segment, max_index):
            return False
        value = int(segment)
        if value in seen:
            return False
        seen.add(value)

    if partial is not None:
        if not _valid_number(partial, max_index):
            return False
        if int(partial) in seen:
            return False
    elif len(seen) >= max_index:
        # 以逗号结尾但所有编号已用完，没有合法的后续数字
        return False
    return True


def reject_list_complete(text: str, max_index: int) -> bool:
    """Whether text is already a complete, valid reject-list output."""
    if text == "NONE":
        return True
    if not text or text[0] == "N" or text.endswith(","):
        return False
    return valid_reject_list_prefix(text, max_index)


def _token_whitelist(tokenizer) -> dict[int, str]:
    """Token ids able to spell digits, commas, and NONE."""
    whitelist: dict[int, str] = {}
    for piece in [str(digit) for digit in range(10)] + [",", "NONE"]:
        for token_id in tokenizer.encode(piece, add_special_tokens=False):
            whitelist[token_id] = tokenizer.decode([token_id])
    return whitelist


def build_reject_list_prefix_fn(
    tokenizer,
    prompt_length: int,
    max_index: int,
) -> Callable:
    """Build a prefix_allowed_tokens_fn enforcing the reject-list grammar."""
    whitelist = _token_whitelist(tokenizer)
    eos_token_id = tokenizer.eos_token_id

    def prefix_allowed_tokens_fn(batch_id: int, input_ids) -> list[int]:
        generated = input_ids[prompt_length:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        allowed = [
            token_id
            for token_id, piece in whitelist.items()
            if valid_reject_list_prefix(text + piece, max_index)
        ]
        if reject_list_complete(text, max_index):
            allowed.append(eos_token_id)
        if not allowed:
            allowed = [eos_token_id]
        return allowed

    return prefix_allowed_tokens_fn
