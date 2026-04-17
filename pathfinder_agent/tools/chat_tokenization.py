"""Utilities for preparing chat prompts for generation."""

import torch


def _single_sequence_input_ids(template_output):
    if isinstance(template_output, list):
        input_ids = template_output
    elif isinstance(template_output, tuple):
        input_ids = list(template_output)
    elif isinstance(template_output, dict) or hasattr(template_output, "__getitem__"):
        input_ids = template_output["input_ids"]
    else:
        raise TypeError(f"Unsupported chat template output: {type(template_output)!r}")

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().cpu().tolist()

    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("Expected a single chat template sequence")
        input_ids = input_ids[0]

    return input_ids


def apply_template_and_pad(tokenizer, messages, device):
    """Apply a chat template and return padded tensors for model.generate()."""
    template_output = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    input_ids = _single_sequence_input_ids(template_output)
    return tokenizer.pad(
        [{"input_ids": input_ids}],
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(device)
