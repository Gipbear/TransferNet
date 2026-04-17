import sys
import unittest
from pathlib import Path

import torch
from transformers.tokenization_utils_base import BatchEncoding

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathfinder_agent.tools.query_rewriter import rewrite_question


class _FakePaddedInputs(dict):
    def to(self, device):
        return self


class _FakeModel:
    device = "cpu"

    def generate(self, **inputs):
        prompt = inputs["input_ids"][0].tolist()
        return torch.tensor([prompt + [999]])


class _BatchEncodingTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.padded_input_ids = None
        self.messages = None

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        self.messages = messages
        return BatchEncoding({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]})

    def pad(self, encoded_inputs, return_tensors=None, padding=None, padding_side=None):
        self.padded_input_ids = encoded_inputs[0]["input_ids"]
        if isinstance(self.padded_input_ids, BatchEncoding):
            raise ValueError("BatchEncoding should be normalized before tokenizer.pad")
        return _FakePaddedInputs(
            {
                "input_ids": torch.tensor([self.padded_input_ids]),
                "attention_mask": torch.ones(1, len(self.padded_input_ids), dtype=torch.long),
            }
        )

    def decode(self, token_ids, skip_special_tokens=True):
        return "1. Which language do people in [Jamaica] speak?"


class QueryRewriterTest(unittest.TestCase):
    def test_rewrite_question_normalizes_batch_encoding_template_output(self):
        tokenizer = _BatchEncodingTokenizer()

        rewritten = rewrite_question(
            _FakeModel(),
            tokenizer,
            "what does jamaican people speak",
            "Jamaica",
        )

        self.assertEqual(tokenizer.padded_input_ids, [1, 2, 3])
        self.assertEqual(
            rewritten,
            [
                "what does jamaican people speak",
                "Which language do people in [Jamaica] speak?",
            ],
        )

    def test_rewrite_question_preserves_temporal_constraints_in_variants(self):
        tokenizer = _BatchEncodingTokenizer()
        tokenizer.decode = lambda *_args, **_kwargs: "\n".join(
            [
                "What team did [David Beckham] play for?",
                "Which club did [David Beckham] play for in 2011?",
                "Who did [David Beckham] play for first in 2011?",
            ]
        )

        rewritten = rewrite_question(
            _FakeModel(),
            tokenizer,
            "what team did david beckham play for in 2011",
            "David Beckham",
        )

        self.assertEqual(
            rewritten,
            [
                "what team did david beckham play for in 2011",
                "Which club did [David Beckham] play for in 2011?",
                "Who did [David Beckham] play for first in 2011?",
            ],
        )

    def test_rewrite_question_keeps_natural_language_variants_when_topic_entity_is_mid(self):
        tokenizer = _BatchEncodingTokenizer()
        tokenizer.decode = lambda *_args, **_kwargs: "\n".join(
            [
                "What electorate does Anna Bligh represent?",
                "Which electorate is Anna Bligh the representative for?",
            ]
        )

        rewritten = rewrite_question(
            _FakeModel(),
            tokenizer,
            "what electorate does anna bligh represent",
            "m.topic",
        )

        self.assertEqual(
            rewritten,
            [
                "what electorate does anna bligh represent",
                "What electorate does Anna Bligh represent?",
                "Which electorate is Anna Bligh the representative for?",
            ],
        )

    def test_rewrite_question_does_not_leak_raw_mid_into_prompt(self):
        tokenizer = _BatchEncodingTokenizer()

        rewrite_question(
            _FakeModel(),
            tokenizer,
            "what electorate does anna bligh represent",
            "m.topic",
        )

        user_prompt = tokenizer.messages[1]["content"]
        self.assertNotIn("[m.topic]", user_prompt)
        self.assertIn("Original question: what electorate does anna bligh represent", user_prompt)


if __name__ == "__main__":
    unittest.main()
