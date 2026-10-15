"""验证服务编码已渲染聊天模板时不会重复添加 BOS。"""

import unittest
from unittest.mock import MagicMock

import torch

from kgqa.serving.llm.engine import ModelEngine


class TestPfitServerTokenization(unittest.TestCase):
    def test_rendered_chat_is_encoded_without_extra_special_tokens(self):
        engine = ModelEngine()
        engine._model = MagicMock()
        engine._model.device = "cpu"
        engine._model.generate.return_value = torch.tensor([[1, 2, 3]])
        engine._tokenizer = MagicMock()
        engine._tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])
        }
        engine._tokenizer.eos_token_id = 0
        engine._tokenizer.decode.return_value = "Answer: A"
        engine.generate_batch(["<BOS>rendered chat"], False, 256, 0.0)
        self.assertFalse(engine._tokenizer.call_args.kwargs.get("add_special_tokens", True))


if __name__ == "__main__":
    unittest.main()
