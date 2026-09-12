"""验证 TransferNet 的推理步数可由数据集入口配置。"""

import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)


class TransferNetStepTest(unittest.TestCase):
    def test_constructor_builds_three_step_modules_when_requested(self):
        # Given
        from PharmKG.model import TransferNet

        args = Namespace(bert_name="unused", num_steps=3)
        triples = torch.tensor([[0, 0, 1], [1, 0, 2]], dtype=torch.long)

        # When
        with patch("PharmKG.model.from_pretrained_local_first", return_value=TinyEncoder()):
            model = TransferNet(args, {"0": 0, "1": 1, "2": 2}, {"r": 0}, triples)

        # Then
        self.assertEqual(model.num_steps, 3)
        self.assertEqual(len(model.step_encoders), 3)
        self.assertEqual(model.hop_selector.out_features, 3)


if __name__ == "__main__":
    unittest.main()
