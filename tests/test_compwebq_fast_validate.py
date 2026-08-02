import unittest
from unittest.mock import patch

import torch

from CompWebQ.predict import validate


class _Top1Model(torch.nn.Module):
    def forward(self, *batch):
        return {"e_score": torch.tensor([[0.1, 0.9], [0.8, 0.2]])}


class TestCompWebQFastValidate(unittest.TestCase):
    def test_fast_mode_only_computes_top1_accuracy(self):
        batch = (
            torch.zeros(2, 2), {"input_ids": torch.tensor([[1], [2]])},
            torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            (torch.tensor([[0, 0, 0]]), torch.tensor([[0, 0, 0]])),
            torch.zeros(2, 2), torch.tensor([0, 1]),
        )
        with patch("CompWebQ.predict.mmr_diversity_beam_search") as mmr, \
                patch("CompWebQ.predict.open", create=True) as output_file:
            accuracy = validate(None, _Top1Model(), [batch], "cpu", fast=True)

        self.assertEqual(accuracy, 0.5)
        mmr.assert_not_called()
        output_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
