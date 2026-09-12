"""验证 Prime TransferNet 训练入口参数。"""

import unittest


class PrimeTrainEntrypointTest(unittest.TestCase):
    def test_defaults_target_three_hop_training_and_five_epoch_checkpoints(self):
        # Given
        from Prime.train import build_parser

        # When
        args = build_parser().parse_args(["--input_dir", "input", "--save_dir", "ckpt"])

        # Then
        self.assertEqual(args.num_steps, 3)
        self.assertEqual(args.num_epoch, 30)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.save_every, 5)
        self.assertEqual(args.seed, 17)


if __name__ == "__main__":
    unittest.main()
