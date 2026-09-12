"""验证 ADInt TransferNet 训练入口参数。"""

import unittest

from ADInt.train import build_parser


class ADIntTrainEntrypointTest(unittest.TestCase):
    def test_defaults_save_every_five_epochs_for_background_training(self):
        # Given
        parser = build_parser()

        # When
        args = parser.parse_args(["--input_dir", "input", "--save_dir", "ckpt"])

        # Then
        self.assertEqual(args.num_epoch, 30)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.save_every, 5)
        self.assertEqual(args.seed, 17)


if __name__ == "__main__":
    unittest.main()
