"""验证 WebQSP 三强模型拒答微调配置文件结构与参数。"""

import json
import unittest
from pathlib import Path


class Ch4WebqspFusionRejectionConfigTest(unittest.TestCase):
    def test_fusion_rejection_config_matches_b2_and_safe_model_params(self):
        # Given: WebQSP 三强补齐拒答版本的配置路径
        root = Path(__file__).resolve().parents[2]
        config_path = root / "experiments/configs/ch4/webqsp_rejection_blocks_fusion_v1.json"
        self.assertTrue(config_path.is_file(), f"配置文件不存在: {config_path}")

        expected_experiments = {
            "Mistral-7B_排名分块_2ep": (
                "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
                "4bit",
                False,
            ),
            "Qwen3.5-4B_排名分块_2ep": (
                "unsloth/Qwen3.5-4B",
                "16bit",
                True,
            ),
        }

        # When: 读取配置
        config = json.loads(config_path.read_text(encoding="utf-8"))
        experiments = config.get("experiments", [])

        # Then: 字段与实验项完整
        self.assertEqual(config.get("dataset"), "webqsp")
        self.assertEqual(config.get("config_id"), "rejection_blocks_fusion_v1")
        self.assertEqual([e["id"] for e in experiments], list(expected_experiments))

        for entry in experiments:
            model, precision, text_only = expected_experiments[entry["id"]]
            train_args = entry["train_args"]
            eval_args = entry["eval_args"]
            build_args = entry["build_args"]

            # 构建阶段应与 B2 排名分块对齐
            self.assertIn("--block_group", build_args)
            self.assertIn("blocks", build_args)
            self.assertIn("--block_size", build_args)
            self.assertIn("20", build_args)

            # 训练阶段应设置 340 步与 group_split
            self.assertIn("--group_split", train_args)
            self.assertIn("--max_steps", train_args)
            self.assertEqual(train_args[train_args.index("--max_steps") + 1], "340")

            # 训练与评测加载参数对齐
            expected_loading = ["--model", model, "--model_precision", precision]
            self.assertEqual(train_args[:4], expected_loading)
            self.assertEqual(eval_args[-4 - int(text_only):][:4], expected_loading)
            self.assertEqual(train_args[train_args.index("--batch_size"):train_args.index("--batch_size") + 4],
                             ["--batch_size", "1", "--grad_accum", "32"])
            self.assertIn("--batch_size", eval_args)
            self.assertEqual("--text_only" in train_args, text_only)
            self.assertEqual("--text_only" in eval_args, text_only)


if __name__ == "__main__":
    unittest.main()
