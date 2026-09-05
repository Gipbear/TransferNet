"""验证第四章本地模型串行实验矩阵。"""

import json
import unittest
from pathlib import Path


class Ch4LocalModelSweepConfigTest(unittest.TestCase):
    def test_pending_models_have_safe_training_and_eval_args(self):
        # Given: 本地已下载、但尚未完成正式实验的模型集合。
        root = Path(__file__).resolve().parents[2]
        config_path = root / "experiments/configs/ch4/webqsp_local_models_v1.json"
        expected_models = {
            "Llama3.2-1B主实验": ("unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit", "4bit", False),
            "Gemma3-1B主实验": ("unsloth/gemma-3-1b-it-unsloth-bnb-4bit", "4bit", False),
            "Llama3.2-3B主实验": ("unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit", "4bit", False),
            "Qwen3.5-4B主实验": ("unsloth/Qwen3.5-4B", "16bit", True),
            "Llama3.1-8B-Dynamic4bit": (
                "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit", "4bit", False,
            ),
            "Qwen2.5-7B主实验": ("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "4bit", False),
            "Mistral-7B主实验": ("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", "4bit", False),
        }

        # When: 读取正式矩阵并解析每个实验的加载参数。
        config = json.loads(config_path.read_text(encoding="utf-8"))
        experiments = config["experiments"]

        # Then: 模型完整覆盖，训练和评测使用一致精度，且采用安全的单样本批次。
        self.assertEqual([entry["id"] for entry in experiments], list(expected_models))
        for entry in experiments:
            model, precision, text_only = expected_models[entry["id"]]
            train_args = entry["train_args"]
            eval_args = entry["eval_args"]
            expected_loading = ["--model", model, "--model_precision", precision]
            self.assertEqual(train_args[:4], expected_loading)
            self.assertEqual(eval_args[-4 - int(text_only):][:4], expected_loading)
            self.assertEqual(train_args[-4:], ["--batch_size", "1", "--grad_accum", "32"])
            self.assertIn("--batch_size", eval_args)
            self.assertEqual("--text_only" in train_args, text_only)
            self.assertEqual("--text_only" in eval_args, text_only)


if __name__ == "__main__":
    unittest.main()
