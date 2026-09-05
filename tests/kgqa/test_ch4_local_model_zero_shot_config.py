"""验证第四章本地模型零样本配对评测矩阵。"""

import json
import unittest
from pathlib import Path


class Ch4LocalModelZeroShotConfigTest(unittest.TestCase):
    def test_completed_models_have_adapter_free_eval_entries(self):
        # Given: 已完成 SFT 的本地模型及其正式加载参数。
        root = Path(__file__).resolve().parents[2]
        config_path = root / "experiments/configs/ch4/webqsp_local_models_zero_shot_v1.json"
        expected_models = {
            "Qwen3.5-0.8B零样本": ("unsloth/Qwen3.5-0.8B", "16bit", True),
            "Llama3.2-1B零样本": ("unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit", "4bit", False),
            "Qwen3.5-2B零样本": ("unsloth/Qwen3.5-2B", "16bit", True),
            "Llama3.2-3B零样本": ("unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit", "4bit", False),
            "Qwen3.5-4B零样本": ("unsloth/Qwen3.5-4B", "16bit", True),
            "Mistral-7B零样本": ("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", "4bit", False),
            "Gemma3-1B零样本": ("unsloth/gemma-3-1b-it-unsloth-bnb-4bit", "4bit", False),
        }

        # When: 读取零样本矩阵并解析每个评测项。
        config = json.loads(config_path.read_text(encoding="utf-8"))
        experiments = config["experiments"]

        # Then: 每个模型只做评测，且不声明任何 adapter 来源。
        self.assertEqual([entry["id"] for entry in experiments], list(expected_models))
        for entry in experiments:
            model, precision, text_only = expected_models[entry["id"]]
            eval_args = entry["eval_args"]
            self.assertEqual(entry["mode"], "eval_only")
            self.assertNotIn("adapter_from", entry)
            self.assertNotIn("train_args", entry)
            self.assertEqual(eval_args[-4 - int(text_only):][:4], ["--model", model, "--model_precision", precision])
            self.assertEqual("--text_only" in eval_args, text_only)


if __name__ == "__main__":
    unittest.main()
