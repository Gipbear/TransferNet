"""验证 Qwen3.5 非思考模式的零样本与微调配对评测矩阵。"""

import json
import unittest
from pathlib import Path


class Ch4Qwen35NonThinkingConfigTest(unittest.TestCase):
    def test_each_model_has_zero_shot_and_adapter_eval(self):
        # Given: 三个已完成微调的 Qwen3.5 模型及其 adapter 来源。
        root = Path(__file__).resolve().parents[2]
        config_path = root / "experiments/configs/ch4/webqsp_qwen35_non_thinking_eval_v1.json"
        expected = {
            "Qwen3.5-0.8B": "Qwen3.5-0.8B主实验",
            "Qwen3.5-2B": "Qwen3.5-2B主实验",
            "Qwen3.5-4B": "Qwen3.5-4B主实验",
        }

        # When: 读取非思考模式评测矩阵。
        config = json.loads(config_path.read_text(encoding="utf-8"))
        experiments = config["experiments"]

        # Then: 每个尺寸都有同设置的零样本和 adapter 评测项。
        self.assertEqual(len(experiments), 6)
        for model_label, adapter_from in expected.items():
            paired = [entry for entry in experiments if entry["id"].startswith(model_label)]
            self.assertEqual(len(paired), 2)
            self.assertTrue(all("--direct_answer" in entry["eval_args"] for entry in paired))
            self.assertEqual([entry.get("adapter_from") for entry in paired], [None, adapter_from])


if __name__ == "__main__":
    unittest.main()
