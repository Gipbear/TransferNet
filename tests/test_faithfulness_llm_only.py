"""LLM 忠实度指标(hallucination)只能在 LLM 实际产出的答案上算,不能混入
large_answer_expansion 等确定性 pipeline 后处理补出的实体。"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.common.metrics import compute_faithfulness, llm_produced_answers


class LlmProducedAnswersTests(unittest.TestCase):
    def test_excludes_expansion_补出的实体(self):
        # mc 由 expansion 补出 → 剔除;LLM 产出的 A、B 保留
        self.assertEqual(
            llm_produced_answers(["A", "B", "C"], ["ma", "mb", "mc"], ["mc"]),
            ["A", "B"],
        )

    def test_no_expansion_保留全部(self):
        self.assertEqual(
            llm_produced_answers(["A", "B"], ["ma", "mb"], []),
            ["A", "B"],
        )

    def test_all_expanded_返回空(self):
        self.assertEqual(llm_produced_answers(["A"], ["ma"], ["ma"]), [])

    def test_norm_不区分大小写(self):
        self.assertEqual(
            llm_produced_answers(["A", "B"], ["M.A", "m.b"], ["m.a"]),
            ["B"],
        )


class FaithfulnessExcludesExpansionTests(unittest.TestCase):
    def test_hallucination_只算_llm_答案(self):
        # 最终答案 = LLM 产出 [Spain, USA](都在路径里)+ expansion 补出 [Belize](路径外)
        # 旧口径:Belize 算幻觉 → rate>0;新口径:只看 LLM 答案 → rate=0
        pred_names = ["Spain", "United States", "Belize"]
        pred_mids = ["m.spain", "m.usa", "m.belize"]
        expanded_mids = ["m.belize"]
        path_entities = {"spain", "united states"}

        llm_answers = llm_produced_answers(pred_names, pred_mids, expanded_mids)
        faith = compute_faithfulness(
            cited_indices={1, 2},
            golden_indices={1, 2},
            pred_answers=llm_answers,
            path_entities=path_entities,
        )
        self.assertEqual(faith["hallucination_rate"], 0.0)
        self.assertEqual(faith["hallucinated_entities"], [])


if __name__ == "__main__":
    unittest.main()
