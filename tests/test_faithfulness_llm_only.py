"""LLM 忠实度指标(hallucination)只能在 LLM 实际产出的答案上算,不能混入
large_answer_expansion 等确定性 pipeline 后处理补出的实体。"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kgqa.agent.common.metrics import (
    cited_indices_for_answers,
    compute_faithfulness,
    label_golden_indices,
    llm_produced_answers,
)


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


class CitedIndicesForAnswersTests(unittest.TestCase):
    """引用集合必须与最终(校准后)答案对齐:被精确性过滤剔除的尾实体,
    对应路径不再作为答案证据,应离开 citation_accuracy 的分母。"""

    @staticmethod
    def _paths():
        return [
            {"path": [["q", "r1", "m.gold"]]},    # idx 1, 尾实体为黄金答案且保留
            {"path": [["q", "r2", "m.pruned"]]},  # idx 2, 被精确性过滤剪掉
            {"path": [["q", "r3", "m.keep"]]},    # idx 3, 尾实体保留在最终答案
        ]

    def test_drops_cited_paths_whose_tail_left_final_answers(self):
        kept = cited_indices_for_answers({1, 2, 3}, self._paths(), ["m.gold", "m.keep"])
        self.assertEqual(kept, {1, 3})

    def test_no_pruning_keeps_all(self):
        kept = cited_indices_for_answers(
            {1, 2, 3}, self._paths(), ["m.gold", "m.pruned", "m.keep"]
        )
        self.assertEqual(kept, {1, 2, 3})

    def test_norm_case_insensitive(self):
        kept = cited_indices_for_answers({1}, [{"path": [["q", "r", "M.Gold"]]}], ["m.gold"])
        self.assertEqual(kept, {1})

    def test_out_of_range_index_ignored(self):
        kept = cited_indices_for_answers({1, 99}, self._paths(), ["m.gold"])
        self.assertEqual(kept, {1})

    def test_corrects_citation_accuracy_denominator(self):
        # idx2 尾实体既非黄金答案、又被剪枝:旧口径把它算进分母(cite=2/3),
        # 对齐口径剔除后(cite=2/2=1.0)。
        paths = self._paths()
        golden = label_golden_indices(paths, ["m.gold", "m.keep"])  # {1, 3}
        cited = {1, 2, 3}
        old = compute_faithfulness(cited, golden, [], set())["citation_accuracy"]
        aligned = cited_indices_for_answers(cited, paths, ["m.gold", "m.keep"])
        new = compute_faithfulness(aligned, golden, [], set())["citation_accuracy"]
        self.assertAlmostEqual(old, 2 / 3, places=4)
        self.assertAlmostEqual(new, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
