"""验证 CWQ 比较属性链的文本选择与独立评测。"""
from __future__ import annotations

import unittest

from kgqa.retrieve.cwq_attribute_selector import (
    ComparisonQuestion,
    evaluate_rankings,
    rank_attribute_chains,
    rank_by_frequency,
)


class TestCwqAttributeSelector(unittest.TestCase):
    def test_ranks_distinct_chains_from_question_text(self):
        # Given: 同一属性链在训练集中有两种问法，另一条链也已出现。
        train = [
            ComparisonQuestion("Which athletes are taller than 2 meters?", "comparative", ("height",)),
            ComparisonQuestion("Who has the greatest height?", "superlative", ("height",)),
            ComparisonQuestion("Which athletes were born before 2000?", "comparative", ("birth",)),
        ]

        # When: 只用待预测问题的文本检索属性链。
        rankings = rank_attribute_chains(train, ["Which athletes were born after 1990?"], "word")

        # Then: 首位是出生属性，返回的候选链不重复。
        self.assertEqual(rankings[0].chains[0], ("birth",))
        self.assertEqual(len(rankings[0].chains), 2)
        self.assertEqual(len(set(rankings[0].chains)), 2)

    def test_evaluates_seen_and_unseen_labels_separately(self):
        # Given: 训练集不含 date 链，开发集有两道同类型题。
        train = [ComparisonQuestion("Which people are taller?", "comparative", ("height",))]
        dev = [
            ComparisonQuestion("Who is taller?", "comparative", ("height",)),
            ComparisonQuestion("Who is older?", "comparative", ("date",)),
        ]

        # When: 评测两题的 Top-1/Top-3。
        rankings = rank_attribute_chains(train, [item.question for item in dev], "char_wb")
        result = evaluate_rankings(train, dev, rankings)

        # Then: 只把 seen-label 题计入可达上界，未见链不能被误报为命中。
        self.assertEqual(result.overall.sample_count, 2)
        self.assertEqual(result.overall.seen_label_count, 1)
        self.assertEqual(result.overall.top1_hit_count, 1)
        self.assertEqual(result.overall.top3_hit_count, 1)
        self.assertEqual(result.comparative.seen_top1_hit_count, 1)
        self.assertEqual(result.superlative.sample_count, 0)

    def test_frequency_baseline_uses_train_labels_only(self):
        # Given: height 在训练问题里出现两次，birth 出现一次。
        train = [
            ComparisonQuestion("tall", "comparative", ("height",)),
            ComparisonQuestion("short", "comparative", ("height",)),
            ComparisonQuestion("old", "comparative", ("birth",)),
        ]

        # When: 构建不看开发问题的高频标签基线。
        ranking = rank_by_frequency(train, 3)

        # Then: 仅按训练频次排序且不重复。
        self.assertEqual(ranking.chains, (("height",), ("birth",)))


if __name__ == "__main__":
    unittest.main()
