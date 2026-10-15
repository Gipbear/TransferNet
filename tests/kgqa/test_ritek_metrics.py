"""验证 RiTeK 论文口径的 Exact Match 与 ROUGE-1 指标。"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kgqa.pfit.ritek_metrics import aggregate_ritek_metrics, evaluate_predictions, score_exact_match, score_rouge1


class RiTeKMetricsTest(unittest.TestCase):
    def test_exact_match_scores_answer_entities_when_overlap_is_partial(self) -> None:
        score = score_exact_match(["alpha beta", "gamma"], ["alpha beta", "delta"])

        self.assertEqual(score.precision, 0.5)
        self.assertEqual(score.recall, 0.5)
        self.assertEqual(score.f1, 0.5)

    def test_rouge1_scores_unigram_overlap_when_entities_are_partial_matches(self) -> None:
        score = score_rouge1(["alpha beta", "gamma"], ["alpha beta", "delta"])

        self.assertAlmostEqual(score.precision, 2 / 3)
        self.assertAlmostEqual(score.recall, 2 / 3)
        self.assertAlmostEqual(score.f1, 2 / 3)

    def test_aggregate_uses_macro_average_when_samples_have_different_answer_counts(self) -> None:
        records = [
            (["alpha"], ["alpha"]),
            (["wrong"], ["beta", "gamma"]),
        ]

        metrics = aggregate_ritek_metrics(records)

        self.assertEqual(metrics.n, 2)
        self.assertEqual(metrics.exact_match.precision, 0.5)
        self.assertEqual(metrics.exact_match.recall, 0.5)
        self.assertEqual(metrics.exact_match.f1, 0.5)
        self.assertEqual(metrics.rouge1.f1, 0.5)

    def test_evaluate_uses_disambiguated_entity_ids_only_for_exact_match(self) -> None:
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "predictions.jsonl"
            input_path.write_text(
                '{"llm_pred":["entity name"],"llm_pred_disambiguated_mids":["p.42"],'
                '"golden":["p.42"]}\n',
                encoding="utf-8",
            )

            metrics = evaluate_predictions(input_path)

        self.assertEqual(metrics.exact_match.f1, 1.0)
        self.assertEqual(metrics.rouge1.f1, 0.0)

    def test_evaluate_maps_gold_ids_to_names_for_rouge1(self) -> None:
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "predictions.jsonl"
            map_path = Path(directory) / "mid2name.txt"
            input_path.write_text(
                '{"llm_pred":["magnesium sulfate"],"llm_pred_disambiguated_mids":["p.42"],'
                '"golden":["p.42"]}\n',
                encoding="utf-8",
            )
            map_path.write_text("p.42\tmagnesium sulfate\n", encoding="utf-8")

            metrics = evaluate_predictions(input_path, entity_map_path=map_path)

        self.assertEqual(metrics.exact_match.f1, 1.0)
        self.assertEqual(metrics.rouge1.f1, 1.0)


if __name__ == "__main__":
    unittest.main()
