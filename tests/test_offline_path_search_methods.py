import io
import math
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.offline_path_search import (
    PathCandidate,
    build_parser,
    candidate_to_tuple,
    compute_candidate_score,
    run_experiment,
    select_path_candidates,
)


class OfflinePathMethodTest(unittest.TestCase):
    def test_candidate_score_uses_tail_blend_length_norm_score(self):
        candidate = PathCandidate(
            nodes=[1, 2, 3],
            rels=[10, 20],
            hop=2,
            base_score=-8.0,
            final_tail_score=0.25,
            order=0,
        )

        score = compute_candidate_score(candidate, alpha_final=2.0)

        expected = (-8.0 + 2.0 * math.log(0.25 + 1e-9)) / 2
        self.assertAlmostEqual(score, expected)

    def test_tail_blend_selection_uses_mmr_and_lambda_val_changes_ranking(self):
        candidates = [
            PathCandidate([1, 2], [10], 1, -1.0, 0.9, order=0),
            PathCandidate([1, 3], [10], 1, -1.1, 0.9, order=1),
            PathCandidate([1, 4], [20], 1, -2.0, 0.9, order=2),
        ]

        no_penalty = select_path_candidates(
            candidates, k=2, alpha_final=0.0, lambda_val=0.0
        )
        penalized = select_path_candidates(
            candidates, k=2, alpha_final=0.0, lambda_val=1.0
        )

        self.assertEqual([c.order for c in no_penalty], [0, 1])
        self.assertEqual([c.order for c in penalized], [0, 2])

    def test_candidate_to_tuple_preserves_scored_value_for_metrics(self):
        candidate = PathCandidate([1, 2], [10], 1, -4.0, 0.9, order=0, score=-2.5)

        self.assertEqual(candidate_to_tuple(candidate), ([1, 2], [10], -2.5))

    def test_run_experiment_uses_sample_triples_for_cwq_cache(self):
        import torch

        cache = {
            "version": 1,
            "meta": {
                "dataset": "CWQ",
                "split": "val",
                "topk_entities": 1,
                "id2ent": {0: "topic", 1: "answer"},
                "id2rel": {0: "rel"},
            },
            "samples": [{
                "question": "dummy",
                "topic_ids": [0],
                "gold_ids": [1],
                "triples": [[0, 0, 1]],
                "hop_attn": torch.tensor([1.0]),
                "rel_probs": [torch.tensor([0.9])],
                "ent_indices": [torch.tensor([1])],
                "ent_scores": [torch.tensor([0.8])],
                "e_score_indices": torch.tensor([1]),
                "e_score_values": torch.tensor([0.9]),
            }],
        }

        metrics = run_experiment(
            cache,
            valid_edges_dict=None,
            threshold=0.01,
            beam_size=1,
        )

        self.assertEqual(metrics["empty_path"], 0)
        self.assertEqual(metrics["answer_hit_rate"], 1.0)

    def test_parser_exposes_only_formal_method_parameters(self):
        help_buf = io.StringIO()
        parser = build_parser()
        with redirect_stdout(help_buf):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--help"])
        help_text = help_buf.getvalue()

        self.assertNotIn("--method", help_text)
        self.assertNotIn("final 为", help_text)
        self.assertIn("--alpha_final", help_text)
        self.assertIn("--lambda_val", help_text)
        for removed in [
            "--candidate_hops",
            "--score_mode",
            "--selector",
            "--gamma_hop",
            "--tail_budget",
            "--length_norm",
            "--scoring",
            "--diversity",
        ]:
            self.assertNotIn(removed, help_text)

    def test_wrapper_supports_tail_blend_grid_without_old_search_knobs(self):
        wrapper = (ROOT / "scripts" / "run_offline_path_search.sh").read_text(encoding="utf-8")

        self.assertIn("--dataset", wrapper)
        self.assertIn("kgqa.cli.dump_scores", wrapper)
        self.assertNotIn("CompWebQ.dump_scores", wrapper)
        self.assertNotIn("WebQSP.dump_scores", wrapper)
        self.assertNotIn("--method", wrapper)
        self.assertIn("--alpha_final", wrapper)
        self.assertIn("--lambda_val", wrapper)
        self.assertIn("--grid", wrapper)
        self.assertIn("SUMMARY_FILE", wrapper)
        self.assertIn("relation_jaccard_diversity", wrapper)
        self.assertIn("relation_coverage", wrapper)
        self.assertIn("GRID_ALPHAS", wrapper)
        self.assertIn('GRID_LAMBDAS="0 0.2 0.5 0.7 1.0"', wrapper)
        self.assertIn('GRID_BEAMS="3 5 10 15 20 30 40 50"', wrapper)
        self.assertNotIn("--scoring", wrapper)
        self.assertNotIn("--diversity", wrapper)

    def test_wrapper_defaults_keep_dataset_artifacts_under_data_output(self):
        wrapper = (ROOT / "scripts" / "run_offline_path_search.sh").read_text(encoding="utf-8")

        self.assertIn('OFFLINE_DIR="${PROJ_DIR}/data/output/WebQSP/offline_search"', wrapper)
        self.assertIn('OFFLINE_DIR="${PROJ_DIR}/data/output/CWQ/offline_search"', wrapper)
        self.assertIn('OUTPUT_DIR="${OFFLINE_DIR}/score_cache"', wrapper)
        self.assertIn('LOG_DIR="${OFFLINE_DIR}/logs"', wrapper)
        self.assertIn('PATHS_DIR="${OFFLINE_DIR}/paths"', wrapper)
        self.assertIn('SUMMARY_FILE="${OFFLINE_DIR}/summary.csv"', wrapper)
        self.assertNotIn('OUTPUT_DIR="${PROJ_DIR}/output/score_cache"', wrapper)


if __name__ == "__main__":
    unittest.main()
