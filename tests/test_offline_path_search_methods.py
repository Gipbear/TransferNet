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
    select_path_candidates,
)


class OfflinePathMethodTest(unittest.TestCase):
    def test_tail_blend_method_uses_tail_blend_length_norm_score(self):
        candidate = PathCandidate(
            nodes=[1, 2, 3],
            rels=[10, 20],
            hop=2,
            base_score=-8.0,
            final_tail_score=0.25,
            tail_id=3,
            order=0,
        )

        score = compute_candidate_score(candidate, method="tail_blend", alpha_final=2.0)

        expected = (-8.0 + 2.0 * math.log(0.25 + 1e-9)) / 2
        self.assertAlmostEqual(score, expected)

    def test_baseline_method_keeps_log_norm_score_without_tail_blend(self):
        candidate = PathCandidate(
            nodes=[1, 2, 3],
            rels=[10, 20],
            hop=2,
            base_score=-8.0,
            final_tail_score=0.25,
            tail_id=3,
            order=0,
        )

        score = compute_candidate_score(candidate, method="baseline", alpha_final=2.0)

        self.assertEqual(score, -8.0)

    def test_tail_blend_selection_uses_mmr_and_lambda_val_changes_ranking(self):
        candidates = [
            PathCandidate([1, 2], [10], 1, -1.0, 0.9, 2, 0),
            PathCandidate([1, 3], [10], 1, -1.1, 0.9, 3, 1),
            PathCandidate([1, 4], [20], 1, -2.0, 0.9, 4, 2),
        ]

        no_penalty = select_path_candidates(
            candidates, k=2, method="tail_blend", alpha_final=0.0, lambda_val=0.0
        )
        penalized = select_path_candidates(
            candidates, k=2, method="tail_blend", alpha_final=0.0, lambda_val=1.0
        )

        self.assertEqual([c.order for c in no_penalty], [0, 1])
        self.assertEqual([c.order for c in penalized], [0, 2])

    def test_candidate_to_tuple_preserves_scored_value_for_metrics(self):
        candidate = PathCandidate([1, 2], [10], 1, -4.0, 0.9, 2, 0, score=-2.5)

        self.assertEqual(candidate_to_tuple(candidate), ([1, 2], [10], -2.5))

    def test_parser_exposes_only_formal_method_parameters(self):
        help_buf = io.StringIO()
        parser = build_parser()
        with redirect_stdout(help_buf):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--help"])
        help_text = help_buf.getvalue()

        self.assertIn("--method", help_text)
        self.assertIn("tail_blend", help_text)
        self.assertNotIn("--method {final,baseline}", help_text)
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

        self.assertIn("--method", wrapper)
        self.assertIn("tail_blend", wrapper)
        self.assertIn("--alpha_final", wrapper)
        self.assertIn("--lambda_val", wrapper)
        self.assertIn("--grid", wrapper)
        self.assertIn("GRID_ALPHAS", wrapper)
        self.assertIn('GRID_LAMBDAS="0 0.2 0.5 0.7 1.0"', wrapper)
        self.assertIn('GRID_BEAMS="3 5 10 15 20 30 40 50"', wrapper)
        self.assertNotIn("--scoring", wrapper)
        self.assertNotIn("--diversity", wrapper)

    def test_wrapper_defaults_keep_webqsp_artifacts_under_data_output(self):
        wrapper = (ROOT / "scripts" / "run_offline_path_search.sh").read_text(encoding="utf-8")

        self.assertIn('OFFLINE_DIR="${PROJ_DIR}/data/output/WebQSP/offline_search"', wrapper)
        self.assertIn('OUTPUT_DIR="${OFFLINE_DIR}/score_cache"', wrapper)
        self.assertIn('LOG_DIR="${OFFLINE_DIR}/logs"', wrapper)
        self.assertIn('PATHS_DIR="${OFFLINE_DIR}/paths"', wrapper)
        self.assertNotIn('OUTPUT_DIR="${PROJ_DIR}/output/score_cache"', wrapper)


if __name__ == "__main__":
    unittest.main()
