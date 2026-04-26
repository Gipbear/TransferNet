import sys
import threading
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_answer_check_prompt import EvalJob, build_arg_parser, format_duration, format_progress_bar, run_jobs_with_concurrency


class EvalAnswerCheckPromptTests(unittest.TestCase):
    def test_parser_accepts_underscored_concurrent_request_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--concurrent_requests", "2"])
        self.assertEqual(args.concurrent_requests, 2)

    def test_format_progress_bar_renders_expected_summary(self):
        rendered = format_progress_bar(3, 10, width=10)
        self.assertEqual(rendered, "[###-------] 3/10 (30.0%)")

    def test_format_duration_renders_short_and_long_values(self):
        self.assertEqual(format_duration(9.4), "9s")
        self.assertEqual(format_duration(75), "1m15s")
        self.assertEqual(format_duration(3671), "1h01m11s")

    def test_run_jobs_with_concurrency_preserves_order_and_caps_parallelism(self):
        jobs = [
            EvalJob(index=idx, rec={}, question=f"q{idx}", pred_answers=[], paths_text="", required_max_new_tokens=256)
            for idx in range(1, 6)
        ]
        active = 0
        max_active = 0
        lock = threading.Lock()

        def worker(job, barrier, batch):
            nonlocal active, max_active
            if barrier is not None:
                barrier.wait()
            self.assertLessEqual(len(batch), 3)
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return job.index

        results = run_jobs_with_concurrency(jobs, 3, worker)

        self.assertEqual(results, [1, 2, 3, 4, 5])
        self.assertEqual(max_active, 3)

    def test_run_jobs_with_concurrency_falls_back_to_sequential_mode(self):
        jobs = [
            EvalJob(index=idx, rec={}, question=f"q{idx}", pred_answers=[], paths_text="", required_max_new_tokens=256)
            for idx in range(1, 4)
        ]
        active = 0
        max_active = 0
        lock = threading.Lock()

        def worker(job, barrier, batch):
            nonlocal active, max_active
            self.assertIsNone(barrier)
            self.assertEqual(len(batch), 1)
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return job.index

        results = run_jobs_with_concurrency(jobs, 1, worker)

        self.assertEqual(results, [1, 2, 3])
        self.assertEqual(max_active, 1)


if __name__ == "__main__":
    unittest.main()
