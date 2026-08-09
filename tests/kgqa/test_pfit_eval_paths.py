"""pfit eval 路径预算截断(truncate_paths_by_score)单元测试。"""
import unittest

from kgqa.pfit.eval import truncate_paths_by_score


class TestTruncatePathsByScore(unittest.TestCase):
    def _paths(self, scores):
        return [{"path": [["a", "r", f"t{i}"]], "log_score": s}
                for i, s in enumerate(scores)]

    def test_keeps_top_k_by_log_score_even_if_input_unsorted(self):
        paths = self._paths([-12.0, -9.0, -15.0])
        out = truncate_paths_by_score(paths, 2)
        self.assertEqual([p["log_score"] for p in out], [-9.0, -12.0])

    def test_non_positive_max_paths_keeps_all(self):
        paths = self._paths([-9.0, -10.0])
        self.assertEqual(len(truncate_paths_by_score(paths, 0)), 2)
        self.assertEqual(len(truncate_paths_by_score(paths, -1)), 2)

    def test_max_paths_beyond_length_keeps_all(self):
        paths = self._paths([-9.0, -10.0])
        self.assertEqual(len(truncate_paths_by_score(paths, 5)), 2)

    def test_does_not_mutate_input(self):
        paths = self._paths([-9.0])
        original = list(paths)
        truncate_paths_by_score(paths, 1)
        self.assertEqual(paths, original)


if __name__ == "__main__":
    unittest.main()
