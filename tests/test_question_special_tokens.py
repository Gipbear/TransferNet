import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LLM_INFER_DIR = ROOT / "llm_infer"
if str(LLM_INFER_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_INFER_DIR))

from build_kgcot_dataset import make_sample
from kg_format import build_user_content, build_user_content_no_paths


class QuestionSpecialTokenTests(unittest.TestCase):
    def test_build_user_content_strips_question_special_tokens_when_requested(self):
        content = build_user_content(
            [],
            "[CLS] who plays ken barlow in coronation street [SEP]",
            strip_question_special_tokens=True,
        )

        self.assertIn("Question: who plays ken barlow in coronation street", content)
        self.assertNotIn("[CLS]", content)
        self.assertNotIn("[SEP]", content)

    def test_build_user_content_joins_wordpiece_markers_when_requested(self):
        content = build_user_content(
            [],
            "[CLS] what did george or ##well died of [SEP]",
            strip_question_special_tokens=True,
        )

        self.assertIn("Question: what did george orwell died of", content)
        self.assertNotIn("##", content)

    def test_no_paths_prompt_strips_question_special_tokens_when_requested(self):
        content = build_user_content_no_paths(
            "[CLS] where did kevin love go to college [SEP]",
            strip_question_special_tokens=True,
        )

        self.assertEqual(content, "Question: where did kevin love go to college")

    def test_make_sample_strips_question_special_tokens_from_training_prompt(self):
        record = {
            "question": "[CLS] who directed movie x [SEP]",
            "golden": ["director_x"],
            "mmr_reason_paths": [
                {"path": [["movie_x", "directed_by", "director_x"]], "log_score": 0.9},
            ],
        }

        sample = make_sample(
            record,
            "v2",
            shuffle=False,
            distractor_ratio=None,
            show_score=False,
            rng=random.Random(0),
            path_format="chain",
            strip_question_special_tokens=True,
        )

        self.assertIn("Question: who directed movie x", sample["messages"][1]["content"])
        self.assertNotIn("[CLS]", sample["messages"][1]["content"])
        self.assertNotIn("[SEP]", sample["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
